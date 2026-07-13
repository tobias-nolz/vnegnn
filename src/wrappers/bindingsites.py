from typing import Dict, List

import torch
import torchmetrics
from einops._torch_specific import allow_ops_in_compiled_graph
from hydra.utils import instantiate
from torch import Tensor, nn
from torch_geometric.nn.pool import knn
from torchmetrics import AUROC, Accuracy, JaccardIndex

from src.modules.losses import ConfidenceLoss, DiceLoss
from src.modules.metrics import DCA, DCC
from src.utils.misc import calc_group_var, multi_predictions
from src.wrappers.base import WrapperBase

allow_ops_in_compiled_graph()


class BindingSitesLoss(nn.Module):
    def __init__(
        self,
        segmentation_loss: nn.Module = DiceLoss(),
        global_node_pos_loss: nn.Module = nn.HuberLoss(),
        confidence_loss: nn.Module = ConfidenceLoss(),
        classification_loss: nn.Module = None,
        segmentation_loss_weight: float = 1.0,
        global_node_pos_loss_weight: float = 1.0,
        confidence_loss_weight: float = 1.0,
        classification_loss_weight: float = 1.0,
        classification_foreground_dist: float = None,
    ):
        super().__init__()
        self.segmentation_loss = segmentation_loss
        self.global_node_pos_loss = global_node_pos_loss
        self.confidence_loss = confidence_loss
        # Binary head supervising orthosteric (0) vs allosteric (1) sites. Default to
        # BCEWithLogitsLoss; a config may pass one with a `pos_weight` for imbalance.
        self.classification_loss = (
            classification_loss
            if classification_loss is not None
            else nn.BCEWithLogitsLoss()
        )
        self.segmentation_loss_weight = segmentation_loss_weight
        self.global_node_pos_loss_weight = global_node_pos_loss_weight
        self.confidence_loss_weight = confidence_loss_weight
        self.classification_loss_weight = classification_loss_weight
        # If set, only virtual nodes whose nearest annotated site is within this
        # distance (Angstrom) contribute to the classification loss. Background nodes
        # far from any real site do not belong to any class, so supervising them adds
        # noise. `None` supervises every virtual node.
        self.classification_foreground_dist = classification_foreground_dist

    def forward(self, model, batch):
        pred_seg, pred_pos_global_node, _, preds_confidence, preds_class = model(batch)

        seg_loss = self.segmentation_loss(pred_seg.squeeze(), batch["atom"].y)

        x = pred_pos_global_node
        y = batch["atom"].bindingsite_center
        x_batch = batch["global_node"].batch
        y_batch = batch["atom"]["bindingsite_center_batch"]

        assign_index = knn(
            x=x,
            y=y,
            batch_x=x_batch,
            batch_y=y_batch,
            k=1,
        )

        dists = torch.norm(y[assign_index[0]] - x[assign_index[1]], dim=-1)
        global_node_pos_loss = self.global_node_pos_loss(
            y[assign_index[0]], x[assign_index[1]]
        )
        pos_var = calc_group_var(x, x_batch).mean()
        conf_var = calc_group_var(preds_confidence, x_batch).mean()

        confidence_assign_index = knn(
            x=y,
            y=x,
            batch_x=y_batch,
            batch_y=x_batch,
            k=1,
        )
        dists_confidence = torch.norm(
            y[confidence_assign_index[1]] - x[confidence_assign_index[0]], dim=-1
        ).detach()

        confidence_loss = self.confidence_loss(
            dists_confidence,
            preds_confidence.squeeze(),
        )

        # Per-virtual-node classification target: label each virtual node with the site
        # type (0 = orthosteric, 1 = allosteric) of its NEAREST annotated site center,
        # rather than broadcasting one protein-level label. `confidence_assign_index`
        # (computed above) maps each virtual node -> its nearest center, in the same
        # order as `dists_confidence` and `preds_*.squeeze()`.
        site_types = batch["atom"].bindingsite_site_type
        class_target = site_types[confidence_assign_index[1]].float()
        preds_class_flat = preds_class.squeeze()

        if self.classification_foreground_dist is not None:
            foreground = dists_confidence <= self.classification_foreground_dist
            if foreground.any():
                class_loss = self.classification_loss(
                    preds_class_flat[foreground], class_target[foreground]
                )
            else:
                # No virtual node is close to a real site in this batch; contribute a
                # differentiable zero so the graph/optimizer stay well-defined.
                class_loss = preds_class_flat.sum() * 0.0
        else:
            class_loss = self.classification_loss(preds_class_flat, class_target)

        loss_dict = {
            "dist": dists.mean(),
            "pos_loss": global_node_pos_loss,
            "pos_var": pos_var,
            "seg_loss": seg_loss,
            "confidence_loss": confidence_loss,
            "confidence_var": conf_var,
            "class_loss": class_loss,
            "loss": global_node_pos_loss * self.global_node_pos_loss_weight
            + seg_loss * self.segmentation_loss_weight
            + confidence_loss * self.confidence_loss_weight
            + class_loss * self.classification_loss_weight,
        }

        return loss_dict, (
            pred_seg,
            pred_pos_global_node,
            preds_confidence,
            preds_class,
        )


class BindingSitesWrapper(WrapperBase):
    """Wrapper that coordinates model, sampling, and training."""

    backbone: nn.Module

    # Allow loading checkpoints trained before the classification head existed
    # (the new classifier_mlp weights are simply left at their initialization).
    strict_loading = False

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.backbone = instantiate(self.hparams.backbone)
        self.backbone = torch.compile(
            self.backbone,
            disable=not self.hparams.compile,
            fullgraph=True,
            dynamic=True,
        )

        self.loss = instantiate(self.hparams.loss)
        metrics = torchmetrics.MetricCollection(
            {
                "acc": Accuracy(task="binary"),
                "auroc": AUROC(task="binary"),
                "iou": JaccardIndex(task="binary"),
            }
        )
        self.train_seg_metrics = metrics.clone(prefix="train/")
        self.val_seg_metrics = metrics.clone(prefix="val/")

        class_metrics = torchmetrics.MetricCollection(
            {
                "class_acc": Accuracy(task="binary"),
                "class_auroc": AUROC(task="binary"),
            }
        )
        self.val_class_metrics = class_metrics.clone(prefix="val/")
        # Same classifier metrics but only over virtual nodes that are successful
        # detections (within the DCC threshold of a true site): "given a found pocket,
        # is its type correct".
        self.val_class_metrics_detected = class_metrics.clone(
            prefix="val/", postfix="_detected"
        )

        threshold = self.hparams.threshold
        self.val_dcc = DCC(threshold=threshold)
        self.val_dca = DCA(threshold=threshold)
        self.val_dcc_ranked = DCC(threshold=threshold)
        self.val_dca_ranked = DCA(threshold=threshold)
        self.val_dcc_rand_ranked = DCC(threshold=threshold)
        self.val_dca_rand_ranked = DCA(threshold=threshold)

    def forward(self, batch: Dict[str, Tensor]) -> Tensor:
        x_atom = batch.x_dict["atom"]
        pos_atom = batch.pos_dict["atom"] / self.hparams.scaling_factor
        x_global_node = batch.x_dict["global_node"]
        pos_global_node = batch.pos_dict["global_node"] / self.hparams.scaling_factor
        edge_index_atom_atom = batch.edge_index_dict[("atom", "to", "atom")]
        edge_index_atom_global_node = batch.edge_index_dict[
            ("atom", "to", "global_node")
        ]
        edge_index_global_node_atom = batch.edge_index_dict[
            ("global_node", "to", "atom")
        ]

        x_atom, pos_global_node, x_global_node, confidence_out, class_out = self.backbone(
            x_atom,
            pos_atom,
            x_global_node,
            pos_global_node,
            edge_index_atom_atom,
            edge_index_atom_global_node,
            edge_index_global_node_atom,
        )

        pos_global_node = pos_global_node * self.hparams.scaling_factor
        return x_atom, pos_global_node, x_global_node, confidence_out, class_out

    def model_step(self, batch: Dict[str, Tensor]) -> tuple[Dict[str, Tensor], Tensor]:
        loss_dict, preds = self.loss(model=self, batch=batch)
        return loss_dict, preds

    def training_step(self, batch: Dict[str, Tensor]) -> Tensor:
        loss, _ = self.model_step(batch)
        self.log_dict(
            {f"train/{k}": v for k, v in loss.items()},
            prog_bar=True,
            sync_dist=True,
            batch_size=batch["global_node"].batch.unique().numel(),
        )
        return loss

    def validation_step(self, batch: Dict[str, Tensor]) -> Tensor:
        loss, preds = self.model_step(batch)
        self.log_dict(
            {f"val/{k}": v for k, v in loss.items()},
            prog_bar=True,
            sync_dist=True,
            batch_size=batch["global_node"].batch.unique().numel(),
        )

        pred_seg, pred_pos_global_node, preds_confidence, preds_class = preds

        self.val_seg_metrics(pred_seg.squeeze(), batch["atom"].y)
        self.log_dict(self.val_seg_metrics, on_step=False, on_epoch=True)

        preds_pos_global_node = pred_pos_global_node
        batch_global_nodes = batch["global_node"].batch

        # Per-site classification metric: label each virtual node with the site type of
        # its nearest annotated center (same scheme as the training loss).
        vn_nearest_center = knn(
            x=batch["atom"].bindingsite_center,
            y=preds_pos_global_node,
            batch_x=batch["atom"]["bindingsite_center_batch"],
            batch_y=batch_global_nodes,
            k=1,
        )
        vn_center_dist = torch.norm(
            preds_pos_global_node[vn_nearest_center[0]]
            - batch["atom"].bindingsite_center[vn_nearest_center[1]],
            dim=-1,
        )
        class_target = batch["atom"].bindingsite_site_type[vn_nearest_center[1]].long()
        preds_class_flat = preds_class.squeeze()

        fg_dist = getattr(self.loss, "classification_foreground_dist", None)
        if fg_dist is not None:
            foreground = vn_center_dist <= fg_dist
            if foreground.any():
                self.val_class_metrics(
                    preds_class_flat[foreground], class_target[foreground]
                )
        else:
            self.val_class_metrics(preds_class_flat, class_target)
        self.log_dict(self.val_class_metrics, on_step=False, on_epoch=True)

        # Detection-conditioned classifier metric: only virtual nodes that actually
        # landed on a true site (within the DCC threshold) count.
        detected = vn_center_dist <= self.hparams.threshold
        if detected.any():
            self.val_class_metrics_detected(
                preds_class_flat[detected], class_target[detected]
            )
            self.log_dict(self.val_class_metrics_detected, on_step=False, on_epoch=True)

        binding_site_center = batch["atom"].bindingsite_center
        self.val_dcc(
            coords_global_nodes=preds_pos_global_node,
            coords_bindingsites=binding_site_center,
            batch_global_nodes=batch_global_nodes,
            batch_bindingsites=batch["atom"]["bindingsite_center_batch"],
        )
        self.val_dca(
            coords_global_nodes=preds_pos_global_node,
            coords_ligands=batch["ligand"].ligand_coords,
            ligand_ids=batch["ligand"].ligand_ids,
            batch_global_nodes=batch_global_nodes,
            batch_ligands=batch["ligand"].ligand_coords_batch,
        )

        self.val_dcc_ranked(
            coords_global_nodes=preds_pos_global_node,
            coords_bindingsites=binding_site_center,
            batch_global_nodes=batch_global_nodes,
            batch_bindingsites=batch["atom"]["bindingsite_center_batch"],
            global_node_confidence=preds_confidence,
        )

        self.val_dca_ranked(
            coords_global_nodes=preds_pos_global_node,
            coords_ligands=batch["ligand"].ligand_coords,
            ligand_ids=batch["ligand"].ligand_ids,
            batch_global_nodes=batch_global_nodes,
            batch_ligands=batch["ligand"].ligand_coords_batch,
            global_node_confidence=preds_confidence,
        )
        rand_confs = torch.rand_like(preds_confidence)
        self.val_dcc_rand_ranked(
            coords_global_nodes=preds_pos_global_node,
            coords_bindingsites=binding_site_center,
            batch_global_nodes=batch_global_nodes,
            batch_bindingsites=batch["atom"]["bindingsite_center_batch"],
            global_node_confidence=rand_confs,
        )
        self.val_dca_rand_ranked(
            coords_global_nodes=preds_pos_global_node,
            coords_ligands=batch["ligand"].ligand_coords,
            ligand_ids=batch["ligand"].ligand_ids,
            batch_global_nodes=batch_global_nodes,
            batch_ligands=batch["ligand"].ligand_coords_batch,
            global_node_confidence=rand_confs,
        )

        self.log(
            "val/dcc",
            self.val_dcc,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val/dca",
            self.val_dca,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val/dcc_ranked",
            self.val_dcc_ranked,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val/dca_ranked",
            self.val_dca_ranked,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val/dcc_rand_ranked",
            self.val_dcc_rand_ranked,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val/dca_rand_ranked",
            self.val_dca_rand_ranked,
            on_step=False,
            on_epoch=True,
        )

    def predict_step(
        self, batch: Dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> List[Dict[str, Tensor]]:
        (
            pred_pos,
            pred_conf,
            pred_class,
            batch_global_nodes,
            init_pos,
        ) = multi_predictions(
            model=self,
            batch=batch,
            num_cycles=self.hparams.pred_cycles,
        )

        coords_global_nodes = pred_pos
        allosteric_prob = torch.sigmoid(pred_class)
        protein_names = batch["protein_name"]

        batch_outputs: List[Dict[str, Tensor]] = []
        for batch_global_node, i in enumerate(batch_global_nodes.unique()):
            s_coords = coords_global_nodes[batch_global_node == batch_global_nodes]
            s_confs = pred_conf[batch_global_node == batch_global_nodes]
            s_allo = allosteric_prob[batch_global_node == batch_global_nodes]

            output_dict = {
                "protein_name": [protein_names[i]] * s_coords.shape[0],
                "coords": s_coords,
                "confidence": s_confs,
                "allosteric_prob": s_allo,
            }

            if self.hparams.save_vn_initial_pos:
                output_dict["vn_initial_pos"] = init_pos[
                    batch_global_node == batch_global_nodes
                ]
            batch_outputs.append(output_dict)

        return batch_outputs
