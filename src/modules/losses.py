import torch
from torch import Tensor
from torch_geometric.nn.pool import knn
from torch_geometric.utils import scatter
from torch_geometric.utils import softmax as group_softmax


def gompertz(x: Tensor, a: float, b: float, c: float) -> Tensor:
    return a * torch.exp(-b * torch.exp(-c * x))


class HuberLoss(torch.nn.Module):
    def __init__(self, delta: float = 1.0, scaling_factor: float = 1.0):
        super(HuberLoss, self).__init__()
        self.loss = torch.nn.HuberLoss(delta=delta / scaling_factor)

    def forward(self, pred: Tensor, target: Tensor):
        return self.loss(pred, target)


class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1):
        """Dice loss.

        Args:
            smooth (int, optional): The smoothing factor for dice loss. Defaults to 1.
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, input, targets):
        probs = torch.sigmoid(input)

        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        dice_coef = (2.0 * intersection + self.smooth) / (
            probs_flat.sum() + targets_flat.sum() + self.smooth
        )
        return 1 - dice_coef


class ConfidenceLoss(torch.nn.Module):
    def __init__(self, gamma=4, c0=0.001):
        super(ConfidenceLoss, self).__init__()
        self.c0 = c0
        self.gamma = gamma
        self.loss = torch.nn.MSELoss()

    def forward(
        self,
        dists: Tensor,
        confs: Tensor,
    ):
        c = dists.detach().clone()
        c[c <= self.gamma] = 1 - c[c <= self.gamma] / (self.gamma * 2)
        c[c > self.gamma] = self.c0
        return self.loss(confs, c)


def group_standardize(x: Tensor, batch: Tensor, eps: float = 1e-6) -> Tensor:
    """Per-protein z-score (mean 0, std 1 within each ``batch`` group).

    Standardising inside each protein makes the confidence and class signals
    scale-free before they are combined, so the class-aware ranking score matches the
    z-scored blend validated by the no-training re-rank probe without a hand-tuned
    logit scale.
    """
    x = x.float()
    mean = scatter(x, batch, dim=0, reduce="mean")
    xc = x - mean[batch]
    var = scatter(xc * xc, batch, dim=0, reduce="mean")
    std = (var + eps).sqrt()
    return xc / std[batch]


def class_aware_ranking_score(
    confs: Tensor,
    class_logits: Tensor,
    gamma: Tensor,
    batch: Tensor,
    class_weight: float = 1.0,
) -> Tensor:
    """Combined ranking scalar ``z(conf) + w * gamma * z(allo_logit)`` per virtual node.

    ``gamma = +1`` on allosteric proteins (promote allo-ness) and ``-1`` on orthosteric
    ones (penalise it), so a single scalar surfaces the *task-relevant* pocket. Used
    identically by the ranking loss (train), the ranked val metrics (checkpoint
    selection) and deployment, so all three stay aligned. ``class_weight = 0`` recovers
    pure-confidence ranking exactly.
    """
    z_conf = group_standardize(confs, batch)
    if class_weight == 0:
        return z_conf
    z_cls = group_standardize(class_logits, batch)
    return z_conf + class_weight * gamma * z_cls


def allosteric_ranking_geometry(
    node_pos: Tensor,
    node_batch: Tensor,
    centers: Tensor,
    center_batch: Tensor,
    center_types: Tensor,
    dists_any: Tensor,
) -> tuple[Tensor, Tensor]:
    """Per-virtual-node ``(gamma, rel_dists)`` for class-aware ranking.

    ``gamma`` is ``+1`` for nodes whose protein has >=1 allosteric center (allo-hunt),
    ``-1`` otherwise. ``rel_dists`` is the distance to the nearest center of the *task*
    class: allosteric centers for allosteric proteins, else the already-computed
    nearest-any-center distance ``dists_any``. Detached -- it is a target.
    """
    device = node_pos.device
    num_p = int(node_batch.max().item()) + 1
    allo_center = center_types == 1
    has_allo = torch.zeros(num_p, dtype=torch.bool, device=device)
    if allo_center.any():
        has_allo[center_batch[allo_center]] = True

    node_has_allo = has_allo[node_batch]
    gamma = torch.where(
        node_has_allo,
        torch.ones((), device=device),
        -torch.ones((), device=device),
    )

    rel_dists = dists_any.clone()
    if allo_center.any():
        y_allo = centers[allo_center]
        yb_allo = center_batch[allo_center]
        # For each virtual node, its nearest allosteric center. Nodes in proteins with
        # no allosteric center get no assignment and keep dists_any (they are ortho).
        ai = knn(x=y_allo, y=node_pos, batch_x=yb_allo, batch_y=node_batch, k=1)
        d_allo = torch.norm(y_allo[ai[1]] - node_pos[ai[0]], dim=-1)
        rel_dists[ai[0]] = d_allo
        rel_dists = torch.where(node_has_allo, rel_dists, dists_any)
    return gamma, rel_dists.detach()


class ClassAwareConfidenceRankingLoss(torch.nn.Module):
    """Listwise ranking loss on the *class-aware* score (see ``class_aware_ranking_score``).

    Identical in form to ``ConfidenceRankingLoss`` -- a per-protein ListNet top-one
    cross-entropy toward ``softmax(-dist / tau_dist)`` -- but (1) it ranks
    ``z(conf) + w * gamma * z(allo_logit)`` instead of confidence alone, and (2) the
    distance target uses the *task-class* site (``rel_dists``), so on a multi-pocket
    allosteric protein it promotes the allosteric pocket rather than the (usually
    higher-confidence) orthosteric one. Motivated by the re-rank probe
    (``CHANGELOG-allosteric-fixes.md`` §12): confidence alone ranks the orthosteric
    pocket first. ``class_weight = 0`` reduces this to the plain confidence ranking loss.
    """

    def __init__(
        self,
        tau_dist: float = 2.0,
        tau_conf: float = 0.5,
        class_weight: float = 1.0,
        eps: float = 1e-9,
    ):
        super().__init__()
        self.tau_dist = tau_dist
        self.tau_conf = tau_conf
        self.class_weight = class_weight
        self.eps = eps

    def forward(
        self,
        confs: Tensor,
        class_logits: Tensor,
        gamma: Tensor,
        rel_dists: Tensor,
        batch: Tensor,
    ) -> Tensor:
        if confs.numel() == 0:
            return confs.sum() * 0.0
        score = class_aware_ranking_score(
            confs, class_logits, gamma, batch, self.class_weight
        )
        target = group_softmax(-rel_dists.detach() / self.tau_dist, batch)
        log_p = torch.log(
            group_softmax(score / self.tau_conf, batch).clamp_min(self.eps)
        )
        num_groups = batch.unique().numel()
        return -(target * log_p).sum() / num_groups


class ConfidenceRankingLoss(torch.nn.Module):
    """Listwise (ListNet top-one) ranking loss over each protein's virtual nodes.

    The confidence scalar is only ever *ranked* -- rank-0 DCC/DCA pick the single
    highest-confidence node, rank-n the top ``num_sites + n`` -- so its absolute value is
    irrelevant; only the ordering *within a protein* matters. `ConfidenceLoss` regresses
    absolute values with MSE and, dominated by the many far ("background") nodes whose
    target is ~0, only weakly shapes that ordering. This term optimises it directly: for
    each protein it builds a soft target distribution over its virtual nodes that puts
    mass on the node(s) closest to a true site (``softmax(-dist / tau_dist)``) and drives
    the softmax of the predicted confidences toward it via cross-entropy.

    It touches only the confidence output, not the coordinate head, so it narrows the
    rank-0 vs rank-(K-1) gap without moving the localisation ceiling. It is K-agnostic:
    the softmax is taken per protein over whatever virtual nodes that protein has, so it
    is unaffected by raising ``number_of_global_nodes``.
    """

    def __init__(self, tau_dist: float = 2.0, tau_conf: float = 0.5, eps: float = 1e-9):
        super().__init__()
        self.tau_dist = tau_dist
        self.tau_conf = tau_conf
        self.eps = eps

    def forward(self, confs: Tensor, dists: Tensor, batch: Tensor):
        """
        Args:
            confs: (N,) predicted confidence per virtual node (carries grad).
            dists: (N,) distance from each virtual node to its nearest true site center,
                in the same node order as ``confs`` (detached; it is a target).
            batch: (N,) protein index per virtual node (``global_node.batch``).
        """
        if confs.numel() == 0:
            return confs.sum() * 0.0

        # Per-protein target: closest-to-site node gets the most mass. Detached so this
        # is a fixed target, never a path for gradients back into the coordinates.
        target = group_softmax(-dists.detach() / self.tau_dist, batch)
        log_p = torch.log(
            group_softmax(confs / self.tau_conf, batch).clamp_min(self.eps)
        )
        # target sums to 1 within each protein, so summing target*log_p over all nodes
        # equals the summed per-protein cross-entropy; divide by the protein count for a
        # mean-per-protein loss that is comparable across batch sizes.
        num_groups = batch.unique().numel()
        return -(target * log_p).sum() / num_groups
