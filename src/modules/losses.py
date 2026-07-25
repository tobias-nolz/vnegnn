import torch
from torch import Tensor
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
