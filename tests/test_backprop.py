import torch

from torchvectorized.nn import ExpmLogm
from torchvectorized.utils import sym
from torchvectorized.vlinalg import vSymEig, vExpm, vLogm
import matplotlib.pyplot as plt

OPTIMIZER_STEPS = 2000


def spd_matrices():
    data = sym((-15 - 3) * torch.rand(500, 9, 1, 1, 1) + 3)
    data = vExpm(data, replace_nans=True)
    return vLogm(data, replace_nans=True)


if __name__ == "__main__":
    cos_sim_computer = torch.nn.CosineSimilarity()

    gt = spd_matrices().cuda().float()
    gt_vals, gt_vecs = vSymEig(gt, eigenvectors=True, descending_eigenvals=True)

    input = torch.nn.Parameter(spd_matrices().float().cuda(), requires_grad=True)

    optimizer = torch.optim.Adam([input], lr=0.001)

    steps = []
    eig_vals_loss = []
    eig_vecs_loss = []
    cos_sim_metrics_v1 = []
    cos_sim_metrics_v2 = []
    cos_sim_metrics_v3 = []
    losses = []
    non_spd = []

    exp_log = ExpmLogm()

    for optim_step in range(OPTIMIZER_STEPS):
        optimizer.zero_grad()
        eig_vals, eig_vecs = vSymEig(input, eigenvectors=True, descending_eigenvals=True)

        loss_eig_val = torch.nn.functional.l1_loss(eig_vals, gt_vals)
        loss_eig_vecs = torch.nn.functional.l1_loss(torch.abs(eig_vecs), torch.abs(gt_vecs))

        loss = 10 * torch.nn.functional.l1_loss(exp_log(input), gt)

        steps.append(optim_step)
        eig_vals_loss.append(loss_eig_val.detach().cpu().data)
        eig_vecs_loss.append(loss_eig_vecs.detach().cpu().data)
        cos_sim_metrics_v1.append(
            torch.abs(
                cos_sim_computer(eig_vecs[:, :, 0, :, :, :], gt_vecs[:, :, 0, :, :, :])).mean().detach().cpu().data)
        cos_sim_metrics_v2.append(
            torch.abs(
                cos_sim_computer(eig_vecs[:, :, 1, :, :, :], gt_vecs[:, :, 1, :, :, :])).mean().detach().cpu().data)
        cos_sim_metrics_v3.append(
            torch.abs(
                cos_sim_computer(eig_vecs[:, :, 2, :, :, :], gt_vecs[:, :, 2, :, :, :])).mean().detach().cpu().data)
        losses.append(loss.detach().cpu().data /10)

        loss.backward()
        optimizer.step()
        print(loss.data)

    plt.plot(steps, eig_vals_loss, label="Eigenvalues L1 error")
    plt.plot(steps, eig_vecs_loss, label="Eigenvectors L1 error")
    plt.plot(steps, losses, label="Tensor L1 error")
    plt.plot(steps, cos_sim_metrics_v1, label="Cosine Similarity \u03B51")
    plt.plot(steps, cos_sim_metrics_v2, label="Cosine Similarity \u03B52")
    plt.plot(steps, cos_sim_metrics_v3, label="Cosine Similarity \u03B53")
    plt.xlabel('Step')
    plt.legend()
    plt.savefig("/data/users/benoit/optimplot.png")
    plt.show()
