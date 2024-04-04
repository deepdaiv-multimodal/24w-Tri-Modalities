import torch
import torch.nn.functional as F

def cluster_contrast(fushed, centroid, labels, bs):
    S = torch.matmul(fushed, centroid.t())

    target = torch.zeros(bs, centroid.shape[0]).to(S.device)

    target[range(target.shape[0]), labels] = 1

    S = S - target * (0.001)

    I2C_loss = F.nll_loss(F.log_softmax(S, dim=1), labels)

    return I2C_loss