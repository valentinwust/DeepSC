import torch

def NB_loss( k_true, k_pred, theta_, mean=True , eps=1e-10, maxtheta=1e6):
    """ Negative Binomial loss.
    """
    theta = torch.minimum(theta_, torch.tensor(maxtheta)) + torch.tensor(eps)
    t1 = torch.lgamma(theta) + torch.lgamma(k_true + 1) - torch.lgamma(k_true + theta)
    t2 = torch.xlogy(theta + k_true, 1 + k_pred/theta) + torch.xlogy(k_true, theta) - torch.xlogy(k_true, k_pred)
    loss = t1+t2
    if not mean:
        return loss
    else:
        return torch.mean(loss, -1)