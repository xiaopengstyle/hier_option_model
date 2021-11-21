import torch


def toTensor(obs):
    obs = obs.transpose(2, 0, 1)
    return torch.FloatTensor(obs).unsqueeze(0)