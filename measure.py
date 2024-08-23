import torch


def Expectation(obj, grad, p):
    if p is not None:
        return (torch.matmul(p, obj), torch.matmul(p, grad))