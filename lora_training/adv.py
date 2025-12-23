import math
import torch


def project_onto_l2_ball(delta, eps):
    """将扰动投影到 L2 范数球内以限制最大扰动幅度。

    参数:
    - delta: 扰动张量，形状 (B, ...)
    - eps: 每个样本允许的最大 L2 范数

    返回值:
    - 被缩放到 L2 球内的扰动张量
    """
    with torch.no_grad():
        B = delta.size(0)
        flat = delta.view(B, -1)
        norms = flat.norm(p=2, dim=1, keepdim=True)
        norms = torch.clamp(norms, min=1e-12)
        scale = torch.clamp(eps / norms, max=1.0)
        scale = scale.view(B, 1, 1)
        return delta * scale


def grad_norm(params):
    """计算参数梯度的全局 L2 范数（仅用于监控）。

    参数:
    - params: 可迭代的参数集合（通常是 model.parameters()）

    返回值:
    - 梯度的 L2 范数（浮点数）
    """
    total = 0.0
    for p in params:
        if p.grad is not None:
            v = p.grad.detach().float().view(-1)
            total += (v * v).sum().item()
    return math.sqrt(total + 1e-30)
