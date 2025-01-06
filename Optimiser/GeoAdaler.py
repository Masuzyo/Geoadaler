import torch
from torch import Tensor

class Geoadaler(torch.optim.Optimizer):
    r"""Implements the geoadler optimization algorithm.
    It has been proposed in `Eleh et al. 2024`__.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        beta (float, optional): coefficient used for computing
            running averages of gradient and its square (default: 0.9)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
    __ https://arxiv.org/pending
    """
    def __init__(self, params, lr=1e-3, beta=0.9,beta2=.99, eps=1,geomax=False):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >=0.0".format(lr))
        if not 0.0 <= beta < 1.0:
            raise ValueError("Invalid beta parameter: {}- should be in [0.0,1.0[".format(beta))
        if eps < 0.0:
            raise ValueError("Invalid epsilon value: {}- should be >=0.0".format(eps))
        defaults = dict(lr=lr, beta=beta,beta2=beta2, eps=eps,geomax=geomax)
        super(Geoadaler, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                
                if len(state) == 0:
                    state['step'] = 0
                    state['grad_avg'] = grad #torch.zeros_like(p.data)
                    state['denom'] = grad.norm(p=2).pow(2).add(group['eps'])#torch.zeros_like(p.data)

                grad_avg = state['grad_avg']
                beta = group['beta']
                #beta2 = group['beta2']
                denom = state['denom']

                state['step'] += 1
                grad_avg.mul_(beta).add_(grad,alpha=1-beta)

                if group['geomax']:
                    denom = denom.max(grad_avg.norm(p=2).pow(2).add(group['eps']))
                else:
                    denom = grad_avg.norm(p=2).pow(2).add(group['eps'])
                    #denom.mul_(beta2).addcmul_(grad, grad,alpha=1 - beta).add_(group['eps'])

                step_size = group['lr'] / (denom).sqrt()
                p.data.addcmul_(-step_size, grad_avg)
        
        return loss