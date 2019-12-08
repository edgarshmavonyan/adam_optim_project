import torch


class AdamOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, beta=(0.9, 0.999),
                 eps=1e-8, l2=0, amsgrad=False):
        defaults = dict(lr=lr, beta=beta, eps=eps, l2=l2, amsgrad=amsgrad)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.params.groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for param_group in self.param_groups:
            for param in param_group['params']:
                if param.grad is None:
                    continue
                grad = param.grad
                state = self.state[param]
                amsgrad = param_group['amsgrad']

                if len(state) == 0:
                    # init state
                    state['step'] = 0
                    state['avg'] = torch.zeros_like(param)
                    state['sq_avg'] = torch.zeros_like(param)
                    if amsgrad:
                        state['max_sq_avg'] = torch.zeros_like(param)

                # load current state
                state['step'] += 1
                step = state['step']
                avg = state['avg']
                sq_avg = state['sq_avg']
                beta1, beta2 = param_group['beta']
                eps = param_group['eps']
                lr = param_group['lr']
                l2 = param_group['l2']
                if amsgrad:
                    max_sq_avg = state['max_sq_avg']

                # perform step
                if l2 != 0:
                    grad.add_(l2 * param)
                avg.mul_(beta1)
                avg.add_((1 - beta1) * grad)
                sq_avg.mul_(beta2)
                sq_avg.add_((1 - beta2) * grad**2)
                unbiased_avg = avg / (1 - beta1**step)
                if amsgrad:
                    torch.max(max_sq_avg, sq_avg, out=max_sq_avg)
                    unbiased_sq_avg = max_sq_avg / (1 - beta2**step)
                else:
                    unbiased_sq_avg = sq_avg / (1 - beta2**step)
                quotient = unbiased_avg / (unbiased_sq_avg.sqrt() + eps)
                with torch.no_grad():
                    param.sub_(lr * quotient)

        return loss
