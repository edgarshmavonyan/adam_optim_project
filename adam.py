import torch


class AdamOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, beta=(0.9, 0.999),
                 eps=1e-8, l2=0):
        defaults = dict(lr=lr, beta=beta, eps=eps, l2=l2)
        super().__init__(params, defaults)

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

                if len(state) == 0:
                    # init state
                    state['step'] = 0
                    state['avg'] = torch.zeros_like(param)
                    state['sq_avg'] = torch.zeros_like(param)

                # load current state
                state['step'] += 1
                step = state['step']
                avg = state['avg']
                sq_avg = state['sq_avg']
                beta1, beta2 = param_group['beta']
                eps = param_group['eps']
                lr = param_group['lr']
                l2 = param_group['l2']

                # perform step
                if l2 != 0:
                    grad.add_(l2 * param)
                avg.mul_(beta1)
                avg.add_((1 - beta1) * grad)
                sq_avg.mul_(beta2)
                sq_avg.add_((1 - beta2) * grad**2)
                unbiased_avg = avg / (1 - beta1**step)
                unbiased_sq_avg = sq_avg / (1 - beta2**step)
                quotient = unbiased_avg / (unbiased_sq_avg.sqrt() + eps)
                with torch.no_grad():
                    param.sub_(lr * quotient)

        return loss
