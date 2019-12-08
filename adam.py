import torch


class AdamOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, beta=(0.9, 0.999),
                 eps=1e-8, l2=0):
        self.lr = lr
        self.beta1, self.beta2 = beta
        self.eps = eps
        self.l2 = l2
        self.params = params
        self.__avg = tuple([None] * len(params))
        self.__sq_avg = tuple([None] * len(params))
        self.__step = 0

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for i, param in enumerate(self.params):
            avg, sq_avg = self.__avg[i], self.__sq_avg[i]
            grad = param.grad

            if self.__step == 0:
                avg = torch.zeros_like(param)
                sq_avg = torch.zeros_like(param)

            # perform step
            self.__step += 1
            if self.l2 != 0:
                with torch.no_grad():
                    grad.add_(self.l2 * param)

            avg.mul_(self.beta1)
            avg.add_((1 - self.beta1) * grad)
            sq_avg.mul_(self.beta2)
            sq_avg.add_((1 - self.beta2) * grad ** 2)
            unbiased_avg = avg / (1 - self.beta1 ** self.__step)
            unbiased_sq_avg = sq_avg / (1 - self.beta2 ** self.__step)
            quotient = unbiased_avg / (unbiased_sq_avg.sqrt() + self.eps)
            with torch.no_grad():
                param.sub_(self.lr * quotient)

        return loss
