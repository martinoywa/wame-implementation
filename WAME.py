import torch
from torch.optim import Optimizer


class WAME(Optimizer):
    """Implements the Weight–wise Adaptive learning rates with Moving average
    Estimator.

    Arguments:
        params (iterable): iterable of paramenters to optimize
        alpha (float, optional): smoothing constant (default: 0.9)
        etas (Tuple[float, float], optional): pair of (etaminus, etaplus), that
            are multiplicative increase and decrease factors
            (default: (0.1, 1.2))
        step_sizes (Tuple[float, float], optional): a pair of minimal and
            maximal allowed step sizes (default: (0.01, 100))
    """

    def __init__(self, params, alpha=0.9, etas=(0.1, 1.2), step_sizes=(0.01, 100)):
        if not 0.0 <= alpha:
            raise ValueError(f"Invalid theta: {alpha}")
        if not 0.0 < etas[0] < 1.0 < etas[1]:
            raise ValueError(f"Invalid eta values: {etas[0]}, {etas[1]}")

        defaults = dict(alpha=alpha, etas=etas, step_sizes=step_sizes)
        super(WAME, self).__init__(params, defaults)

    def __setstate__(self, state):
         super(WAME, self).__setstate__(state)


    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('WAME does not support sparse gradients')
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["prev"] = torch.zeros_like(p.data)
                    state["theta"] = torch.zeros_like(p.data)
                    state["z"] = torch.zeros_like(p.data)
                    state["step_size"] = grad.new().resize_as_(grad).fill_(1)

                etaminus, etaplus = group["etas"]
                step_size_min, step_size_max = group["step_sizes"]
                alpha = group["alpha"]
                step_size = state["step_size"]
                theta = state["theta"]
                z = state["z"]

                #print(state)

                state["step"] += 1

                mul_dx = grad.mul(state["prev"])

                if mul_dx > 0:
                    step_size = min(step_size*etaplus, step_size_max)
                elif mul_dx < 0:
                    step_size = max(step_size*etaminus, step_size_min)

                z = alpha * z + (1 - alpha) * step_size
                theta = alpha * theta + (1 - alpha) * (grad ** 2)

                new_grad = (z * grad) * (1 / theta)

                # update paramenters
                p.sub_(new_grad)

                state["prev"].copy_(grad)

        return loss
