import torch
import torch.nn.functional as F


# activation functions

def swish(x: torch.Tensor):
    return x * torch.nn.functional.sigmoid(x)


class Swish(torch.nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            x.mul_(torch.sigmoid(x))
            return x
        else:
            return x * torch.sigmoid(x)


def log_contraction(x: torch.Tensor):
    positive = torch.log(1 + F.relu(x))
    negative = torch.log(1 - F.relu(-x))
    return positive - negative


class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class LogContraction(torch.nn.Module):
    def __init__(self, inplace=False):
        super(LogContraction, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return log_contraction(x, inplace=self.inplace)


class Softsign(torch.nn.Module):
    def __init__(self, a: float = 1.0, b: float = 1.0):
        super(Softsign, self).__init__()
        self.a = torch.nn.Parameter(
            torch.tensor(a, dtype=torch.float32, requires_grad=True)
        )
        self.b = torch.nn.Parameter(
            torch.tensor(b, dtype=torch.float32, requires_grad=True)
        )

    def forward(self, x: torch.Tensor):
        return self.a * x / ((x * self.b).abs() + 1)


def get_activation(activation_type: str):
    if activation_type == 'relu':
        return torch.nn.ReLU(inplace=True)
    elif activation_type == 'elu':
        return torch.nn.ELU(alpha=1.0, inplace=True)
    elif activation_type == 'lrelu':
        return torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)
    elif activation_type == 'prelu':
        return torch.nn.PReLU(num_parameters=1, init=0.25)
    elif activation_type == 'rrelu':
        return torch.nn.RReLU(inplace=True)
    elif activation_type == "selu":
        return torch.nn.SELU(inplace=True)
    elif activation_type == 'sigmoid':
        return torch.nn.Sigmoid()
    elif activation_type == 'tanh':
        return torch.nn.Tanh()
    elif activation_type == 'softplus':
        return torch.nn.Softplus()
    elif activation_type == 'softsign':
        return Softsign()
    elif activation_type == 'swish':
        return Swish()
    elif activation_type == 'log':
        return LogContraction(inplace=True)
    elif activation_type in {None, 'none', 'identity'}:
        return Identity()
    else:
        raise ValueError(f'activation_type ({activation_type} must be None or'
                         f'one of the following: \n'
                         f'     - relu \n'
                         f'     - elu \n'
                         f'     - lrelu \n'
                         f'     - rrelu \n'
                         f'     - selu \n'
                         f'     - sigmoid \n'
                         f'     - tanh \n'
                         f'     - softplus \n'
                         f'     - softsign \n'
                         f'     - swish \n')
