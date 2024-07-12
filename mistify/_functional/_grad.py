# 1st party
from abc import abstractmethod
import typing

# 3rd party
import torch

# local
from ..utils import reduce_as


class G(object):

    @abstractmethod
    def __call__(self, x: torch.Tensor, grad: torch.Tensor, oob: torch.Tensor=None):
        pass


class ClipG(G):

    def __init__(self, val: float):

        self.val = val

    def __call__(self, x: torch.Tensor, grad: torch.Tensor, oob: torch.Tensor=None):
        
        grad = grad.clone()
        if oob is None or False:
            return grad
        if oob is True:
            grad = grad.clip(-self.val, self.val)
        else:
            grad[oob] = grad[oob].clip(-self.val, self.val)
        return grad


class MulG(G):

    def __init__(self, val: float):
        """Multiply all values that are out of bounds

        Args:
            val (float): The value to multiply by
        """

        self.val = val

    def __call__(self, x: torch.Tensor, grad: torch.Tensor, oob: torch.Tensor=None):
        
        grad = grad.clone()
        if oob is None or False:
            return grad
        if oob is True:
            grad = grad * self.val
        else:
            grad[oob] = grad[oob] * self.val
        return grad


class BindG(G):

    def __init__(
        self, val: typing.Union[typing.Tuple[float, float], float]
    ):
        """Bind the gradient based on the value of x
        (This one ignores oob)

        Args:
            val (float): The value to bind by
        """
        if isinstance(val, typing.Tuple):
            self.lower, self.upper = val
        else:
            self.upper = val
            self.lower = -self.upper

    def __call__(self, x: torch.Tensor, grad: torch.Tensor, oob: torch.Tensor=None) -> torch.Tensor:
        """Use the Bind

        Args:
            x (torch.Tensor): The input value
            grad (torch.Tensor): The output gradient
            oob (torch.Tensor, optional): Whether it is out of bounds. Defaults to None.

        Returns:
            torch.Tensor: the gradient
        """
        grad = grad.clone()
        oob = (x < self.lower) | (x > self.upper)
        grad[oob] = 0.0
        return grad


class ZeroG(G):
    """Zero all gradients that are out of bounds
    """

    def __call__(self, x: torch.Tensor, grad: torch.Tensor, oob: torch.Tensor=None):
        
        grad = grad.clone()
        if oob is None or False:
            return grad
        
        if oob is True:
            grad = grad * 0.0
        else:
            grad[oob] = 0.0
        return grad


class AllG(G):
    """Allow all gradients to pass through as normal
    """

    def __call__(self, x: torch.Tensor, grad: torch.Tensor, oob: torch.Tensor=None):
        
        return grad.clone()


class SignG(torch.autograd.Function):
    """Use to clip the grad between two values
    Useful for smooth maximum/smooth minimum
    """

    @staticmethod
    def forward(ctx, x, g: G=None):
        """
        Forward pass of the Binary Step function.
        """
        ctx.g = g
        y = torch.sign(x)
        ctx.save_for_backward(x, y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of the Binary Step function using the Straight-Through Estimator.
        """
        x, y = ctx.saved_tensors
        grad_input = grad_output.clone()
        if ctx.g is None:
            return grad_input, None
        
        grad_input = ctx.g(x, grad_input, True)
        return ctx.g(x, grad_output, True), None


class HeavisideG(torch.autograd.Function):
    """Use to clip the grad between two values
    Useful for smooth maximum/smooth minimum
    """

    @staticmethod
    def forward(ctx, x, g: G=None):
        """
        Forward pass of the Binary Step function.
        """
        ctx.g = g
        y = torch.clamp(x, 0, 1).round()
        ctx.save_for_backward(x, y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of the Binary Step function 
        using the Straight-Through Estimator.
        """
        x, y = ctx.saved_tensors
        grad_input = grad_output.clone()
        if ctx.g is None:
            return grad_input, None
        
        return ctx.g(x, grad_input, True), None
        # return grad_input.clamp(-ctx.clip, ctx.clip), None


class ClampG(torch.autograd.Function):
    """Use to clip the grad between two values
    Useful for smooth maximum/smooth minimum
    """

    @staticmethod
    def forward(ctx, x, min=None, max=None, g: G=None):
        """
        Forward pass of the Binary Step function.
        """
        if min is not None or max is not None:
            y = torch.clamp(x, min, max)
        else:
            y = x
        # ctx.save_for_backward(x)
        ctx.min = min
        ctx.max = max
        ctx.save_for_backward(x, y)
        ctx.g = g
        return y

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of the Binary Step function using the Straight-Through Estimator.
        """
        x, y = ctx.saved_tensors
        grad_input = grad_output.clone()

        if ctx.g is None:
            return grad_input, None, None, None

        if ctx.min is not None and ctx.max is not None:
            x_range = (x < ctx.min) | (x > ctx.max)
        elif ctx.min is not None:
            x_range = (x < ctx.min)
        elif ctx.max is not None:
            x_range = (x > ctx.max)
        else: x_range = None

        if x_range is not None:
            # grad_input[x_range] = grad_input[x_range].clamp(-ctx.clip, ctx.clip)
            grad_input = ctx.g(x, grad_input, x_range)
        
        return grad_input, None, None, None


# class MaxG(torch.autograd.Function):
#     """Use to clip the grad between two values
#     Useful for smooth maximum/smooth minimum
#     """

#     @staticmethod
#     def forward(ctx, x1, x2, g: G=None):
#         """
#         Forward pass of the Max function
#         """
#         y = torch.max(x1, x2)
#         ctx.save_for_backward(x1, x2, y)
#         ctx.g = g
#         return y

#     @staticmethod
#     def backward(ctx, grad_output):
#         """
#         Backward pass of the Binary Step function using the Straight-Through Estimator.
#         """
#         x1, x2, y = ctx.saved_tensors
#         t = y - grad_output

#         # abs_grad = grad_output.abs()
#         # Check if this works if they are different sizes
#         x1_grad = grad_output.clone()
#         x1_grad[(x1 < x2)] = 0.0
        
#         # print(x1_grad, grad_output, x1, x2, t)
#         if ctx.g is not None:

#             condition = (x1 < x2)
#             # x1_grad[condition] = torch.relu(x1 - t)[condition]

#             x1_grad[condition] = (
#                 torch.relu(x1 - t) 
#                 - torch.relu(t - x2)
#             )[condition]
#             # condition2 = (x1 < x2) & (x2 < t)
#             # x1_grad[condition2] = -torch.min(abs_grad, torch.relu(t - x1))[condition2]
#             x1_grad = ctx.g(x1, x1_grad, condition)

#         x1_grad = reduce_as(x1_grad, x1)
        
#         # Handle x2
#         x2_grad = grad_output.clone()
#         if ctx.g is not None:
#             x2_grad[(x2 < x1)] = 0.0
#             condition = (x2 < x1)
#             # x2_grad[condition] = torch.relu(x2 - t)[condition]
#             x2_grad[condition] = (
#                 torch.relu(x2 - t) 
#                 - torch.relu(t - x1)
#             )[condition]
#             # condition2 = (x2 < x1) & (x1 < t)
#             # x2_grad[condition2] = -torch.min(abs_grad, torch.relu(t - x2))[condition2]
#             x2_grad = ctx.g(x2, x2_grad, condition)

#         x2_grad = reduce_as(x2_grad, x2)

#         return x1_grad, x2_grad, None


# class MinG(torch.autograd.Function):
#     """Use to clip the grad between two values
#     Useful for smooth maximum/smooth minimum
#     """

#     @staticmethod
#     def forward(ctx, x1, x2, g: G=None):
#         """
#         Forward pass of the Max function
#         """
#         y = torch.min(x1, x2)
#         ctx.save_for_backward(x1, x2, y)
#         ctx.g = g
#         return y

#     @staticmethod
#     def backward(ctx, grad_output):
#         """
#         Backward pass of the Binary Step function using the Straight-Through Estimator.
#         """
#         x1, x2, y = ctx.saved_tensors
#         grad_input = grad_output.clone()
#         t = y - grad_input

#         # abs_grad = grad_output.abs()
#         x1_grad = grad_output.clone()

#         x1_grad[(x1 > x2)] = 0.0

#         if ctx.g is not None:
#             condition = (x1 > x2)
#             x1_grad[condition] = (
#                 -torch.relu(t - x1) 
#                 + torch.relu(x2 - t)
#             )[condition]
#             # condition2 = condition & (x2 > t)
#             # condition2 = (x1 > x2) & (x1 > t)
#             # x1_grad[condition2] = torch.min(abs_grad, torch.relu(x1 - t))[condition2]
#             x1_grad = ctx.g(x1, x1_grad, condition)

#         x1_grad = reduce_as(x1_grad, x1)
        
#         x2_grad = grad_output.clone()
#         x2_grad[(x2 > x1)] = 0.0

#         if ctx.g is not None:
#             condition = (x2 > x1)
#             x2_grad[condition] = (
#                 -torch.relu(t - x2)
#                 + torch.relu(x1 - t)
#             )[condition]
#             # condition2 = (x2 > x1) & (x2 > t)
#             # x2_grad[condition2] = torch.min(abs_grad, torch.relu(x2 - t))[condition2]
#             x2_grad = ctx.g(x2, x2_grad, condition)
#         x2_grad = reduce_as(x2_grad, x2)

#         return x1_grad, x2_grad, None


def expand_as(x: torch.Tensor, ref: torch.Tensor, dim: int=-1) -> torch.Tensor:

    r = [1] * ref.dim()
    r[dim] = ref.size(dim)
    return x.repeat(r)


class MaxOnG(torch.autograd.Function):
    """Use to clip the grad between two values
    Useful for smooth maximum/smooth minimum
    """

    @staticmethod
    def forward(ctx, x, dim: int=-1, keepdim: bool=False, g: G=None):
        """
        Forward pass of the Max function
        """
        y = torch.max(x, dim, keepdim)
        ctx.save_for_backward(x, y[0])
        ctx.keepdim = keepdim
        ctx.dim = dim
        ctx.g = g
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor, ind: torch.Tensor):
        """
        Backward pass of the Binary Step function using the Straight-Through Estimator.
        """
        x, y = ctx.saved_tensors
        t = y - grad_output

        if not ctx.keepdim:
            grad_output = grad_output.unsqueeze(ctx.dim)
            y = y.unsqueeze(ctx.dim)
            t = t.unsqueeze(ctx.dim)

        grad_input = expand_as(grad_output, x, ctx.dim)

        base_cond = x != y
        grad_input[base_cond] = 0.0

        if ctx.g is not None:

            less_than = expand_as(y < t, x, ctx.dim)
            greater_than = (y > t) & (x > t)

            less_than_adj = expand_as(-torch.relu(t - y), x, ctx.dim)
            greater_than_adj = torch.relu(x - t)

            grad_input[less_than] = less_than_adj[less_than]
            grad_input[greater_than] = greater_than_adj[greater_than]
            grad_input = ctx.g(x, grad_input, base_cond &  (less_than | greater_than))        
            
            # currently this is a bit inefficient
            # less_than_t = torch.min(torch.relu(t - x), grad_output)
            # greater_than_t = torch.min(torch.relu(x - t), grad_output)
            # grad_input[(x > y) & (x < t)]

            # condition = (x < y)
            # grad_input[cond] = (
            #     greater_than_t
            #     - less_than_t
            #     # torch.relu(x - t)
            #     # - torch.relu(t - y)
            # )[cond]
            # min_diff = (x - t).min(dim=ctx.dim, keepdim=True)[0].abs()
            # condition2 = (x < t) & condition
            # grad_input[condition2] = -torch.min(min_diff, torch.relu(t - x))[condition2]

        return grad_input, None, None, None



class MinOnG(torch.autograd.Function):
    """Use to clip the grad between two values
    Useful for smooth maximum/smooth minimum
    """

    @staticmethod
    def forward(ctx, x, dim: int=-1, keepdim: bool=False, g: G=None):
        """
        Forward pass of the Max function
        """
        y = torch.min(x, dim, keepdim)
        ctx.save_for_backward(x, y[0])
        ctx.keepdim = keepdim
        ctx.dim = dim
        ctx.g = g
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor, ind: torch.Tensor):
        """
        Backward pass of the Binary Step function using the Straight-Through Estimator.
        """
    
        x, y = ctx.saved_tensors
        t = y - grad_output

        if not ctx.keepdim:
            grad_output = grad_output.unsqueeze(ctx.dim)
            y = y.unsqueeze(ctx.dim)
            t = t.unsqueeze(ctx.dim)

        grad_input = expand_as(grad_output, x, ctx.dim)

        base_cond = x != y
        grad_input[base_cond] = 0.0
        if ctx.g is not None:

            less_than = (y < t) & (x < t)
            greater_than = expand_as(
                y > t, x, ctx.dim
            )
            less_than_adj = -torch.relu(t - x)
            greater_than_adj = expand_as(
                torch.relu(y - t), 
                x, ctx.dim
            )
            grad_input[less_than] = less_than_adj[less_than]
            grad_input[greater_than] = greater_than_adj[greater_than]
            grad_input = ctx.g(
                x, grad_input, base_cond & (less_than | greater_than)
            ) #  )       
            
            # print(less_than.shape, grad_input.shape, less_than_adj.shape)
            # print(greater_than.shape, grad_input.shape, greater_than_adj.shape)
            # condition = (x > y)
            # grad_input[condition] = -torch.relu(t - x)[condition]

            # less_than_t = torch.min(torch.relu(t - x), grad_output)
            # greater_than_t = torch.min(torch.relu(x - t), grad_output)
            # # (0.3, 0.4, 0.5).. t=0.4... increase 0.3, decrease 0.5)
            # #   it will decrease by 0.1
            # # y < t
            # # # is it really necessary to decrease 0.5, though?
            # # (0.4, 0.5, 0.6).. t=0.3.. In this case decrease them all
            # # y > t
            # # i am not sure which is better

            # grad_input[cond] = (
            #     greater_than_t
            #     - less_than_t
            #     # - torch.relu(t - x)
            #     # + torch.relu(y - t)
            # )[cond]
            # # min_diff = (x - t).min(dim=ctx.dim, keepdim=True)[0].abs()

            # # condition2 = (x > t) & condition
            # # grad_input[condition2] = torch.min(min_diff, torch.relu(x - t))[condition2]

            # grad_input = ctx.g(x, grad_input, cond)

        return grad_input, None, None, None
