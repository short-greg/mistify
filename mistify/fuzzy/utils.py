import torch


def smooth_max(x: torch.Tensor, x2: torch.Tensor, a: float) -> torch.Tensor:
    """Smooth approximation to the max function of two tensors

    Args:
        x (torch.Tensor): Tensor to take max of
        x2 (torch.Tensor): Other tensor to take max of
        a (float): Value to 

    Returns:
        torch.Tensor: Tensor containing the maximum of x1 and x2
    """
    z1 = ((x + 1) ** a).detach()
    z2 = ((x2 + 1) ** a).detach()
    return (x * z1 + x2 * z2) / (z1 + z2)


def smooth_max_on(x: torch.Tensor, dim: int, a: float, keepdim: bool=False) -> torch.Tensor:
    """Take smooth max over specified dimension

    Args:
        x (torch.Tensor): 
        dim (int): Dimension to take max over
        a (float): Smoothing value. The larger the value the smoother

    Returns:
        torch.Tensor: Result of the smooth max
    """
    z = ((x + 1) ** a).detach()
    return (x * z).sum(dim=dim, keepdim=keepdim) / z.sum(dim=dim, keepdim=keepdim)


def smooth_min(x: torch.Tensor, x2: torch.Tensor, a: float) -> torch.Tensor:
    """Take smooth m over specified dimension

    Args:
        x (torch.Tensor): 
        dim (int): Dimension to take max over
        a (float): Smoothing value. The larger the value the smoother

    Returns:
        torch.Tensor: Result of the smooth max
    """
    return smooth_max(x, x2, -a)


def smooth_min_on(
        x: torch.Tensor, dim: int, a: float, keepdim: bool=False
    ) -> torch.Tensor:
    """Take smooth min over specified dimension

    Args:
        x (torch.Tensor): 
        dim (int): Dimension to take max over
        a (float): Smoothing value. The larger the value the smoother
        keepdim (bool): Whether to keep the dimension or not

    Returns:
        torch.Tensor: Result of the smooth max
    """
    return smooth_max_on(x, dim, -a, keepdim=keepdim)


def adamax(x: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Smooth approximation to the max function of two tensors

    Args:
        x (torch.Tensor): Tensor to take max of
        x2 (torch.Tensor): Other tensor to take max of
    
    Returns:
        torch.Tensor: Tensor containing the maximum of x1 and x2
    """
    q = torch.clamp(-69 / torch.log(torch.min(x, x2)), max=1000, min=-1000).detach()  
    return ((x ** q + x2 ** q) / 2) ** (1 / q)


def adamin(x: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Smooth approximation to the min function of two tensors

    Args:
        x (torch.Tensor): Tensor to take max of
        x2 (torch.Tensor): Other tensor to take max of
    
    Returns:
        torch.Tensor: Tensor containing the maximum of x1 and x2
    """
    q = torch.clamp(69 / torch.log(torch.min(x, x2)).detach(), max=1000, min=-1000)
    result = ((x ** q + x2 ** q) / 2) ** (1 / q)
    return result


def adamax_on(x: torch.Tensor, dim: int, keepdim: bool=False) -> torch.Tensor:
    """Take smooth max over specified dimension

    Args:
        x (torch.Tensor): 
        dim (int): Dimension to take max over
        a (float): Smoothing value. The larger the value the smoother

    Returns:
        torch.Tensor: Result of the smooth max
    """
    q = torch.clamp(-69 / torch.log(torch.min(x, dim=dim)[0]).detach(), max=1000, min=-1000)
    return (torch.sum(x ** q.unsqueeze(dim), dim=dim, keepdim=keepdim) / x.size(dim)) ** (1 / q)


def adamin_on(x: torch.Tensor, dim: int, keepdim: bool=False) -> torch.Tensor:
    """Take smooth min over specified dimension

    Args:
        x (torch.Tensor): 
        dim (int): Dimension to take max over
        keepdim (bool): Whether to keep the dimension or not

    Returns:
        torch.Tensor: Result of the smooth max
    """
    q = torch.clamp(69 / torch.log(torch.min(x, dim=dim)[0]).detach(), max=1000, min=-1000)
    return (torch.sum(x ** q.unsqueeze(dim), dim=dim, keepdim=keepdim) / x.size(dim)) ** (1 / q)


def differ(m: torch.Tensor, m2: torch.Tensor) -> torch.Tensor:
    """
    Take the difference between two fuzzy sets
    
    Args:
        m (torch.Tensor): Fuzzy set to subtract from 
        m2 (torch.Tensor): Fuzzy set to subtract

    Returns:
        torch.Tensor: 
    """
    return (m - m2).clamp(0.0, 1.0)


def positives(*size: int, dtype=torch.float32, device='cpu') -> torch.Tensor:
    """
    Generate a positive fuzzy set

    Args:
        dtype (_type_, optional): . Defaults to torch.float32.
        device (str, optional): . Defaults to 'cpu'.

    Returns:
        torch.Tensor: Positive fuzzy set
    """
    return torch.ones(*size, dtype=dtype, device=device)


def negatives(*size: int, dtype: torch.dtype=torch.float32, device='cpu') -> torch.Tensor:
    """
    Generate a negative fuzzy set

    Args:
        dtype (torch.dtype, optional): The data type for the fuzzys set. Defaults to torch.float32.
        device (str, optional): The device for the fuzzy set. Defaults to 'cpu'.

    Returns:
        torch.Tensor: Negative fuzzy set
    """
    return torch.zeros(*size, dtype=dtype, device=device)


def intersect(m1: torch.Tensor, m2: torch.Tensor) -> torch.Tensor:
    """intersect two fuzzy sets

    Args:
        m1 (torch.Tensor): Fuzzy set to intersect
        m2 (torch.Tensor): Fuzzy set to intersect with

    Returns:
        torch.Tensor: Intersection of two fuzzy sets
    """
    return torch.min(m1, m2)


def intersect_on(m: torch.Tensor, dim: int=-1) -> torch.Tensor:
    """Intersect elements of a fuzzy set on specfiied dimension

    Args:
        m (torch.Tensor): Fuzzy set to intersect

    Returns:
        torch.Tensor: Intersection of two fuzzy sets
    """
    return torch.min(m, dim=dim)[0]


def unify(m: torch.Tensor, m2: torch.Tensor) -> torch.Tensor:
    """union on two fuzzy sets

    Args:
        m (torch.Tensor):  Fuzzy set to take union of
        m2 (torch.Tensor): Fuzzy set to take union with

    Returns:
        torch.Tensor: Union of two fuzzy sets
    """
    return torch.max(m, m2)


def unify_on(m: torch.Tensor, dim: int=-1) -> torch.Tensor:
    """Unify elements of a fuzzy set on specfiied dimension

    Args:
        m (torch.Tensor): Fuzzy set to take the union of

    Returns:
        torch.Tensor: Union of two fuzzy sets
    """
    return torch.max(m, dim=dim)[0]


def inclusion(m1: torch.Tensor, m2: torch.Tensor) -> 'torch.Tensor':
    return (1 - m2) + torch.min(m1, m2)


def exclusion(m1: torch.Tensor, m2: torch.Tensor) -> 'torch.Tensor':
    return (1 - m1) + torch.min(m1, m2)


def rand(*size: int,  dtype=torch.float32, device='cpu'):

    return (torch.rand(*size, device=device, dtype=dtype))


def negatives(*size: int,  dtype=torch.float32, device='cpu'):

    return (torch.zeros(*size, device=device, dtype=dtype))


def positives(*size: int,  dtype=torch.float32, device='cpu'):

    return (torch.ones(*size, device=device, dtype=dtype))
