# import torch


# def differ(m1: torch.Tensor, m2: torch.Tensor) -> 'torch.Tensor':
#     """Take the difference between set m1 and set m2

#     Args:
#         m1 (torch.Tensor): The set to take the difference from
#         m2 (torch.Tensor): The set to take the difference with

#     Returns:
#         torch.Tensor: The difference
#     """
#     return (m1 - m2).clamp(0.0, 1.0)


# def unify(m1: torch.Tensor, m2: torch.Tensor) -> torch.Tensor:
#     """Take the union of two sets

#     Args:
#         m1 (torch.Tensor): First set
#         m2 (torch.Tensor): Second set

#     Returns:
#         torch.Tensor: The unified set
#     """
#     return torch.max(m1, m2)


# def intersect(m1: torch.Tensor, m2: torch.Tensor) -> 'torch.Tensor':
#     """Take the intersection of two sets

#     Args:
#         m1 (torch.Tensor): First set
#         m2 (torch.Tensor): Second set

#     Returns:
#         torch.Tensor: The intersection of the sets
#     """
#     return torch.min(m1, m2)


# def unify_on(m1: torch.Tensor, dim: int=-1, keepdim: bool=False) -> 'torch.Tensor':
#     """Take the union on a dimension

#     Args:
#         m1 (torch.Tensor): The set to take the union on
#         dim (int, optional): The dimension to take the union on. Defaults to -1.
#         keepdim (bool, optional): Whether to keep the dimension. Defaults to False.

#     Returns:
#         torch.Tensor: The unioned set
#     """
#     return torch.max(m1, dim=dim, keepdim=keepdim)


# def intersect_on(m1: torch.Tensor, dim: int=-1, keepdim: bool=False) -> 'torch.Tensor':
#     """Take the intersection on a dimension

#     Args:
#         m1 (torch.Tensor): The set to take the intersection on
#         dim (int, optional): The dimension to take the intersection on. Defaults to -1.
#         keepdim (bool, optional): Whether to keep the dimension. Defaults to False.

#     Returns:
#         torch.Tensor: The intersected set
#     """
#     return torch.min(m1, dim=dim, keepdim=keepdim)


# def inclusion(m1: torch.Tensor, m2: torch.Tensor, dim: int=None) -> 'torch.Tensor':
#     """Calculate whether m1 is included in m2. If dim is None then it will calculate per
#     element otherwise it will aggregate over that dimension

#     Args:
#         m1 (torch.Tensor): The membership to calculate the inclusion of
#         m2 (torch.Tensor): The membership to check if m1 is included
#         dim (int, optional): The dimension to aggregate over. Defaults to None.

#     Returns:
#         torch.Tensor: the tensor describing inclusion
#     """
#     base = (m1 <= m2).type_as(m1)
#     if dim is None:
#         return base
#     return base.min(dim=dim)[0]


# def exclusion(m1: torch.Tensor, m2: torch.Tensor, dim: int=None) -> 'torch.Tensor':
#     """Calculate whether m1 is excluded from m2. If dim is None then it will calculate per
#     element otherwise it will aggregate over that dimension

#     Args:
#         m1 (torch.Tensor): The membership to calculate the exclusion of
#         m2 (torch.Tensor): The membership to check if m1 is excluded
#         dim (int, optional): The dimension to aggregate over. Defaults to None.

#     Returns:
#         torch.Tensor: the tensor describing exclusion
#     """
#     base = (m1 >= m2).type_as(m1)    
#     if dim is None:
#         return base
#     return base.min(dim=dim)[0]

# def complement(m: torch.Tensor) -> torch.Tensor:
#     """Calculate the complement

#     Args:
#         m (torch.Tensor): The membership

#     Returns:
#         torch.Tensor: The fuzzy complement
#     """
#     return 1 - m


# def forget(m: torch.Tensor, p: float) -> torch.Tensor:
#     """Randomly forget values (this will make them unknown)

#     Args:
#         m (torch.Tensor): the membership matrix
#         p (float): the probability of forgetting

#     Returns:
#         torch.Tensor: the tensor with randomly forgotten values
#     """
#     return m * (torch.rand_like(m) < p).type_as(m) + 0.5


# def else_(m: torch.Tensor, dim: int=-1, keepdim: bool=False) -> torch.Tensor:
#     """Take the 'else' on the set

#     Args:
#         m (torch.Tensor): The set
#         dim (int, optional): The dimension to take else on. Defaults to -1.
#         keepdim (bool, optional): Whether to keep the dim. Defaults to False.

#     Returns:
#         torch.Tensor: The 'else' set
#     """
#     return 1 - m.max(dim=dim, keepdim=keepdim)[0]
