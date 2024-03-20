# # 3rd party
# import torch


# def differ(m: torch.Tensor, m2: torch.Tensor) -> torch.Tensor:
#     """
#     Take the difference between two fuzzy sets
    
#     Args:
#         m (torch.Tensor): Fuzzy set to subtract from 
#         m2 (torch.Tensor): Fuzzy set to subtract

#     Returns:
#         torch.Tensor: 
#     """
#     return (m - m2).clamp(0.0, 1.0)


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
#     base = (1 - m1) + torch.min(m2, m1)
#     if dim is None:
#         return base.type_as(m1)
#     return base.min(dim=dim)[0].type_as(m1)


# def exclusion(m1: torch.Tensor, m2: torch.Tensor, dim: int=None) -> 'torch.Tensor':
#     """Calculate whether m1 is excluded from m2. If dim is None then it will calculate per
#     element otherwise it will aggregate over that dimension

#     Args:
#         m1 (torch.Tensor): The membership to calculate the exclusion of
#         m2 (torch.Tensor): The membership to check if m1 is excluded
#         dim (int, optional): The dimension to aggregate over. Defaults to None.

#     Returns:
#         torch.Tensor: the tensor describing inclusion
#     """
#     base = (1 - m2) + torch.min(m2, m1)
#     if dim is None:
#         return base.type_as(m1)
#     return base.min(dim=dim)[0].type_as(m1)


# def complement(m: torch.Tensor) -> torch.Tensor:
#     """Calculate the complement

#     Args:
#         m (torch.Tensor): The membership

#     Returns:
#         torch.Tensor: The fuzzy complement
#     """
#     return 1 - m


# def else_(m: torch.Tensor, dim: int=-1, keepdim: bool=False) -> torch.Tensor:
#     """Take the 'else' on a set

#     Args:
#         m (torch.Tensor): The fuzzy set
#         dim (int, optional): The dimension to calculate on. Defaults to -1.
#         keepdim (bool, optional): Whether to keep the dimension of m. Defaults to False.

#     Returns:
#         torch.Tensor: the else value of m along the dimension
#     """
#     return 1 - m.max(dim=dim, keepdim=keepdim)[0]


# def intersect(m1: torch.Tensor, m2: torch.Tensor) -> torch.Tensor:
#     """intersect two fuzzy sets

#     Args:
#         m1 (torch.Tensor): Fuzzy set to intersect
#         m2 (torch.Tensor): Fuzzy set to intersect with

#     Returns:
#         torch.Tensor: Intersection of two fuzzy sets
#     """
#     return torch.min(m1, m2)


# def intersect_on(m: torch.Tensor, dim: int=-1) -> torch.Tensor:
#     """Intersect elements of a fuzzy set on specfiied dimension

#     Args:
#         m (torch.Tensor): Fuzzy set to intersect

#     Returns:
#         torch.Tensor: Intersection of two fuzzy sets
#     """
#     return torch.min(m, dim=dim)[0]


# def unify(m: torch.Tensor, m2: torch.Tensor) -> torch.Tensor:
#     """union on two fuzzy sets

#     Args:
#         m (torch.Tensor):  Fuzzy set to take union of
#         m2 (torch.Tensor): Fuzzy set to take union with

#     Returns:
#         torch.Tensor: Union of two fuzzy sets
#     """
#     return torch.max(m, m2)


# def unify_on(m: torch.Tensor, dim: int=-1) -> torch.Tensor:
#     """Unify elements of a fuzzy set on specfiied dimension

#     Args:
#         m (torch.Tensor): Fuzzy set to take the union of

#     Returns:
#         torch.Tensor: Union of two fuzzy sets
#     """
#     return torch.max(m, dim=dim)[0]
