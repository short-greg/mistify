import torch


# def get_strided_indices(n_points: int, stride: int, step: int=1):
#     initial_indices = torch.arange(0, n_points).as_strided((n_points - stride + 1, stride), (1, 1))
#     return initial_indices[torch.arange(0, len(initial_indices), step)]


# def stride_coordinates(coordinates: torch.Tensor, stride: int, step: int=1):

#     dim2_index = get_strided_indices(coordinates.size(1), stride, step)
#     return coordinates[:, dim2_index]


def stride_coordinates(coordinates: torch.Tensor, n_terms: int=1, step: int=1, n_points: int=1):

    batch = coordinates.size(0)
    n_vars = coordinates.size(1)
    n_length = coordinates.size(2)
    result = coordinates.as_strided((batch, n_vars, n_terms, n_points), (batch * n_vars * n_length, n_length, step, 1))
    return result
