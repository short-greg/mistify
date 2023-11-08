import torch


def get_strided_indices(n_points: int, stride: int, step: int=1):
    initial_indices = torch.arange(0, n_points).as_strided((n_points - stride + 1, stride), (1, 1))
    return initial_indices[torch.arange(0, len(initial_indices), step)]


def stride_coordinates(coordinates: torch.Tensor, stride: int, step: int=1):

    dim2_index = get_strided_indices(coordinates.size(1), stride, step)
    return coordinates[:, dim2_index]
