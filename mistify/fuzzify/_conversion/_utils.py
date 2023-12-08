import torch


def stride_coordinates(coordinates: torch.Tensor, n_terms: int=1, step: int=1, n_points: int=1, skip: int=1) -> torch.Tensor:
    """Use to convert the coordinates for a set of fuzzy membership functions into 
    coordinates for each function that makes up the set

    Args:
        coordinates (torch.Tensor): The coordinates that make up the 
        n_terms (int, optional): The number of terms. Defaults to 1.
        step (int, optional): The step size between terms . Defaults to 1.
        n_points (int, optional): How many points make up the term. Defaults to 1.
        skip (int, optional): The number of coordinates to skip between each point of a term. Defaults to 1.

    Returns:
        torch.Tensor: The coordinates for each membership function
    """
    batch = coordinates.size(0)
    n_vars = coordinates.size(1)
    n_length = coordinates.size(2)
    result = coordinates.as_strided((batch, n_vars, n_terms, n_points), (batch * n_vars * n_length, n_length, step, skip))
    return result
