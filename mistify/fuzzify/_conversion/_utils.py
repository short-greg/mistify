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


def generate_spaced_params(n_steps: int, lower: float=0, upper: float=1, in_features: int=None) -> torch.Tensor:
    """Generate parameters that are equally spaced

    Args:
        n_steps (int): The number of steps to generate
        lower (float, optional): The lower bound for the spaced parameters. Defaults to 0.
        upper (float, optional): The upper bound for the spaced parameters. Defaults to 1.
        in_features (int, optional):

    Returns:
        torch.Tensor: 
    """
    features = torch.linspace(lower, upper, n_steps)[None, None, :]
    if in_features is not None:
        features = features.repeat(1, in_features, 1)
    return features


def generate_repeat_params(n_steps: int, value: float, in_features: int=None) -> torch.Tensor:
    """Generate several parameters

    Args:
        n_steps (int): The number of parameters
        value (float): The value for the parameters
        in_features (int, optional):

    Returns:
        torch.Tensor: The set of parameters
    """
    features = torch.full((1, 1, n_steps), value)
    if in_features is not None:
        features = features.repeat(1, in_features, 1)
    return features


