import torch


def create_meshgrid(height: int, width: int, normalized: bool = False, flatten: bool = False, centering: bool = False):
    """Create meshgrid.

    Args:
        height (int): the height of the grid.
        width (int): the width of the grid.
        normalized (bool): if True, the resulting coordinates are normalized into [0, 1] inclusive. Default: False.
        flatten (bool): if True, the resulting coordinates are flattened. Default: False.
        centering (bool): if True, use the centers (+0.5) instead of the upperleft corners. Default: False.

    Returns:
        A tensor of the grid coordinates in the shape of (H, W, 2) if not flatten or (HxW, 2) otherwise. The tensor is
            float if normalized or int otherwise.
    """
    if normalized and not centering:
        # TODO: use torch.arange in normalized non-centering implemention.
        # Currently, this affects 2D-3D registration models.
        h_values = torch.linspace(0.0, 1.0, steps=height).cuda()
        w_values = torch.linspace(0.0, 1.0, steps=width).cuda()
    else:
        h_values = torch.arange(height).cuda()
        w_values = torch.arange(width).cuda()
        if centering:
            h_values = h_values.float() + 0.5
            w_values = w_values.float() + 0.5
        if normalized:
            h_values = h_values.float() / float(height)
            w_values = w_values.float() / float(width)

    pixels = torch.cartesian_prod(h_values, w_values)  # (H) x (W) -> (HxW, 2)

    if not flatten:
        pixels = pixels.view(height, width, 2)  # (HxW, 2) -> (H, W, 2)
    return pixels
