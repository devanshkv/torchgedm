"""
Mathematical utility functions for NE2001 model

All functions are vectorized (support batching) and differentiable for use with PyTorch.
"""

import torch
from typing import Tuple, Union


def sech2(x: torch.Tensor) -> torch.Tensor:
    """
    Compute sech²(x) = 1/cosh²(x) in a numerically stable way.

    Args:
        x: Input tensor of any shape

    Returns:
        sech²(x) with same shape as input

    Note:
        Uses the identity sech²(x) = 4/(e^x + e^(-x))²
        This is more numerically stable than 1/cosh(x)²
    """
    # Compute e^x and e^(-x)
    exp_x = torch.exp(x)
    exp_neg_x = torch.exp(-x)

    # sech²(x) = 4 / (e^x + e^(-x))²
    denominator = exp_x + exp_neg_x
    return 4.0 / (denominator * denominator)


def rotation_matrix_z(theta: torch.Tensor) -> torch.Tensor:
    """
    Create 3D rotation matrix for rotation around Z-axis.

    Args:
        theta: Rotation angle(s) in radians. Can be scalar or batched.

    Returns:
        Rotation matrix of shape (*batch_dims, 3, 3) if theta is batched,
        or (3, 3) if theta is scalar

    Note:
        R_z(θ) = [[cos(θ), -sin(θ), 0],
                  [sin(θ),  cos(θ), 0],
                  [0,       0,      1]]
    """
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    # Handle both scalar and batched inputs
    if theta.dim() == 0:
        # Scalar case
        return torch.stack([
            torch.stack([cos_theta, -sin_theta, torch.zeros_like(theta)]),
            torch.stack([sin_theta, cos_theta, torch.zeros_like(theta)]),
            torch.stack([torch.zeros_like(theta), torch.zeros_like(theta), torch.ones_like(theta)])
        ])
    else:
        # Batched case
        batch_shape = theta.shape
        zeros = torch.zeros_like(theta)
        ones = torch.ones_like(theta)

        # Build rotation matrix with shape (*batch_shape, 3, 3)
        row1 = torch.stack([cos_theta, -sin_theta, zeros], dim=-1)
        row2 = torch.stack([sin_theta, cos_theta, zeros], dim=-1)
        row3 = torch.stack([zeros, zeros, ones], dim=-1)

        return torch.stack([row1, row2, row3], dim=-2)


def rotation_matrix_y(theta: torch.Tensor) -> torch.Tensor:
    """
    Create 3D rotation matrix for rotation around Y-axis.

    Args:
        theta: Rotation angle(s) in radians. Can be scalar or batched.

    Returns:
        Rotation matrix of shape (*batch_dims, 3, 3) if theta is batched,
        or (3, 3) if theta is scalar

    Note:
        R_y(θ) = [[cos(θ),  0, sin(θ)],
                  [0,       1, 0     ],
                  [-sin(θ), 0, cos(θ)]]
    """
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    # Handle both scalar and batched inputs
    if theta.dim() == 0:
        # Scalar case
        return torch.stack([
            torch.stack([cos_theta, torch.zeros_like(theta), sin_theta]),
            torch.stack([torch.zeros_like(theta), torch.ones_like(theta), torch.zeros_like(theta)]),
            torch.stack([-sin_theta, torch.zeros_like(theta), cos_theta])
        ])
    else:
        # Batched case
        batch_shape = theta.shape
        zeros = torch.zeros_like(theta)
        ones = torch.ones_like(theta)

        # Build rotation matrix with shape (*batch_shape, 3, 3)
        row1 = torch.stack([cos_theta, zeros, sin_theta], dim=-1)
        row2 = torch.stack([zeros, ones, zeros], dim=-1)
        row3 = torch.stack([-sin_theta, zeros, cos_theta], dim=-1)

        return torch.stack([row1, row2, row3], dim=-2)


def rotation_matrix_x(theta: torch.Tensor) -> torch.Tensor:
    """
    Create 3D rotation matrix for rotation around X-axis.

    Args:
        theta: Rotation angle(s) in radians. Can be scalar or batched.

    Returns:
        Rotation matrix of shape (*batch_dims, 3, 3) if theta is batched,
        or (3, 3) if theta is scalar

    Note:
        R_x(θ) = [[1, 0,       0      ],
                  [0, cos(θ), -sin(θ)],
                  [0, sin(θ),  cos(θ)]]
    """
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    # Handle both scalar and batched inputs
    if theta.dim() == 0:
        # Scalar case
        return torch.stack([
            torch.stack([torch.ones_like(theta), torch.zeros_like(theta), torch.zeros_like(theta)]),
            torch.stack([torch.zeros_like(theta), cos_theta, -sin_theta]),
            torch.stack([torch.zeros_like(theta), sin_theta, cos_theta])
        ])
    else:
        # Batched case
        batch_shape = theta.shape
        zeros = torch.zeros_like(theta)
        ones = torch.ones_like(theta)

        # Build rotation matrix with shape (*batch_shape, 3, 3)
        row1 = torch.stack([ones, zeros, zeros], dim=-1)
        row2 = torch.stack([zeros, cos_theta, -sin_theta], dim=-1)
        row3 = torch.stack([zeros, sin_theta, cos_theta], dim=-1)

        return torch.stack([row1, row2, row3], dim=-2)


def apply_rotation(points: torch.Tensor, rotation_matrix: torch.Tensor) -> torch.Tensor:
    """
    Apply rotation matrix to 3D points.

    Args:
        points: Points of shape (*batch_dims, 3) or (3,)
        rotation_matrix: Rotation matrix of shape (*batch_dims, 3, 3) or (3, 3)

    Returns:
        Rotated points with same shape as input points

    Note:
        Performs matrix multiplication: R @ p for each point
    """
    # Handle batched and unbatched cases
    if points.dim() == 1:
        # Single point: (3,)
        return torch.matmul(rotation_matrix, points)
    else:
        # Batched points: (*batch_dims, 3)
        # Use Einstein summation for flexibility with arbitrary batch dimensions
        return torch.einsum('...ij,...j->...i', rotation_matrix, points)


def ellipsoid_distance(
    points: torch.Tensor,
    center: Union[torch.Tensor, Tuple[float, float, float]],
    axes: Union[torch.Tensor, Tuple[float, float, float]],
    theta: torch.Tensor = None,
    phi: torch.Tensor = None,
    psi: torch.Tensor = None
) -> torch.Tensor:
    """
    Compute normalized distance from point to ellipsoid surface.

    For a point to be inside the ellipsoid, the distance should be ≤ 1.

    Args:
        points: Points to evaluate, shape (*batch_dims, 3) or (3,)
                where last dimension is [x, y, z]
        center: Ellipsoid center (x0, y0, z0)
        axes: Ellipsoid semi-axes (a, b, c)
        theta: Rotation angle around Z-axis (radians), optional
        phi: Rotation angle around Y-axis (radians), optional
        psi: Rotation angle around X-axis (radians), optional

    Returns:
        Normalized distance. Values ≤ 1 are inside ellipsoid.
        Shape matches batch dimensions of points.

    Note:
        The normalized distance d is computed as:
        d² = (x'/a)² + (y'/b)² + (z'/c)²
        where (x', y', z') are coordinates relative to rotated ellipsoid
    """
    # Convert center and axes to tensors if needed
    if not isinstance(center, torch.Tensor):
        center = torch.tensor(center, dtype=points.dtype, device=points.device)
    if not isinstance(axes, torch.Tensor):
        axes = torch.tensor(axes, dtype=points.dtype, device=points.device)

    # Translate to ellipsoid-centered coordinates
    relative_points = points - center

    # Apply inverse rotations if specified
    # Order: inverse of Z, Y, X rotations (applied in reverse: X, Y, Z)
    if psi is not None:
        R_x = rotation_matrix_x(-psi)
        relative_points = apply_rotation(relative_points, R_x)

    if phi is not None:
        R_y = rotation_matrix_y(-phi)
        relative_points = apply_rotation(relative_points, R_y)

    if theta is not None:
        R_z = rotation_matrix_z(-theta)
        relative_points = apply_rotation(relative_points, R_z)

    # Compute normalized distance: sqrt((x/a)² + (y/b)² + (z/c)²)
    normalized_coords = relative_points / axes
    distance_squared = torch.sum(normalized_coords ** 2, dim=-1)

    return torch.sqrt(distance_squared)


def ellipsoid_mask(
    points: torch.Tensor,
    center: Union[torch.Tensor, Tuple[float, float, float]],
    axes: Union[torch.Tensor, Tuple[float, float, float]],
    theta: torch.Tensor = None,
    phi: torch.Tensor = None,
    psi: torch.Tensor = None
) -> torch.Tensor:
    """
    Create boolean mask for points inside an ellipsoid.

    Args:
        points: Points to evaluate, shape (*batch_dims, 3)
        center: Ellipsoid center (x0, y0, z0)
        axes: Ellipsoid semi-axes (a, b, c)
        theta: Rotation angle around Z-axis (radians), optional
        phi: Rotation angle around Y-axis (radians), optional
        psi: Rotation angle around X-axis (radians), optional

    Returns:
        Boolean mask, True for points inside ellipsoid.
        Shape matches batch dimensions of points.
    """
    distance = ellipsoid_distance(points, center, axes, theta, phi, psi)
    return distance <= 1.0


def galactocentric_to_cylindrical(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert galactocentric Cartesian (x, y, z) to cylindrical (R, phi, z).

    Args:
        x: Galactocentric x coordinate (kpc)
        y: Galactocentric y coordinate (kpc)
        z: Galactocentric z coordinate (kpc)

    Returns:
        Tuple of (R, phi, z) where:
        - R: Cylindrical radius in xy-plane (kpc)
        - phi: Azimuthal angle (radians, 0 to 2π)
        - z: Height above galactic plane (kpc, same as input)

    Note:
        All inputs must have the same shape. Supports batching.
    """
    R = torch.sqrt(x**2 + y**2)
    phi = torch.atan2(y, x)
    return R, phi, z


def cylindrical_to_galactocentric(R: torch.Tensor, phi: torch.Tensor, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert cylindrical (R, phi, z) to galactocentric Cartesian (x, y, z).

    Args:
        R: Cylindrical radius in xy-plane (kpc)
        phi: Azimuthal angle (radians)
        z: Height above galactic plane (kpc)

    Returns:
        Tuple of (x, y, z) galactocentric coordinates (kpc)

    Note:
        All inputs must have the same shape. Supports batching.
    """
    x = R * torch.cos(phi)
    y = R * torch.sin(phi)
    return x, y, z


def galactic_to_galactocentric(
    l: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    r_sun: float = 8.5
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert galactic coordinates (l, b, d) to galactocentric Cartesian (x, y, z).

    This transformation converts from observer-centric galactic coordinates to
    galactocentric Cartesian coordinates. The Sun is located at (0, r_sun, 0)
    in the galactocentric frame.

    Args:
        l: Galactic longitude (degrees), shape (...)
        b: Galactic latitude (degrees), shape (...)
        d: Distance from Sun (kpc), shape (...)
        r_sun: Distance of Sun from Galactic center (kpc), default 8.5

    Returns:
        Tuple of (x, y, z) galactocentric coordinates (kpc), each with shape (...)

    Note:
        - All inputs must have the same shape or be broadcastable
        - Supports batching over arbitrary dimensions
        - Fully differentiable with respect to all inputs
        - Coordinate convention:
            * x points towards Galactic center from Sun
            * y points in direction of Galactic rotation
            * z points towards North Galactic Pole
        - Sun position: (x=0, y=r_sun, z=0)
    """
    # Convert degrees to radians
    l_rad = l * (torch.pi / 180.0)
    b_rad = b * (torch.pi / 180.0)

    # Trigonometric functions
    sin_l = torch.sin(l_rad)
    cos_l = torch.cos(l_rad)
    sin_b = torch.sin(b_rad)
    cos_b = torch.cos(b_rad)

    # Project distance onto galactic plane
    r_gal = d * cos_b

    # Convert to galactocentric Cartesian coordinates
    # Sun is at (x=0, y=r_sun, z=0)
    x = r_gal * sin_l
    y = r_sun - r_gal * cos_l
    z = d * sin_b

    return x, y, z


def galactocentric_to_galactic(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    r_sun: float = 8.5
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert galactocentric Cartesian (x, y, z) to galactic coordinates (l, b, d).

    This is the inverse transformation of galactic_to_galactocentric.

    Args:
        x: Galactocentric x coordinate (kpc), shape (...)
        y: Galactocentric y coordinate (kpc), shape (...)
        z: Galactocentric z coordinate (kpc), shape (...)
        r_sun: Distance of Sun from Galactic center (kpc), default 8.5

    Returns:
        Tuple of (l, b, d) where:
        - l: Galactic longitude (degrees), shape (...)
        - b: Galactic latitude (degrees), shape (...)
        - d: Distance from Sun (kpc), shape (...)

    Note:
        - All inputs must have the same shape or be broadcastable
        - Supports batching over arbitrary dimensions
        - Fully differentiable with respect to all inputs
        - Returns longitude in range [0, 360) degrees
        - Returns latitude in range [-90, 90] degrees
    """
    # Translate to Sun-centered coordinates
    # Sun is at (x=0, y=r_sun, z=0) in galactocentric frame
    x_sun = x
    y_sun = r_sun - y
    z_sun = z

    # Distance from Sun
    d = torch.sqrt(x_sun**2 + y_sun**2 + z_sun**2)

    # Galactic longitude (in radians, then convert to degrees)
    # atan2(x, y_sun) gives angle from y-axis towards x-axis
    l_rad = torch.atan2(x_sun, y_sun)
    l = l_rad * (180.0 / torch.pi)

    # Ensure longitude is in [0, 360) range
    l = torch.where(l < 0, l + 360.0, l)

    # Galactic latitude (in radians, then convert to degrees)
    # Avoid division by zero for points at origin
    eps = torch.finfo(d.dtype).eps
    sin_b = z_sun / torch.clamp(d, min=eps)
    # Clamp to avoid numerical issues with arcsin
    sin_b = torch.clamp(sin_b, -1.0, 1.0)
    b_rad = torch.asin(sin_b)
    b = b_rad * (180.0 / torch.pi)

    return l, b, d
