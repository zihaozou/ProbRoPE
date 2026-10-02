import cv2
import numpy as np
import torch


def get_camera_center(extrinsics):
    """
    Compute the camera center (optical center) in world coordinates from an extrinsics matrix.

    The camera center C is computed as C = -R^T @ t, where the extrinsics matrix is [R|t].

    Args:
        extrinsics (torch.Tensor): Camera extrinsics matrix in OpenCV convention.
            Shape can be (4, 4) for a single camera or (B, 4, 4) for a batch.
            Format: [R|t; 0 0 0 1] where R is 3x3 rotation, t is 3x1 translation.

    Returns:
        torch.Tensor: Camera center(s) in world coordinates.
            Shape is (3,) for single camera or (B, 3) for batch.
    """
    if extrinsics.dim() == 2:
        R = extrinsics[:3, :3]
        t = extrinsics[:3, 3]
        return -R.T @ t
    else:
        R = extrinsics[:, :3, :3]
        t = extrinsics[:, :3, 3]
        return -torch.bmm(R.transpose(1, 2), t.unsqueeze(-1)).squeeze(-1)


def unproject_points(points_ndc, intrinsics, extrinsics):
    """
    Unproject points from Normalized Device Coordinates (NDC) to world coordinates.

    NDC convention: x=1 is left, x=-1 is right, y=1 is top, y=-1 is bottom.
    The depth component represents the distance along the camera's z-axis.

    Supports both pixel-unit OpenCV intrinsics (fx~500, cx~256) and normalized
    intrinsics (fx~2, cx~0). The conversion is:
        x_cam = (-x_ndc * cx + cx) / fx  =  cx * (1 - x_ndc) / fx
        y_cam = (-y_ndc * cy + cy) / fy  =  cy * (1 - y_ndc) / fy
    For centered pixel intrinsics (cx = W/2) with NDC in [-1,1], this maps
    left edge (x_ndc=1)  → pixel 0   → x_cam = 0
    right edge (x_ndc=-1) → pixel W   → x_cam = W/fx ≈ 1/f_norm
    center (x_ndc=0)      → pixel W/2 → x_cam = cx/fx ≈ 0.5/f_norm

    Args:
        points_ndc (torch.Tensor): Points in NDC space with shape (N, 3).
            Each point is (x_ndc, y_ndc, depth).
        intrinsics (torch.Tensor): Camera intrinsics matrix with shape (3, 3).
            Format: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
            Accepts both pixel-unit and normalized intrinsics.
        extrinsics (torch.Tensor): Camera extrinsics matrix with shape (4, 4).
            Format: [R|t; 0 0 0 1] in OpenCV convention.

    Returns:
        torch.Tensor: Points in world coordinates with shape (N, 3).
    """

    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    x_ndc = points_ndc[:, 0]
    y_ndc = points_ndc[:, 1]
    depth = points_ndc[:, 2]


    x_cam = -cx * x_ndc / fx * depth
    y_cam = -cy * y_ndc / fy * depth
    z_cam = depth

    points_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)


    R = extrinsics[:3, :3]
    t = extrinsics[:3, 3]


    points_world = (points_cam - t) @ R

    return points_world


def intersect_skew_lines_high_dim(p, r, mask=None):
    """Intersect skew lines high dim."""
    dim = p.shape[-1]

    if mask is None:
        mask = torch.ones_like(p[..., 0])
    r = torch.nn.functional.normalize(r, dim=-1)

    eye = torch.eye(dim, device=p.device, dtype=p.dtype)[None, None]
    I_min_cov = (eye - (r[..., None] * r[..., None, :])) * mask[..., None, None]
    sum_proj = I_min_cov.matmul(p[..., None]).sum(dim=-3)


    p_intersect = torch.linalg.lstsq(I_min_cov.sum(dim=-3), sum_proj).solution[..., 0]

    if torch.any(torch.isnan(p_intersect)):
        raise ValueError("NaN encountered in intersect_skew_lines_high_dim")
    return p_intersect, r


class Rays(object):
    """
    A class for representing and manipulating 3D rays.

    Rays can be stored in two representations:
    - Point-Direction: <origin, direction> where origin is a point on the ray
      and direction is the ray's heading vector.
    - Plücker coordinates: <direction, moment> where moment = origin × direction.
      This representation is useful for certain geometric operations.

    The class provides methods to convert between representations and extract
    ray components.
    """

    def __init__(
        self,
        rays=None,
        origins=None,
        directions=None,
        moments=None,
        is_plucker=False,
        moments_rescale=1.0,
        ndc_coordinates=None,
        crop_parameters=None,
        num_patches_x=16,
        num_patches_y=16,
    ):
        """
        Initialize a Rays object.

        Can be constructed in three ways:
        1. From raw ray tensor with `rays` and `is_plucker` flag
        2. From `origins` and `directions` (creates point-direction representation)
        3. From `directions` and `moments` (creates Plücker representation)

        Args:
            rays (torch.Tensor, optional): Raw ray data with shape (..., 6).
            origins (torch.Tensor, optional): Ray origins with shape (..., 3).
            directions (torch.Tensor, optional): Ray directions with shape (..., 3).
            moments (torch.Tensor, optional): Plücker moments with shape (..., 3).
            is_plucker (bool): Whether `rays` is in Plücker coordinates. Default: False.
            moments_rescale (float): Scale factor for moment components. Default: 1.0.
            ndc_coordinates (torch.Tensor, optional): NDC coordinates with shape (..., 2).
            crop_parameters (torch.Tensor, optional): Crop params (cc_x, cc_y, width, scale).
            num_patches_x (int): Number of patches in x direction. Default: 16.
            num_patches_y (int): Number of patches in y direction. Default: 16.

        Raises:
            Exception: If an invalid combination of arguments is provided.
        """
        if rays is not None:
            self.rays = rays
            self._is_plucker = is_plucker
        elif origins is not None and directions is not None:
            self.rays = torch.cat((origins, directions), dim=-1)
            self._is_plucker = False
        elif directions is not None and moments is not None:
            self.rays = torch.cat((directions, moments), dim=-1)
            self._is_plucker = True
        else:
            raise Exception("Invalid combination of arguments")


        self.num_patches_x = num_patches_x
        self.num_patches_y = num_patches_y

        if moments_rescale != 1.0:
            self.rescale_moments(moments_rescale)

        if ndc_coordinates is not None:
            self.ndc_coordinates = ndc_coordinates
        elif crop_parameters is not None:

            xy_grid = compute_ndc_coordinates(
                crop_parameters,
                num_patches_x=num_patches_x,
                num_patches_y=num_patches_y,
            )[..., :2]
            xy_grid = xy_grid.reshape(*xy_grid.shape[:-3], -1, 2)
            self.ndc_coordinates = xy_grid
        else:
            self.ndc_coordinates = None

    def __getitem__(self, index):
        return Rays(
            rays=self.rays[index],
            is_plucker=self._is_plucker,
            ndc_coordinates=(
                self.ndc_coordinates[index]
                if self.ndc_coordinates is not None
                else None
            ),
            num_patches_x=self.num_patches_x,
            num_patches_y=self.num_patches_y,
        )

    def to_spatial(self, include_ndc_coordinates=False):
        """
        Convert rays to spatial (image-like) tensor layout.

        Reshapes rays from flattened format to a 2D grid format suitable for
        convolutions or spatial processing.

        Args:
            include_ndc_coordinates (bool): If True, append NDC coordinates as
                additional channels, resulting in shape (..., 8, H, W). Default: False.

        Returns:
            torch.Tensor: Rays in spatial format with shape (..., 6, H, W),
                or (..., 8, H, W) if include_ndc_coordinates is True.
                The rays are converted to Plücker representation before reshaping.
        """
        rays = self.to_plucker().rays
        *batch_dims, P, D = rays.shape
        H = self.num_patches_y
        W = self.num_patches_x
        assert H * W == P, f"num_patches_y({H}) * num_patches_x({W}) = {H*W} != P({P})"
        rays = torch.transpose(rays, -1, -2)
        rays = rays.reshape(*batch_dims, D, H, W)
        if include_ndc_coordinates:
            ndc_coords = self.ndc_coordinates.transpose(-1, -2)
            ndc_coords = ndc_coords.reshape(*batch_dims, 2, H, W)
            rays = torch.cat((rays, ndc_coords), dim=-3)
        return rays

    def rescale_moments(self, scale):
        """
        Rescale the moment (Plücker normal) component of the rays by a scalar.

        This can be useful for normalizing moment distributions or for numerical
        stability when moments have very small magnitudes.

        Note: This operation modifies the rays in place if already in Plücker form.

        Args:
            scale (float): Scale factor to apply to moment components.

        Returns:
            Rays: Self (if in Plücker form) or a new Rays object in Plücker form.
        """
        if self.is_plucker:
            self.rays[..., 3:] *= scale
            return self
        else:
            return self.to_plucker().rescale_moments(scale)

    @classmethod
    def from_spatial(cls, rays, moments_rescale=1.0, ndc_coordinates=None):
        """
        Create a Rays object from spatial (image-like) tensor layout.

        Inverse of `to_spatial()`. Converts rays from 2D grid format back to
        flattened format.

        Args:
            rays (torch.Tensor): Rays in spatial format with shape (..., 6, H, W).
                Expected to be in Plücker representation.
            moments_rescale (float): Scale factor for moment components. Default: 1.0.
            ndc_coordinates (torch.Tensor, optional): NDC coordinates with shape (..., H*W, 2).

        Returns:
            Rays: New Rays object with shape (..., H * W, 6) in Plücker form.
        """
        *batch_dims, D, H, W = rays.shape
        rays = rays.reshape(*batch_dims, D, H * W)
        rays = torch.transpose(rays, -1, -2)
        return cls(
            rays=rays,
            is_plucker=True,
            moments_rescale=moments_rescale,
            ndc_coordinates=ndc_coordinates,
        )

    def to_point_direction(self, normalize_moment=True):
        """
        Convert rays to point-direction representation <origin, direction>.

        For Plücker rays, computes the point closest to origin on the ray using:
        point = direction × moment (cross product).

        Args:
            normalize_moment (bool): If True, normalize moment by direction norm
                before computing the point. Default: True.

        Returns:
            Rays: A new Rays object in point-direction form with shape (..., 6),
                where rays[..., :3] are origins and rays[..., 3:] are directions.
        """
        if self._is_plucker:
            direction = torch.nn.functional.normalize(self.rays[..., :3], dim=-1)
            moment = self.rays[..., 3:]
            if normalize_moment:
                c = torch.linalg.norm(direction, dim=-1, keepdim=True)
                moment = moment / c
            points = torch.cross(direction, moment, dim=-1)
            return Rays(
                rays=torch.cat((points, direction), dim=-1),
                is_plucker=False,
                ndc_coordinates=self.ndc_coordinates,
                num_patches_x=self.num_patches_x,
                num_patches_y=self.num_patches_y,
            )
        else:
            return self

    def to_plucker(self):
        """
        Convert rays to Plücker representation <direction, moment>.

        Plücker coordinates represent a ray as (d, m) where d is the normalized
        direction and m = origin × direction is the moment (Plücker normal).
        This representation is invariant to the choice of point on the ray.

        Returns:
            Rays: A new Rays object in Plücker form with shape (..., 6),
                where rays[..., :3] are directions and rays[..., 3:] are moments.
        """
        if self.is_plucker:
            return self
        else:
            ray = self.rays.clone()
            ray_origins = ray[..., :3]
            ray_directions = ray[..., 3:]

            ray_directions = ray_directions / ray_directions.norm(dim=-1, keepdim=True)
            plucker_normal = torch.cross(ray_origins, ray_directions, dim=-1)
            new_ray = torch.cat([ray_directions, plucker_normal], dim=-1)
            return Rays(
                rays=new_ray, is_plucker=True, ndc_coordinates=self.ndc_coordinates,
                num_patches_x=self.num_patches_x, num_patches_y=self.num_patches_y,
            )

    def get_directions(self, normalize=True):
        """
        Extract ray direction vectors.

        Args:
            normalize (bool): If True, return unit-length directions. Default: True.

        Returns:
            torch.Tensor: Direction vectors with shape (..., 3).
        """
        if self.is_plucker:
            directions = self.rays[..., :3]
        else:
            directions = self.rays[..., 3:]
        if normalize:
            directions = torch.nn.functional.normalize(directions, dim=-1)
        return directions

    def get_origins(self):
        """
        Extract ray origin points.

        For Plücker rays, computes the point on each ray closest to the origin.

        Returns:
            torch.Tensor: Origin points with shape (..., 3).
        """
        if self.is_plucker:
            origins = self.to_point_direction().get_origins()
        else:
            origins = self.rays[..., :3]
        return origins

    def get_moments(self):
        """
        Extract Plücker moment vectors.

        For point-direction rays, converts to Plücker form first.

        Returns:
            torch.Tensor: Moment vectors (origin × direction) with shape (..., 3).
        """
        if self.is_plucker:
            moments = self.rays[..., 3:]
        else:
            moments = self.to_plucker().get_moments()
        return moments

    def get_ndc_coordinates(self):
        """
        Get the NDC coordinates associated with these rays.

        Returns:
            torch.Tensor or None: NDC coordinates with shape (..., 2), or None if not set.
        """
        return self.ndc_coordinates

    @property
    def is_plucker(self):
        """bool: True if rays are in Plücker representation, False if point-direction."""
        return self._is_plucker

    @property
    def device(self):
        """torch.device: The device where ray data is stored."""
        return self.rays.device

    def __repr__(self, *args, **kwargs):
        """String representation showing ray type and tensor info."""
        ray_str = self.rays.__repr__(*args, **kwargs)[6:]
        if self._is_plucker:
            return "PluRay" + ray_str
        else:
            return "DirRay" + ray_str

    def to(self, device):
        """
        Move ray data to the specified device.

        Args:
            device: Target device (e.g., 'cuda', 'cpu', torch.device).
        """
        self.rays = self.rays.to(device)

    def clone(self):
        """
        Create a deep copy of this Rays object.

        Returns:
            Rays: A new Rays object with copied data.
        """
        return Rays(rays=self.rays.clone(), is_plucker=self._is_plucker)

    @property
    def shape(self):
        """torch.Size: Shape of the underlying ray tensor."""
        return self.rays.shape

    def visualize(self):
        """
        Generate visualization-friendly representations of rays.

        Normalizes directions and moments to unit length and maps from [-1, 1] to [0, 1]
        for visualization as RGB colors.

        Returns:
            tuple:
                - directions (torch.Tensor): Normalized directions mapped to [0, 1], on CPU.
                - moments (torch.Tensor): Normalized moments mapped to [0, 1], on CPU.
        """
        directions = torch.nn.functional.normalize(self.get_directions(), dim=-1).cpu()
        moments = torch.nn.functional.normalize(self.get_moments(), dim=-1).cpu()
        return (directions + 1) / 2, (moments + 1) / 2


def cameras_to_rays(
    intrinsics,
    extrinsics,
    crop_parameters=None,
    use_half_pix=True,
    use_plucker=True,
    num_patches_x=16,
    num_patches_y=16,
):
    """
    Unprojects rays from camera center to grid on image plane.

    Args:
        intrinsics: (B, 3, 3) camera intrinsics matrices.
        extrinsics: (B, 4, 4) camera extrinsics matrices [R|t].
        crop_parameters: Crop parameters in NDC (cc_x, cc_y, crop_width, scale).
            Shape is (B, 4). If None, uses full image.
        use_half_pix: If True, use half pixel offset (Default: True).
        use_plucker: If True, return rays in plucker coordinates (Default: True).
        num_patches_x: Number of patches in x direction (Default: 16).
        num_patches_y: Number of patches in y direction (Default: 16).

    Returns:
        Rays: Ray object with shape (B, num_patches_x * num_patches_y, 6)
    """
    device = intrinsics.device
    batch_size = intrinsics.shape[0]


    if crop_parameters is None:
        crop_parameters_list = [None] * batch_size
    else:
        crop_parameters_list = crop_parameters

    unprojected = []
    for i in range(batch_size):
        crop_param = crop_parameters_list[i] if crop_parameters is not None else None
        xyd_grid = compute_ndc_coordinates(
            crop_parameters=crop_param,
            use_half_pix=use_half_pix,
            num_patches_x=num_patches_x,
            num_patches_y=num_patches_y,
            device=device,
        )

        unprojected.append(
            unproject_points(
                xyd_grid.reshape(-1, 3),
                intrinsics[i],
                extrinsics[i],
            )
        )

    unprojected = torch.stack(unprojected, dim=0)
    origins = get_camera_center(extrinsics).unsqueeze(1)
    origins = origins.repeat(1, num_patches_x * num_patches_y, 1)
    directions = unprojected - origins

    rays = Rays(
        origins=origins,
        directions=directions,
        crop_parameters=crop_parameters,
        num_patches_x=num_patches_x,
        num_patches_y=num_patches_y,
    )
    if use_plucker:
        return rays.to_plucker()
    return rays


def rays_to_cameras(
    rays,
    crop_parameters=None,
    num_patches_x=16,
    num_patches_y=16,
    use_half_pix=True,
    sampled_ray_idx=None,
    intrinsics=None,
    focal_length=(3.453,),
):
    """
    Convert rays to camera intrinsics and extrinsics.

    If intrinsics are provided, will use those. Otherwise will construct
    intrinsics from the provided focal_length(s). Dataset default is 3.453.

    Args:
        rays (Rays): (N, P, 6)
        crop_parameters (torch.Tensor): (N, 4) or None
        num_patches_x: Number of patches in x direction (Default: 16).
        num_patches_y: Number of patches in y direction (Default: 16).
        use_half_pix: If True, use half pixel offset (Default: True).
        sampled_ray_idx: Indices of sampled rays (optional).
        intrinsics: (N, 3, 3) camera intrinsics or None.
        focal_length: Focal length(s) to use if intrinsics not provided.

    Returns:
        intrinsics: (N, 3, 3) camera intrinsics matrices
        extrinsics: (N, 4, 4) camera extrinsics matrices [R|t]
    """
    device = rays.device
    batch_size = rays.shape[0]
    origins = rays.get_origins()
    directions = rays.get_directions()
    camera_centers, _ = intersect_skew_lines_high_dim(origins, directions)


    if intrinsics is None:
        if len(focal_length) == 1:
            focal_length = list(focal_length) * batch_size
        intrinsics = torch.zeros(batch_size, 3, 3, device=device)
        for i in range(batch_size):
            intrinsics[i] = torch.tensor(
                [[focal_length[i], 0, 0], [0, focal_length[i], 0], [0, 0, 1]],
                device=device,
                dtype=torch.float32,
            )


    I_extrinsics = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)

    I_patch_rays = cameras_to_rays(
        intrinsics=intrinsics,
        extrinsics=I_extrinsics,
        num_patches_x=num_patches_x,
        num_patches_y=num_patches_y,
        use_half_pix=use_half_pix,
        crop_parameters=crop_parameters,
    ).get_directions()

    if sampled_ray_idx is not None:
        I_patch_rays = I_patch_rays[:, sampled_ray_idx]


    R = torch.zeros(batch_size, 3, 3, device=device)
    for i in range(batch_size):
        R[i] = compute_optimal_rotation_alignment(
            directions[i],
            I_patch_rays[i],
        )


    t = -torch.bmm(R, camera_centers.unsqueeze(2)).squeeze(2)

    extrinsics = torch.zeros(batch_size, 4, 4, device=device)
    extrinsics[:, :3, :3] = R
    extrinsics[:, :3, 3] = t
    extrinsics[:, 3, 3] = 1.0

    return intrinsics, extrinsics


def ql_decomposition(A):
    """
    Compute QL decomposition of a matrix.

    Decomposes A = Q @ L where Q is orthogonal and L is lower triangular.
    Uses QR decomposition with appropriate permutations.

    Reference: https://www.reddit.com/r/learnmath/comments/v1crd7/linear_algebra_qr_to_ql_decomposition/

    Args:
        A (torch.Tensor): Input matrix with shape (3, 3).

    Returns:
        tuple:
            - Q (torch.Tensor): Orthogonal matrix with shape (3, 3).
            - L (torch.Tensor): Lower triangular matrix with shape (3, 3).
    """
    P = torch.tensor([[0, 0, 1], [0, 1, 0], [1, 0, 0]], device=A.device).float()
    A_tilde = torch.matmul(A, P)
    Q_tilde, R_tilde = torch.linalg.qr(A_tilde)
    Q = torch.matmul(Q_tilde, P)
    L = torch.matmul(torch.matmul(P, R_tilde), P)
    d = torch.diag(L)
    Q[:, 0] *= torch.sign(d[0])
    Q[:, 1] *= torch.sign(d[1])
    Q[:, 2] *= torch.sign(d[2])
    L[0] *= torch.sign(d[0])
    L[1] *= torch.sign(d[1])
    L[2] *= torch.sign(d[2])
    return Q, L


def rays_to_cameras_homography(
    rays,
    crop_parameters=None,
    num_patches_x=16,
    num_patches_y=16,
    use_half_pix=True,
    sampled_ray_idx=None,
    reproj_threshold=0.2,
):
    """
    Convert rays to camera intrinsics and extrinsics using homography estimation.

    Args:
        rays (Rays): (N, P, 6)
        crop_parameters (torch.Tensor): (N, 4) or None
        num_patches_x: Number of patches in x direction (Default: 16).
        num_patches_y: Number of patches in y direction (Default: 16).
        use_half_pix: If True, use half pixel offset (Default: True).
        sampled_ray_idx: Indices of sampled rays (optional).
        reproj_threshold: RANSAC reprojection threshold (Default: 0.2).

    Returns:
        intrinsics: (N, 3, 3) camera intrinsics matrices
        extrinsics: (N, 4, 4) camera extrinsics matrices [R|t]
    """
    device = rays.device
    batch_size = rays.shape[0]
    origins = rays.get_origins()
    directions = rays.get_directions()
    camera_centers, _ = intersect_skew_lines_high_dim(origins, directions)


    I_intrinsics = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    I_extrinsics = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)

    I_patch_rays = cameras_to_rays(
        intrinsics=I_intrinsics,
        extrinsics=I_extrinsics,
        num_patches_x=num_patches_x,
        num_patches_y=num_patches_y,
        use_half_pix=use_half_pix,
        crop_parameters=crop_parameters,
    ).get_directions()

    if sampled_ray_idx is not None:
        I_patch_rays = I_patch_rays[:, sampled_ray_idx]


    Rs = []
    focal_lengths = []
    principal_points = []
    for i in range(batch_size):
        R, f, pp = compute_optimal_rotation_intrinsics(
            directions[i],
            I_patch_rays[i],
            reproj_threshold=reproj_threshold,
        )
        Rs.append(R)
        focal_lengths.append(f)
        principal_points.append(pp)

    R = torch.stack(Rs).to(device)
    focal_lengths = torch.stack(focal_lengths).to(device)
    principal_points = torch.stack(principal_points).to(device)


    intrinsics = torch.zeros(batch_size, 3, 3, device=device)
    intrinsics[:, 0, 0] = focal_lengths[:, 0]
    intrinsics[:, 1, 1] = focal_lengths[:, 1]
    intrinsics[:, 0, 2] = principal_points[:, 0]
    intrinsics[:, 1, 2] = principal_points[:, 1]
    intrinsics[:, 2, 2] = 1.0


    t = -torch.bmm(R, camera_centers.unsqueeze(2)).squeeze(2)

    extrinsics = torch.zeros(batch_size, 4, 4, device=device)
    extrinsics[:, :3, :3] = R
    extrinsics[:, :3, 3] = t
    extrinsics[:, 3, 3] = 1.0

    return intrinsics, extrinsics


def compute_optimal_rotation_alignment(A, B):
    """
    Compute the optimal orthogonal matrix that aligns B to A.

    Finds R that minimizes the Frobenius norm || A - B @ R ||_F using
    SVD-based orthogonal Procrustes solution.

    Supports both proper rotations (det=+1) and improper rotations (det=-1,
    i.e. rotation + reflection), which is needed for OpenCV extrinsics that
    may have det(R)=-1 from coordinate convention conversions.

    Args:
        A (torch.Tensor): Target vectors with shape (N, 3).
        B (torch.Tensor): Source vectors with shape (N, 3).

    Returns:
        torch.Tensor: Optimal orthogonal matrix R with shape (3, 3).
            Applying B @ R gives vectors closest to A in least-squares sense.
            det(R) may be +1 or -1 depending on alignment.
    """
    H = B.T @ A
    U, _, Vh = torch.linalg.svd(H, full_matrices=True)
    return U @ Vh


def rq(M):
    """
    Compute RQ decomposition of a matrix.

    Decomposes M = R @ Q where R is upper triangular and Q is orthogonal.
    Uses QR decomposition with row/column flipping.

    Reference: https://stackoverflow.com/questions/76735175/how-do-i-compute-rq-facorization-using-qr-python-library

    Args:
        M (np.ndarray or torch.Tensor): Input matrix (will be converted to numpy).

    Returns:
        tuple:
            - R (np.ndarray): Upper triangular matrix.
            - Q (np.ndarray): Orthogonal matrix.
    """
    Q, R = np.linalg.qr(np.flipud(M).T)

    R = np.flipud(R.T)
    R = np.fliplr(R)

    Q = Q.T
    Q = np.flipud(Q)

    return R.copy(), Q.copy()


def compute_optimal_rotation_intrinsics(
    rays_origin, rays_target, z_threshold=1e-4, reproj_threshold=0.2
):
    """
    Estimate rotation and camera intrinsics from ray correspondences using homography.

    Projects rays to 2D (by dividing by z), finds a homography between the 2D points,
    then decomposes the homography into rotation and intrinsic parameters using RQ
    decomposition.

    Note: The estimated focal length may be inverted (1/f instead of f) due to the
    homography parameterization.

    Args:
        rays_origin (torch.Tensor): Source ray directions with shape (N, 3).
        rays_target (torch.Tensor): Target ray directions with shape (N, 3).
        z_threshold (float): Minimum absolute z-value for a ray to be considered
            valid (avoids division by near-zero). Default: 1e-4.
        reproj_threshold (float): RANSAC reprojection error threshold for
            homography estimation. Default: 0.2.

    Returns:
        tuple:
            - R (torch.Tensor): Rotation matrix with shape (3, 3).
            - focal_length (torch.Tensor): Focal lengths (fx, fy) with shape (2,).
            - principal_point (torch.Tensor): Principal point (cx, cy) with shape (2,).
    """
    device = rays_origin.device
    z_mask = torch.logical_and(
        torch.abs(rays_target) > z_threshold, torch.abs(rays_origin) > z_threshold
    )[:, 2]
    rays_target = rays_target[z_mask]
    rays_origin = rays_origin[z_mask]
    rays_origin = rays_origin[:, :2] / rays_origin[:, -1:]
    rays_target = rays_target[:, :2] / rays_target[:, -1:]

    A, _ = cv2.findHomography(
        rays_origin.cpu().numpy(),
        rays_target.cpu().numpy(),
        cv2.RANSAC,
        reproj_threshold,
    )
    A = torch.from_numpy(A).float().to(device)
    if torch.linalg.det(A) < 0:
        A = -A

    L, R = rq(A)
    R *= ((L.diagonal() > 0) * 2 - 1)[:, None]
    L = L / L[2][2]
    L[0][0], L[1][1] = np.abs(L[0][0]), np.abs(L[1][1])
    R, L = torch.tensor(R.T), torch.tensor(L)
    f = torch.stack((L[0][0], L[1][1]))
    pp = torch.stack((L[0][2], L[1][2]))

    return R, f, pp


def compute_ndc_coordinates(
    crop_parameters=None,
    use_half_pix=True,
    num_patches_x=16,
    num_patches_y=16,
    device=None,
):
    """
    Compute a grid of Normalized Device Coordinates (NDC) for ray generation.

    Creates a 2D grid of (x, y, depth) coordinates in NDC space, where:
    - Top-left corner is (1, 1)
    - Bottom-right corner is (-1, -1)
    - Depth is set to 1.0 for all points

    The grid can be cropped/scaled according to crop_parameters.

    Args:
        crop_parameters (torch.Tensor, optional): Crop specification with shape (4,)
            containing (cc_x, cc_y, crop_width, scale). If None, uses full NDC range.
            If shape is (B, 4), returns stacked grids for each set of parameters.
        use_half_pix (bool): If True, offset grid by half pixel to sample at
            patch centers. Default: True.
        num_patches_x (int): Number of grid points in x direction. Default: 16.
        num_patches_y (int): Number of grid points in y direction. Default: 16.
        device (torch.device, optional): Device for output tensor. Inferred from
            crop_parameters if provided.

    Returns:
        torch.Tensor: NDC coordinate grid with shape (num_patches_y, num_patches_x, 3)
            or (B, num_patches_y, num_patches_x, 3) if batched crop_parameters.
    """
    if crop_parameters is None:
        cc_x, cc_y, width = 0, 0, 2
    else:
        if len(crop_parameters.shape) > 1:
            return torch.stack(
                [
                    compute_ndc_coordinates(
                        crop_parameters=crop_param,
                        use_half_pix=use_half_pix,
                        num_patches_x=num_patches_x,
                        num_patches_y=num_patches_y,
                    )
                    for crop_param in crop_parameters
                ],
                dim=0,
            )
        device = crop_parameters.device
        cc_x, cc_y, width, _ = crop_parameters

    dx = 1 / num_patches_x
    dy = 1 / num_patches_y
    if use_half_pix:
        min_y = 1 - dy
        max_y = -min_y
        min_x = 1 - dx
        max_x = -min_x
    else:
        min_y = min_x = 1
        max_y = -1 + 2 * dy
        max_x = -1 + 2 * dx

    y, x = torch.meshgrid(
        torch.linspace(min_y, max_y, num_patches_y, dtype=torch.float32, device=device),
        torch.linspace(min_x, max_x, num_patches_x, dtype=torch.float32, device=device),
        indexing="ij",
    )
    x_prime = x * width / 2 - cc_x
    y_prime = y * width / 2 - cc_y
    xyd_grid = torch.stack([x_prime, y_prime, torch.ones_like(x)], dim=-1)
    return xyd_grid
