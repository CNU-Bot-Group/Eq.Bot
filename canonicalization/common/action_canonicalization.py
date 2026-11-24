"""Representation-aware action maps in pixel or physical coordinates.

Frames map canonical coordinates to original coordinates. Camera image angles
alone are not workspace frames. Joint-space maps are conditional on the caller's
fixed IK branch and transformation-invariant reachable domain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


ActionRepresentation = Literal[
    "delta_rotvec", "absolute_rotvec", "absolute_euler_xyz",
    "bimanual_joint", "pixel_pick_place", "spatial_maps",
]


def _check_tensor(value: Tensor, name: str) -> None:
    if not isinstance(value, Tensor) or value.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"{name} must be a float32 or float64 tensor in physical coordinates")
    if value.numel() == 0 or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be nonempty and finite")


@dataclass(frozen=True)
class ActionFrame:
    """Rigid map p_original = rotation @ p_canonical + translation.

    Shapes are (..., 3, 3) and (..., 3), with identical batch dimensions.
    For actions [B, H, D], frame batch dimensions may be [], [B], or [B, H].
    """

    rotation: Tensor
    translation: Tensor

    def __post_init__(self) -> None:
        _check_tensor(self.rotation, "frame.rotation")
        _check_tensor(self.translation, "frame.translation")
        if self.rotation.shape[-2:] != (3, 3):
            raise ValueError("frame.rotation must have shape (..., 3, 3)")
        if self.translation.shape != self.rotation.shape[:-2] + (3,):
            raise ValueError("frame.translation must match the rotation's batch dimensions")
        if self.rotation.dtype != self.translation.dtype or self.rotation.device != self.translation.device:
            raise ValueError("frame rotation and translation must share dtype and device")
        with torch.autocast(device_type=self.rotation.device.type, enabled=False):
            identity = torch.eye(3, dtype=self.rotation.dtype, device=self.rotation.device)
            if not torch.allclose(
                self.rotation.transpose(-1, -2) @ self.rotation,
                identity.expand_as(self.rotation), atol=1e-5, rtol=0,
            ):
                raise ValueError("frame.rotation must be orthogonal")
            if not torch.allclose(
                torch.linalg.det(self.rotation), torch.ones_like(self.translation[..., 0]), atol=1e-5, rtol=0,
            ):
                raise ValueError("frame.rotation must be a proper rotation; reflections are unsupported")

    @classmethod
    def from_yaw(cls, angle_radians: Tensor, center: Tensor) -> ActionFrame:
        """Construct a rotation about an explicitly supplied workspace/pixel center."""
        _check_tensor(angle_radians, "angle_radians")
        _check_tensor(center, "center")
        if center.shape[-1:] != (3,):
            raise ValueError("center must have shape (..., 3)")
        if center.device != angle_radians.device or center.dtype != angle_radians.dtype:
            raise ValueError("center and angle_radians must share dtype and device")
        batch = torch.broadcast_shapes(angle_radians.shape, center.shape[:-1])
        angle = angle_radians.expand(batch)
        center = center.expand(batch + (3,))
        cosine, sine = torch.cos(angle), torch.sin(angle)
        zero, one = torch.zeros_like(angle), torch.ones_like(angle)
        rotation = torch.stack((cosine, -sine, zero, sine, cosine, zero, zero, zero, one), -1)
        rotation = rotation.reshape(batch + (3, 3))
        with torch.autocast(device_type=rotation.device.type, enabled=False):
            return cls(rotation, center - (rotation @ center.unsqueeze(-1)).squeeze(-1))

    @classmethod
    def from_image_rotation(cls, rotation_degrees: Tensor, image_size: tuple[int, int]) -> ActionFrame:
        """Convert the SQ/VLA stored image rotation to a pixel recovery frame.

        image_size is (height, width) of the square-padded input, before either
        internal symmetric padding or the optional bottom/right guard pixels.
        Pixel vectors use (x, y) = (column, row), unlike point_canonicalize's
        (row, column) arguments.
        Kornia recovery uses the stored angle; the right/down pixel basis
        requires its negative in from_yaw. This is not a robot workspace frame.
        """
        _check_tensor(rotation_degrees, "rotation_degrees")
        if len(image_size) != 2 or any(
            not isinstance(size, int) or isinstance(size, bool) or size < 1 for size in image_size
        ):
            raise ValueError("image_size must contain positive integer height and width")
        height, width = image_size
        center = rotation_degrees.new_tensor([width / 2, height / 2, 0])
        return cls.from_yaw(-torch.deg2rad(rotation_degrees), center)

    @classmethod
    def from_matrix(cls, matrix: Tensor) -> ActionFrame:
        """Read a homogeneous transform; no frame calibration is inferred."""
        _check_tensor(matrix, "matrix")
        if matrix.shape[-2:] != (4, 4):
            raise ValueError("matrix must have shape (..., 4, 4)")
        expected = matrix.new_tensor([0, 0, 0, 1]).expand_as(matrix[..., 3, :])
        if not torch.allclose(matrix[..., 3, :], expected, atol=1e-6, rtol=0):
            raise ValueError("matrix must have homogeneous bottom row [0, 0, 0, 1]")
        return cls(matrix[..., :3, :3], matrix[..., :3, 3])

    def inverse(self) -> ActionFrame:
        rotation = self.rotation.transpose(-1, -2)
        with torch.autocast(device_type=rotation.device.type, enabled=False):
            return ActionFrame(rotation, -(rotation @ self.translation.unsqueeze(-1)).squeeze(-1))

    def compose(self, other: ActionFrame) -> ActionFrame:
        """Return self after other (H_self @ H_other), using tensor broadcasting."""
        if self.rotation.dtype != other.rotation.dtype or self.rotation.device != other.rotation.device:
            raise ValueError("composed frames must share dtype and device")
        with torch.autocast(device_type=self.rotation.device.type, enabled=False):
            rotation = self.rotation @ other.rotation
            translation = (self.rotation @ other.translation.unsqueeze(-1)).squeeze(-1) + self.translation
            return ActionFrame(rotation, translation)


class JointKinematics(Protocol):
    """Caller-supplied kinematics for one six-joint arm in the shared workspace.

    inverse must use a fixed, deterministic branch, not a changing nearest-seed
    heuristic. The caller must establish invariance of the reachable domain for
    the chosen group. Tensor operations must remain differentiable if required
    by training; this interface does not make an external IK solver differentiable.
    """

    def forward(self, joints: Tensor) -> Tensor:
        """Map (..., 6) joint coordinates to (..., 4, 4) workspace poses."""
        ...

    def inverse(self, poses: Tensor) -> Tensor:
        """Map (..., 4, 4) poses to (..., 6) joints on the selected branch."""
        ...

    def in_domain(self, joints: Tensor) -> Tensor:
        """Return a boolean tensor of shape (...) for the selected feasible domain."""
        ...


def _skew(vector: Tensor) -> Tensor:
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(vector.shape[:-1] + (3, 3))


def rotvec_to_matrix(vector: Tensor) -> Tensor:
    """Rodrigues' formula with finite values and gradients at zero angle."""
    angle = torch.linalg.vector_norm(vector, dim=-1)
    cross = _skew(vector)
    identity = torch.eye(3, dtype=vector.dtype, device=vector.device)
    a = torch.sinc(angle / torch.pi)[..., None, None]
    b = (0.5 * torch.sinc(angle / (2 * torch.pi)).square())[..., None, None]
    return identity + a * cross + b * (cross @ cross)


def matrix_to_rotvec(matrix: Tensor) -> Tensor:
    """Return a principal rotation vector, including stable half-turn handling.

    Equivalent rotation-vector branches need not round-trip numerically; their
    rotation matrices do. The principal chart is discontinuous at half turns.
    """
    m00, m01, m02 = matrix[..., 0, :].unbind(-1)
    m10, m11, m12 = matrix[..., 1, :].unbind(-1)
    m20, m21, m22 = matrix[..., 2, :].unbind(-1)
    squared = torch.stack((1 + m00 + m11 + m22, 1 + m00 - m11 - m22,
                           1 - m00 + m11 - m22, 1 - m00 - m11 + m22), -1)
    # Only the largest component is selected; flooring the others avoids sqrt'(0).
    magnitude = torch.sqrt(squared.clamp_min(torch.finfo(matrix.dtype).eps))
    candidates = torch.stack((
        torch.stack((squared[..., 0], m21 - m12, m02 - m20, m10 - m01), -1),
        torch.stack((m21 - m12, squared[..., 1], m10 + m01, m02 + m20), -1),
        torch.stack((m02 - m20, m10 + m01, squared[..., 2], m12 + m21), -1),
        torch.stack((m10 - m01, m20 + m02, m21 + m12, squared[..., 3]), -1),
    ), -2) / (2 * magnitude[..., :, None])
    index = magnitude.argmax(-1)[..., None, None].expand(magnitude.shape[:-1] + (1, 4))
    quaternion = candidates.gather(-2, index).squeeze(-2)
    quaternion = quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    quaternion = torch.where(quaternion[..., :1] < 0, -quaternion, quaternion)
    half_angle = torch.atan2(torch.linalg.vector_norm(quaternion[..., 1:], dim=-1), quaternion[..., 0])
    return quaternion[..., 1:] * (2 / torch.sinc(half_angle / torch.pi))[..., None]


def euler_xyz_to_matrix(angles: Tensor) -> Tensor:
    """Extrinsic XYZ Euler angles: Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    roll, pitch, yaw = angles.unbind(-1)
    cr, sr = torch.cos(roll), torch.sin(roll)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    return torch.stack((cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,
                        sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,
                        -sp, cp * sr, cp * cr), -1).reshape(angles.shape[:-1] + (3, 3))


def matrix_to_euler_xyz(matrix: Tensor) -> Tensor:
    """Encode XYZ Euler angles; use roll=0 at gimbal lock."""
    cosine_pitch = torch.linalg.vector_norm(matrix[..., :2, 0], dim=-1)
    singular = cosine_pitch < (1e-6 if matrix.dtype == torch.float32 else 1e-10)
    zero, one = torch.zeros_like(cosine_pitch), torch.ones_like(cosine_pitch)
    roll = torch.atan2(torch.where(singular, zero, matrix[..., 2, 1]),
                       torch.where(singular, one, matrix[..., 2, 2]))
    yaw = torch.atan2(torch.where(singular, zero, matrix[..., 1, 0]),
                      torch.where(singular, one, matrix[..., 0, 0]))
    yaw = torch.where(singular, torch.atan2(-matrix[..., 0, 1], matrix[..., 1, 1]), yaw)
    return torch.stack((roll, torch.atan2(-matrix[..., 2, 0], cosine_pitch), yaw), -1)


def _aligned_frame(frame: ActionFrame, actions: Tensor, leading_shape: tuple[int, ...]) -> tuple[Tensor, Tensor]:
    if not isinstance(frame, ActionFrame):
        raise TypeError("frame must be an ActionFrame; independent camera angles are not a workspace transform")
    batch = frame.rotation.shape[:-2]
    if len(batch) > len(leading_shape) or any(a not in (1, b) for a, b in zip(batch, leading_shape)):
        raise ValueError(f"frame batch shape {tuple(batch)} must be a prefix of action batch shape {leading_shape}")
    extra = (1,) * (len(leading_shape) - len(batch))
    rotation = frame.rotation.to(device=actions.device, dtype=actions.dtype).reshape(batch + extra + (3, 3))
    translation = frame.translation.to(device=actions.device, dtype=actions.dtype).reshape(batch + extra + (3,))
    return rotation, translation


class ActionCanonicalizer(nn.Module):
    """Unified, parameter-free canonicalize/decanonicalize interface.

    Vector actions have shape (..., D). Cartesian inputs are decoded physical
    commands (meters/radians), never normalized network outputs. Spatial-map
    frames use the original square canvas center (S/2, S/2). Border mode
    samples [B, K, S, S] maps like SQ/VLA and is not invertible at edges. Preserve
    mode permutes a caller-padded [B, K, S+1, S+1] canvas without dropping pixels.
    canonicalize maps original -> canonical; forward/decanonicalize map back.
    """

    def __init__(self, representation: ActionRepresentation, *,
                 kinematics: Sequence[JointKinematics] | None = None,
                 angle_channels: Literal["scalar", "relative", "absolute"] = "relative",
                 spatial_map_boundary: Literal["border", "preserve"] = "border",
                 tolerance: float = 1e-5):
        super().__init__()
        dimensions = {"delta_rotvec": 7, "absolute_rotvec": 7, "absolute_euler_xyz": 7,
                      "bimanual_joint": 14, "pixel_pick_place": 6, "spatial_maps": None}
        if representation not in dimensions:
            raise ValueError(f"Unknown action representation: {representation}")
        if not 0 < tolerance < 1:
            raise ValueError("tolerance must be finite and between zero and one")
        if angle_channels not in ("scalar", "relative", "absolute"):
            raise ValueError("angle_channels must be scalar, relative, or absolute")
        if spatial_map_boundary not in ("border", "preserve"):
            raise ValueError("spatial_map_boundary must be border or preserve")
        if spatial_map_boundary == "preserve" and representation != "spatial_maps":
            raise ValueError("spatial_map_boundary='preserve' is only valid for spatial_maps")
        if representation == "bimanual_joint":
            if kinematics is None or len(kinematics) != 2:
                raise ValueError("bimanual_joint requires two explicitly configured FK/IK/domain adapters")
            for adapter in kinematics:
                if not all(callable(getattr(adapter, name, None)) for name in ("forward", "inverse", "in_domain")):
                    raise TypeError("Each kinematics adapter must provide forward, inverse, and in_domain")
        elif kinematics is not None:
            raise ValueError("kinematics is only valid for bimanual_joint")
        self.representation = representation
        self.action_dim = dimensions[representation]
        self.kinematics = tuple(kinematics) if kinematics is not None else ()
        self.angle_channels = angle_channels
        self.spatial_map_boundary = spatial_map_boundary
        self.tolerance = tolerance

    def canonicalize(self, actions: Tensor, frame: ActionFrame) -> Tensor:
        """Transform original-frame actions/labels into canonical coordinates."""
        if not isinstance(frame, ActionFrame):
            raise TypeError("canonicalize requires an explicit ActionFrame")
        return self.decanonicalize(actions, frame.inverse())

    def decanonicalize(self, actions: Tensor, frame: ActionFrame) -> Tensor:
        """Recover original-frame actions without altering gripper commands."""
        _check_tensor(actions, "actions")
        if self.representation == "spatial_maps":
            with torch.autocast(device_type=actions.device.type, enabled=False):
                return self._spatial_maps(actions, frame)
        if actions.ndim < 1 or actions.shape[-1] != self.action_dim:
            raise ValueError(f"{self.representation} requires shape (..., {self.action_dim})")
        rotation, translation = _aligned_frame(frame, actions, tuple(actions.shape[:-1]))
        # Geometry is computed outside mixed-precision autocast, including during training.
        with torch.autocast(device_type=actions.device.type, enabled=False):
            if self.representation == "bimanual_joint":
                left = self._joint_map(actions[..., :6], rotation, translation, self.kinematics[0])
                right = self._joint_map(actions[..., 7:13], rotation, translation, self.kinematics[1])
                return torch.cat((left, actions[..., 6:7], right, actions[..., 13:14]), -1)
            if self.representation == "pixel_pick_place":
                self._check_planar(rotation, translation)
                pick = self._pixel_point(actions[..., :2], rotation, translation)
                place = self._pixel_point(actions[..., 3:5], rotation, translation)
                angle = actions[..., 2] + torch.atan2(rotation[..., 1, 0], rotation[..., 0, 0])
                angle = torch.atan2(torch.sin(angle), torch.cos(angle))[..., None]
                return torch.cat((pick, angle, place, actions[..., 5:6]), -1)
            position = (rotation @ actions[..., :3, None]).squeeze(-1)
            if self.representation == "delta_rotvec":
                orientation = (rotation @ actions[..., 3:6, None]).squeeze(-1)
            else:
                position = position + translation
                decode = rotvec_to_matrix if self.representation == "absolute_rotvec" else euler_xyz_to_matrix
                encode = matrix_to_rotvec if self.representation == "absolute_rotvec" else matrix_to_euler_xyz
                orientation = encode(rotation @ decode(actions[..., 3:6]))
            result = torch.cat((position, orientation, actions[..., 6:7]), -1)
            _check_tensor(result, "recovered actions")
            return result

    def forward(self, actions: Tensor, frame: ActionFrame) -> Tensor:
        return self.decanonicalize(actions, frame)

    def _check_planar(self, rotation: Tensor, translation: Tensor) -> None:
        axis = rotation.new_tensor([0, 0, 1])
        if not torch.allclose(rotation[..., :, 2], axis.expand_as(rotation[..., :, 2]),
                              atol=self.tolerance, rtol=0) or bool((translation[..., 2].abs() > self.tolerance).any()):
            raise ValueError("Pixel outputs require a planar rotation and zero z translation")

    @staticmethod
    def _pixel_point(point: Tensor, rotation: Tensor, translation: Tensor) -> Tensor:
        return (rotation[..., :2, :2] @ point.unsqueeze(-1)).squeeze(-1) + translation[..., :2]

    def _spatial_maps(self, maps: Tensor, frame: ActionFrame) -> Tensor:
        if maps.ndim != 4 or maps.shape[-1] != maps.shape[-2]:
            raise ValueError("Spatial maps must have shape [B, K, S, S]; pad before canonicalization")
        rotation, translation = _aligned_frame(frame, maps, (maps.shape[0],))
        self._check_planar(rotation, translation)
        rotation = rotation.expand(maps.shape[0], 3, 3)
        translation = translation.expand(maps.shape[0], 3)
        side = maps.shape[-1]
        preserve = self.spatial_map_boundary == "preserve"
        original_side = side - 1 if preserve else side
        if original_side < 1:
            raise ValueError("Preserve mode requires an original canvas plus one bottom/right guard pixel")
        center = maps.new_tensor([original_side / 2, original_side / 2, 0])
        expected_translation = center - (rotation @ center[:, None]).squeeze(-1)
        if not torch.allclose(translation, expected_translation, atol=self.tolerance, rtol=0):
            raise ValueError(
                "Spatial maps require the SQ/VLA center (W/2, H/2); "
                "W and H exclude the one-pixel guard canvas in preserve mode"
            )
        angle = torch.atan2(rotation[:, 1, 0], rotation[:, 0, 0])
        quarter_turns = angle / (torch.pi / 2)
        if not torch.allclose(quarter_turns, quarter_turns.round(), atol=self.tolerance, rtol=0):
            raise ValueError("Spatial-map recovery supports C4 quarter turns only")
        if self.angle_channels == "absolute" and maps.shape[1] % 4:
            raise ValueError("Absolute orientation bins must be divisible by four")
        if preserve:
            # On the expanded S+1 canvas, rot90's center is exactly S/2.
            # Keep the guard pixels until after inverse mapping, including masks.
            output = torch.stack([
                torch.rot90(item, k=-int(turns.item()), dims=(-2, -1))
                for item, turns in zip(maps, quarter_turns.round().to(torch.int64))
            ])
        else:
            rows, columns = torch.meshgrid(
                torch.arange(side, dtype=maps.dtype, device=maps.device),
                torch.arange(side, dtype=maps.dtype, device=maps.device),
                indexing="ij",
            )
            output_pixels = torch.stack((columns, rows), dim=-1)
            # Sample the source with the inverse of the canonical-to-original map.
            source_pixels = (
                rotation[:, None, None, :2, :2].transpose(-1, -2)
                @ (output_pixels - translation[:, None, None, :2]).unsqueeze(-1)
            ).squeeze(-1)
            grid = 2 * source_pixels / (side - 1) - 1 if side > 1 else torch.zeros_like(source_pixels)
            # Symmetric edge padding followed by a centered crop has the same
            # effective center and border extension on the original square canvas.
            # rot90 on the unexpanded canvas would use a different center.
            output = F.grid_sample(maps, grid, mode="bilinear", padding_mode="border", align_corners=True)
        if self.angle_channels == "absolute":
            shifts = quarter_turns.round().to(torch.int64) * (maps.shape[1] // 4)
            channels = (torch.arange(maps.shape[1], device=maps.device)[None, :] - shifts[:, None]) % maps.shape[1]
            output = output.gather(1, channels[:, :, None, None].expand_as(output))
        return output

    def _joint_map(self, joints: Tensor, rotation: Tensor, translation: Tensor,
                   adapter: JointKinematics) -> Tensor:
        def check_joints(value: Tensor, name: str) -> None:
            _check_tensor(value, name)
            if value.shape != joints.shape or value.device != joints.device or value.dtype != joints.dtype:
                raise ValueError(f"{name} must preserve joint shape, dtype, and device")
            valid = adapter.in_domain(value)
            if not isinstance(valid, Tensor) or valid.dtype != torch.bool or valid.shape != value.shape[:-1]:
                raise ValueError("in_domain must return a boolean mask matching the joint batch shape")
            if not bool(valid.all()):
                raise ValueError(f"{name} lies outside the configured reachable IK branch; clipping is not allowed")

        def fk(value: Tensor) -> Tensor:
            pose = adapter.forward(value)
            _check_tensor(pose, "FK poses")
            if pose.shape != value.shape[:-1] + (4, 4) or pose.device != value.device or pose.dtype != value.dtype:
                raise ValueError("FK must return (..., 4, 4) poses with the joint tensor's dtype/device")
            ActionFrame.from_matrix(pose)
            return pose

        check_joints(joints, "input joints")
        pose = fk(joints)
        branch_joints = adapter.inverse(pose)
        check_joints(branch_joints, "IK(FK(input))")
        if not torch.allclose(branch_joints, joints, atol=self.tolerance, rtol=0):
            raise ValueError("IK does not invert FK on the selected input branch")
        target_rotation = rotation @ pose[..., :3, :3]
        target_position = (rotation @ pose[..., :3, 3:4]).squeeze(-1) + translation
        upper = torch.cat((target_rotation, target_position[..., None]), -1)
        target = torch.cat((upper, pose[..., 3:4, :]), -2)
        recovered = adapter.inverse(target)
        check_joints(recovered, "recovered joints")
        if not torch.allclose(fk(recovered), target, atol=self.tolerance, rtol=0):
            raise ValueError("IK solution does not reproduce the transformed pose within tolerance")
        return recovered

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        if any(key.startswith(prefix + "shared_mlp.") for key in state_dict):
            error_msgs.append(prefix + "legacy learned MLP weights cannot be loaded as an analytic action map")
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)


class InverseActionCanonicalizationNetwork(ActionCanonicalizer):
    """Compatibility import name for the former learned adapter.

    representation and a workspace ActionFrame are now required. hidden_dim is
    accepted only for constructor compatibility; this module has no MLP weights.
    """

    def __init__(self, action_dim: int = 7, hidden_dim: int = 256, action_chunks: int = 8, *,
                 representation: ActionRepresentation | None = None,
                 kinematics: Sequence[JointKinematics] | None = None):
        if representation is None:
            raise ValueError(
                "Specify representation explicitly; seven action values do not identify a coordinate convention"
            )
        super().__init__(representation, kinematics=kinematics)
        if action_dim != self.action_dim:
            raise ValueError("action_dim does not match representation")
        if not isinstance(action_chunks, int) or isinstance(action_chunks, bool) or action_chunks < 1:
            raise ValueError("action_chunks must be a positive integer")
        self.action_chunks = action_chunks

    def forward(self, predicted_actions: Tensor, canonicalization_info_1=None, canonicalization_info_2=None, *,
                frame: ActionFrame | None = None) -> Tensor:
        if isinstance(canonicalization_info_1, ActionFrame) and canonicalization_info_2 is None and frame is None:
            frame = canonicalization_info_1
        elif canonicalization_info_1 is not None or canonicalization_info_2 is not None:
            raise ValueError(
                "Camera-angle dictionaries cannot define exact action recovery; pass frame=ActionFrame(...)"
            )
        if frame is None:
            raise ValueError("An explicit canonical-to-original ActionFrame is required")
        if predicted_actions.ndim != 3 or predicted_actions.shape[1] != self.action_chunks:
            raise ValueError("Expected [batch, configured action_chunks, action_dim]")
        return self.decanonicalize(predicted_actions, frame)
