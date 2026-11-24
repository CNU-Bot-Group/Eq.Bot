# Eq.Bot: Model-Agnostic Canonicalization for Equivariant Robotic Manipulation

Eq.Bot estimates a shared geometric frame, aligns observations to that frame, evaluates the base policy once, and recovers its predictions in the original coordinates. The base policy keeps its architecture and native action representation. Exact equivariance requires consistent frame estimates and compatible observation and action transformations.

## Code Organization

| Module | Contents |
| --- | --- |
| `common/basecanonicalization.py` | Shared base classes for identity, discrete, and continuous canonicalization. |
| `common/action_canonicalization.py` | Rigid frames and representation-specific action maps. |
| `common/utils.py` | Shared tensor geometry, image warps, and annotation transforms. |
| `canonicalization/` | Discrete image and VLA wrappers and continuous planar canonicalizers. |
| `canonicalization_networks/` | Discrete DirNet, its continuous steerable counterpart, and conventional CNN and Wide ResNet-50 encoders for OptNet. |

Use `eqbot.common` for the action interface. The image modules `discrete_group_sq.py`, `discrete_group_vla.py`, and `continuous_group.py` expose the observation canonicalizers. For discrete rotations, `GroupEquivariantImageCanonicalization` estimates a frame from group scores; `OptimizedGroupEquivariantImageCanonicalization` scores rotated views against a learned reference. The default group in the paper is `C4`. Continuous planar estimation uses an angle predicted from a vector, rather than the discrete view-scoring procedure.

## Dependencies and Local Checks

Use Python 3.10 or later. The action maps require PyTorch. Image canonicalization also requires NumPy, Kornia, TorchVision, and OmegaConf. The discrete DirNet encoder requires `e2cnn`; the continuous steerable encoder requires `escnn`. These two backends are loaded only when their respective encoders are used. The Wide ResNet-50 encoder defaults to `weights=None` so it can be initialized offline. Supply a local checkpoint or explicitly select weights to reproduce an initialization that uses pretrained features.

Run the example below from the directory containing `eqbot/`. The example requires only PyTorch and runs on CPU.

## Action Representations and Recovery

`ActionFrame` maps canonical coordinates to original coordinates. Use `canonicalize` for training targets and `decanonicalize` for inference outputs.

```bash
python - <<'PY'
import torch
from eqbot.common import ActionCanonicalizer, ActionFrame

frame = ActionFrame.from_yaw(torch.tensor([torch.pi / 2]),
                             torch.tensor([0.4, 0.0, 0.0]))
actions = torch.tensor([[[0.5, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0]]])
mapper = ActionCanonicalizer("absolute_rotvec")
targets = mapper.canonicalize(actions, frame)
recovered = mapper.decanonicalize(targets, frame)
torch.testing.assert_close(recovered, actions)
print("Action recovery round trip passed.")
PY
```

| Policy and setting | Representation | Recovery |
| --- | --- | --- |
| CLIPORT on Ravens | `pixel_pick_place` or `spatial_maps` | Recover pixel locations, absolute headings, or spatial score maps. Relative place headings are unchanged. |
| OpenVLA-OFT on LIBERO | `delta_rotvec` | Rotate workspace-frame translation and rotation increments without adding a position offset. |
| OpenVLA-OFT on the physical robot | `absolute_rotvec` | Compose the absolute end-effector pose with the estimated frame. |
| `pi0.5` on VLABench | `absolute_euler_xyz` | Compose the pose and re-encode its orientation as extrinsic XYZ Euler angles. |
| TurboVLA on RoboTwin 2.0 | `bimanual_joint` | Transform each arm's pose through forward and inverse kinematics on a fixed reachable branch. |

Decode normalized policy outputs before recovery. Use float32 or float64 for geometry, with meters and radians for physical actions. Gripper commands are preserved. For square image outputs, `ActionFrame.from_image_rotation` converts the stored discrete image angle into the pixel coordinate convention; it does not provide a physical workspace transform. Pixel actions are expressed in column/row order. On square-padded inputs, account for the padding offsets when moving between image and canvas coordinates.

For multiview control, use a consistent workspace frame across views, proprioception, and action recovery. The dual-camera wrapper does not infer this frame from independently estimated image angles. For joint recovery, supply `forward`, `inverse`, and `in_domain` for each arm in the same workspace coordinates.

## Code Release

This package contains the shared Eq.Bot components described above: observation canonicalizers and action recovery operators for different policy outputs. The CPU example checks action recovery in isolation. Reproducing the reported experiments also requires the corresponding policy integrations, training and evaluation scripts, and data.

We are preparing these materials for public release:

- **Simulation benchmarks:** Eq.Bot policy integrations and training and evaluation scripts for Ravens, LIBERO, VLABench, and RoboTwin 2.0, with instructions for obtaining and preparing each public benchmark dataset.
- **Real-world experiments:** UR5e training, deployment, and evaluation code, together with the collected robot demonstration dataset.

The current package does not include datasets, trained checkpoints, camera calibrations, or robot-specific kinematics. See the paper and appendix for the experimental setup, evaluation protocols, and results.

