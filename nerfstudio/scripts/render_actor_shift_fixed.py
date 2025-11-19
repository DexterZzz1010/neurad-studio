# Copyright 2024 the authors of NeuRAD and contributors.
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Fixed version of render.py with corrected actor_shift implementation.
This module extends render.py without modifying the original code.

Fixes:
- Properly handles actor_shift broadcasting for all time steps
- Updates initial_positions to ensure correct interpolation
- Disables actor_editing to avoid conflicts

Usage:
    python -m nerfstudio.scripts.render_actor_shift_fixed dataset \
        --load-config /path/to/config.yml \
        --actor-shift -5.0 0.0 0.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, Union

import torch
import tyro
from typing_extensions import Annotated

from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.scripts.render import (
    DatasetRender,
    RenderInterpolated,
)


def modify_actors_fixed(
    pipeline: Pipeline,
    actor_shift: Tuple[float, ...],
    actor_removal_time: Optional[float] = None,
    actor_stop_time: Optional[float] = None,
    actor_indices: Optional[List[int]] = None,
):
    """
    Fixed version of modify_actors that properly handles actor_shift for static actors.
    
    Changes from original:
    1. Proper broadcasting with unsqueeze(0) for all time steps
    2. Updates initial_positions to ensure correct interpolation
    3. Disables actor_editing to avoid conflicts with direct position modification
    
    Args:
        pipeline: The rendering pipeline
        actor_shift: Shift vector in world coordinates (x, y, z)
        actor_removal_time: Time at which to remove actors
        actor_stop_time: Time at which to stop actors
        actor_indices: Specific actor indices to modify (None = all)
    """
    actor_shift = torch.tensor(actor_shift, dtype=torch.float32, device=pipeline.model.device)
    with torch.no_grad():
        if actor_indices is not None:
            indices = torch.tensor(actor_indices, device=pipeline.model.device, dtype=torch.int)
        else:
            indices = torch.arange(
                pipeline.model.dynamic_actors.actor_positions.shape[1], device=pipeline.model.device
            )

        # 确保平移应用到所有时间步（形状: [n_times, n_actors, 3]）
        # 使用 unsqueeze(0) 确保正确广播到所有时间步
        pipeline.model.dynamic_actors.actor_positions[:, indices, :] += actor_shift.unsqueeze(0)

        # 同时更新 initial_positions（用于插值参考）
        # 这确保插值计算时使用的是平移后的位置
        pipeline.model.dynamic_actors.initial_positions[:, indices, :] += actor_shift.unsqueeze(0)

        # 禁用 actor_editing 以避免重复应用编辑
        # 当直接修改 actor_positions 时，不需要通过 actor_editing 字典再次应用
        pipeline.model.dynamic_actors.actor_editing["lateral"] = 0.0
        pipeline.model.dynamic_actors.actor_editing["longitudinal"] = 0.0
        pipeline.model.dynamic_actors.actor_editing["height"] = 0.0
        pipeline.model.dynamic_actors.actor_editing["rotation"] = 0.0

        if actor_removal_time is not None:
            no_actor_mask = pipeline.model.dynamic_actors.unique_timestamps > actor_removal_time
            pipeline.model.dynamic_actors.actor_present_at_time[no_actor_mask, indices] = False
        if actor_stop_time is not None:
            actor_stop_idx = torch.searchsorted(
                pipeline.model.dynamic_actors.unique_timestamps, actor_stop_time
            )
            freeze_position = pipeline.model.dynamic_actors.actor_positions[
                actor_stop_idx, indices
            ].unsqueeze(0)
            freeze_rotation = pipeline.model.dynamic_actors.actor_rotations_6d[
                actor_stop_idx, indices
            ].unsqueeze(0)
            pipeline.model.dynamic_actors.actor_positions[actor_stop_idx:, indices] = freeze_position
            pipeline.model.dynamic_actors.actor_rotations_6d[
                actor_stop_idx:, indices
            ] = freeze_rotation


@dataclass
class DatasetRenderFixed(DatasetRender):
    """
    Fixed version of DatasetRender that uses the corrected modify_actors_fixed function.
    
    This class inherits from DatasetRender and only overrides the modify_actors call
    to use the fixed implementation that properly handles static actors.
    """

    def main(self):
        """Main function with fixed actor modification."""
        # Import the original main implementation and patch modify_actors
        from nerfstudio.scripts.render import modify_actors
        import nerfstudio.scripts.render as render_module

        # Temporarily replace modify_actors with the fixed version
        original_modify_actors = render_module.modify_actors
        render_module.modify_actors = modify_actors_fixed

        try:
            # Call parent's main method (which will use our patched modify_actors)
            super().main()
        finally:
            # Restore original function
            render_module.modify_actors = original_modify_actors


@dataclass
class RenderInterpolatedFixed(RenderInterpolated):
    """
    Fixed version of RenderInterpolated that uses the corrected modify_actors_fixed function.
    """

    def main(self) -> None:
        """Main function with fixed actor modification."""
        # Import the original main implementation and patch modify_actors
        from nerfstudio.scripts.render import modify_actors
        import nerfstudio.scripts.render as render_module

        # Temporarily replace modify_actors with the fixed version
        original_modify_actors = render_module.modify_actors
        render_module.modify_actors = modify_actors_fixed

        try:
            # Call parent's main method (which will use our patched modify_actors)
            super().main()
        finally:
            # Restore original function
            render_module.modify_actors = original_modify_actors


Commands = tyro.conf.FlagConversionOff[
    Union[
        Annotated[DatasetRenderFixed, tyro.conf.subcommand(name="dataset")],
        Annotated[RenderInterpolatedFixed, tyro.conf.subcommand(name="interpolate")],
    ]
]


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(Commands).main()


if __name__ == "__main__":
    entrypoint()
