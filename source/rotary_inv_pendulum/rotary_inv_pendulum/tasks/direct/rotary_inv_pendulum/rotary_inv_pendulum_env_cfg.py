# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


@configclass
class RotaryInvPendulumEnvCfg(DirectRLEnvCfg):
    # env
    
    # decimation = 2 # default - from cartpole template
    decimation = 8 # from pend_balc_env.py

    # episode_length_s = 5.0 # default - from cartpole template
    episode_length_s = 20.0  # from pend_balc_env.py

    # - spaces definition - from pend_balc_env.py
    action_space = 2        # [τ1, τ2]
    observation_space = 6   # [sin J1, cos J1, sin J2, cos J2, ω1, ω2]
    state_space = 0

    # simulation
    # sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation) # default - from template
    sim: SimulationCfg = SimulationCfg(dt=1 / 800, render_interval=decimation) # copied from pend_balc_env.py

    # robot(s)
    robot_cfg: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # custom parameters/scales
    # - controllable joint
    cart_dof_name = "slider_to_cart"
    pole_dof_name = "cart_to_pole"
    # - action scale
    action_scale = 100.0  # [N]
    # - reward scales
    rew_scale_alive = 1.0
    rew_scale_terminated = -2.0
    rew_scale_pole_pos = -1.0
    rew_scale_cart_vel = -0.01
    rew_scale_pole_vel = -0.005
    # - reset states/conditions
    initial_pole_angle_range = [-0.25, 0.25]  # pole angle sample range on reset [rad]
    max_cart_pos = 3.0  # reset if cart exceeds this position [m]