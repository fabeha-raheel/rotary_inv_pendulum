# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

# from isaaclab_assets.robots.cartpole import CARTPOLE_CFG
from rotary_inv_pendulum.robots.pend_balc import PEND_BALC_CONFIG

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
    # robot config has been added to a separate file inside source/rotary_inv_pendulum/robots
    robot_cfg: ArticulationCfg = PEND_BALC_CONFIG.replace(prim_path="/World/envs/env_.*/Robot")

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.0, replicate_physics=True)

    # custom parameters/scales
    # - controllable joint
    # cart_dof_name = "slider_to_cart"
    # pole_dof_name = "cart_to_pole"
    dof_names = ["Joint1", "Joint2"] # names of controllable joints of rotary_inv_pendulum
    target_joint2 = 0.0
    # - action scale
    # action_scale = 100.0  # [N] # no action scales were provided in pend_balc
    # - reward scales
    rew_scale_alive     = 1.0
    rew_scale_j2_angle  = -5.0   # penalize Joint2 deviation from upright
    rew_scale_j1_vel    = -0.05  # reward Joint1 spinning (when J2 stable)
    rew_scale_j1_stop   = -0.3   # penalize Joint1 moving (when J2 unstable)
    rew_scale_j2_vel    = -0.05  # penalize fast Joint2 motion
    rew_scale_j2_action = -0.01  # penalize large τ2 torques

    # reward scales from cartpole example
    # rew_scale_terminated = -2.0 
    # rew_scale_pole_pos = -1.0
    # rew_scale_cart_vel = -0.01
    # rew_scale_pole_vel = -0.005

    # - reset states/conditions
    # initial_pole_angle_range = [-0.25, 0.25]  # pole angle sample range on reset [rad]
    # max_cart_pos = 3.0  # reset if cart exceeds this position [m]
    max_angle_j2 = math.pi