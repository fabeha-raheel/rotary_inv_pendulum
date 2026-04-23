# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from .rotary_inv_pendulum_env_cfg import RotaryInvPendulumEnvCfg


class RotaryInvPendulumEnv(DirectRLEnv):
    cfg: RotaryInvPendulumEnvCfg

    def __init__(self, cfg: RotaryInvPendulumEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # self.dof_idx, _ = self.robot.find_joints(self.cfg.dof_names)
        self._joint1_idx, _ = self.robot.find_joints("Joint1")
        self._joint2_idx, _ = self.robot.find_joints("Joint2")

        # Pre-allocate action buffer
        self.actions = torch.zeros(self.num_envs, self.cfg.action_space,
                                    device=self.device)

    @staticmethod
    def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
        """Wrap angle to [-pi, pi] using atan2(sin, cos)."""
        return torch.atan2(torch.sin(angle), torch.cos(angle))

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot 

        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=1000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:

        # scale network output ±1 → ±48 Nm and apply to joints.
        self.actions = actions.clone().clamp(-1.0, 1.0) # clip policy output to [-1, 1]
        tau = torch.zeros(self.num_envs, self.robot.num_joints, device=self.device) # create tensor for torques
        # tau[:, self._joint1_idx] = torques[:, 0:1]   # rotating arm
        # tau[:, self._joint2_idx] = torques[:, 1:2]   # pendulum

        # Apply torque to joint1
        tau[:, self._joint1_idx] = self.actions[:, 0:1] * self.cfg.action_scale

        self.robot.set_joint_effort_target(tau)

    def _apply_action(self) -> None:
        # self.robot.set_joint_effort_target(self.actions * self.cfg.action_scale, joint_ids=self._cart_dof_idx)

        self.robot.write_data_to_sim()

    def _get_observations(self) -> dict:

        j_pos = self.robot.data.joint_pos
        j_vel = self.robot.data.joint_vel

        j1 = j_pos[:, self._joint1_idx[0]]  # arm angle / position
        j2 = j_pos[:, self._joint2_idx[0]]  # pendulum angle / position
        w1 = j_vel[:, self._joint1_idx[0]]  # arm angular velocity
        w2 = j_vel[:, self._joint2_idx[0]]  # pendulum angular velocity
        
        # Compute error of arm from target position (wraps between -pi and pi)
        # Using atan2(sin(delta), cos(delta))
        j1_error = self._wrap_to_pi(j1 - self.cfg.target_joint1)

        obs = torch.stack([
            torch.sin(j1), torch.cos(j1),
            torch.sin(j2), torch.cos(j2),
            w1, w2,
            j1_error,   # tells direction of error
        ], dim=-1)

        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:

        j_pos = self.robot.data.joint_pos
        j_vel = self.robot.data.joint_vel

        j1 = j_pos[:, self._joint1_idx[0]]  # arm angle / position
        j2 = j_pos[:, self._joint2_idx[0]]  # pendulum angle / position
        w1 = j_vel[:, self._joint1_idx[0]]  # arm angular velocity
        w2 = j_vel[:, self._joint2_idx[0]]  # pendulum angular velocity

        # Calculate wrapped error for the reward calculation
        j1_error = self._wrap_to_pi(j1 - self.cfg.target_joint1)
        j2_error = self._wrap_to_pi(j2 - self.cfg.target_joint2)

        # Torque control inputs
        u = self.actions[:, 0] * self.cfg.action_scale

        # 1. Alive bonus
        r_alive = self.cfg.rew_scale_alive

        # 2. Joint2 angle penalty
        r_j2_angle = self.cfg.rew_scale_j2_angle * (1.0 - torch.cos(j2_error))

        # 3. Pendulum angular velocity penalty
        r_j2_vel = self.cfg.rew_scale_j2_vel * w2.pow(2)

        # 4. Arm angular velocity penalty (mild, so arm can still move to balance)
        r_j1_vel = self.cfg.rew_scale_j1_vel * w1.pow(2)

        # 5. Torque regularization penalty - avoid large torque applications
        r_action = self.cfg.rew_scale_j1_action * u.pow(2)

        # 6. Mild arm centering penalty
        #    Keep this small; do not overconstrain Joint1
        r_j1_pos = self.cfg.rew_scale_j1_pos * j1_error.pow(2)

        # 7. Constraint Violations
        r_violated = self.cfg.rew_scale_violation * ((j1_error.abs() > self.cfg.ji_max) | 
                                                      (j2_error.abs() > self.cfg.j2_max)
                                                      ).float()

        total_reward = r_alive + r_j2_angle + r_j2_vel + r_j1_vel + r_action + r_j1_pos + r_violated

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # self.joint_pos = self.robot.data.joint_pos
        # self.joint_vel = self.robot.data.joint_vel

        # time_out = self.episode_length_buf >= self.max_episode_length - 1
        # out_of_bounds = torch.any(torch.abs(self.joint_pos[:, self._cart_dof_idx]) > self.cfg.max_cart_pos, dim=1)
        # out_of_bounds = out_of_bounds | torch.any(torch.abs(self.joint_pos[:, self._pole_dof_idx]) > math.pi / 2, dim=1)

        # termination conditions copied from pend_balc_env.py
        j_pos = self.robot.data.joint_pos

        j2 = j_pos[:, self._joint2_idx[0]]

        # Joint1: NEVER terminates — free rotating arm
        # Joint2: terminate if pendulum falls beyond ±90°
        j2_error = self._wrap_to_pi((j2 - self.cfg.target_joint2).abs())
        fell = j2_error > self.cfg.max_angle_j2

        time_out = self.episode_length_buf >= self.max_episode_length

        return fell, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        # if env_ids is None:
        #     env_ids = self.robot._ALL_INDICES
        # super()._reset_idx(env_ids)

        # joint_pos = self.robot.data.default_joint_pos[env_ids]
        # joint_pos[:, self._pole_dof_idx] += sample_uniform(
        #     self.cfg.initial_pole_angle_range[0] * math.pi,
        #     self.cfg.initial_pole_angle_range[1] * math.pi,
        #     joint_pos[:, self._pole_dof_idx].shape,
        #     joint_pos.device,
        # )
        # joint_vel = self.robot.data.default_joint_vel[env_ids]

        # default_root_state = self.robot.data.default_root_state[env_ids]
        # default_root_state[:, :3] += self.scene.env_origins[env_ids]

        # self.joint_pos[env_ids] = joint_pos
        # self.joint_vel[env_ids] = joint_vel

        # self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        # self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        # self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # reset conditions copied from _pend_balc_env.py
        
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        if len(env_ids) == 0:
            return
        super()._reset_idx(env_ids)

        # Reset the disturbance trackers
        # self._disturbance_counter[env_ids] = 0.0
        # self._current_disturbance_torque[env_ids] = 0.0

        noise = 0.1
        n = len(env_ids)

        joint_pos = torch.zeros(n, self.robot.num_joints, device=self.device)
        joint_vel = torch.zeros(n, self.robot.num_joints, device=self.device)

        # Joint1: random start angle + random initial spin (helps discover Furuta behavior)
        joint_pos[:, self._joint1_idx[0]] = (
            (torch.rand(n, device=self.device) - 0.5) * 2.0 * self._reset_j1_pos_range
        )
        joint_vel[:, self._joint1_idx[0]] = (
            (torch.rand(n, device=self.device) - 0.5) * 2.0 * self._reset_j1_vel_range
        )

        # Joint2: near upright with small angle noise and small initial angular velocity
        joint_pos[:, self._joint2_idx[0]] = self.cfg.target_joint2 + (
            (torch.rand(n, device=self.device) - 0.5) * 2.0 * self._reset_j2_pos_noise
        )
        joint_vel[:, self._joint2_idx[0]] = (
            (torch.rand(n, device=self.device) - 0.5) * 2.0 * self._reset_j2_vel_range
        )

        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.robot.reset(env_ids)


# @torch.jit.script
# def compute_rewards(
#     rew_scale_alive: float,
#     rew_scale_terminated: float,
#     rew_scale_pole_pos: float,
#     rew_scale_cart_vel: float,
#     rew_scale_pole_vel: float,
#     pole_pos: torch.Tensor,
#     pole_vel: torch.Tensor,
#     cart_pos: torch.Tensor,
#     cart_vel: torch.Tensor,
#     reset_terminated: torch.Tensor,
# ):
#     rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
#     rew_termination = rew_scale_terminated * reset_terminated.float()
#     rew_pole_pos = rew_scale_pole_pos * torch.sum(torch.square(pole_pos).unsqueeze(dim=1), dim=-1)
#     rew_cart_vel = rew_scale_cart_vel * torch.sum(torch.abs(cart_vel).unsqueeze(dim=1), dim=-1)
#     rew_pole_vel = rew_scale_pole_vel * torch.sum(torch.abs(pole_vel).unsqueeze(dim=1), dim=-1)
#     total_reward = rew_alive + rew_termination + rew_pole_pos + rew_cart_vel + rew_pole_vel
#     return total_reward