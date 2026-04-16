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
        self._joint1_idx, _ = self._robot.find_joints("Joint1")
        self._joint2_idx, _ = self._robot.find_joints("Joint2")

        # self.joint_pos = self.robot.data.joint_pos
        # self.joint_vel = self.robot.data.joint_vel

        # Pre-allocate action buffer
        self._actions = torch.zeros(self.num_envs, self.cfg.action_space,
                                    device=self.device)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)

        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        
        # add articulation to scene
        # this line has been commented because it was not present in the pend_balc_env.py file
        # self.scene.articulations["robot"] = self.robot 
        
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:

        # scale network output ±1 → ±48 Nm and apply to joints.
        self.actions = actions.clone().clamp(-1.0, 1.0)
        torques = self._actions * 48.0

        tau = torch.zeros(self.num_envs, self._robot.num_joints, device=self.device)
        tau[:, self._joint1_idx] = torques[:, 0:1]   # rotating arm
        tau[:, self._joint2_idx] = torques[:, 1:2]   # pendulum

        self._robot.set_joint_effort_target(tau)

    def _apply_action(self) -> None:
        # self.robot.set_joint_effort_target(self.actions * self.cfg.action_scale, joint_ids=self._cart_dof_idx)

        self._robot.write_data_to_sim()

    def _get_observations(self) -> dict:
        # obs = torch.cat(
        #     (
        #         self.joint_pos[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
        #         self.joint_vel[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
        #         self.joint_pos[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
        #         self.joint_vel[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
        #     ),
        #     dim=-1,
        # )

        # observations copied from pend_balc_env.py
        j_pos = self._robot.data.joint_pos
        j_vel = self._robot.data.joint_vel

        j1 = j_pos[:, self._joint1_idx[0]]
        j2 = j_pos[:, self._joint2_idx[0]]
        w1 = j_vel[:, self._joint1_idx[0]]
        w2 = j_vel[:, self._joint2_idx[0]]

        # sin/cos avoids discontinuity at ±π boundary
        obs = torch.stack([
            torch.sin(j1), torch.cos(j1),   # Joint1 orientation
            torch.sin(j2), torch.cos(j2),   # Joint2 orientation (controlled)
            w1,                              # Joint1 angular velocity
            w2,                              # Joint2 angular velocity
        ], dim=-1)

        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # total_reward = compute_rewards(
        #     self.cfg.rew_scale_alive,
        #     self.cfg.rew_scale_terminated,
        #     self.cfg.rew_scale_pole_pos,
        #     self.cfg.rew_scale_cart_vel,
        #     self.cfg.rew_scale_pole_vel,
        #     self.joint_pos[:, self._pole_dof_idx[0]],
        #     self.joint_vel[:, self._pole_dof_idx[0]],
        #     self.joint_pos[:, self._cart_dof_idx[0]],
        #     self.joint_vel[:, self._cart_dof_idx[0]],
        #     self.reset_terminated,
        # )

        # rewards copied from pend_balc_env.py
        j_pos = self._robot.data.joint_pos
        j_vel = self._robot.data.joint_vel

        j2 = j_pos[:, self._joint2_idx[0]]
        w1 = j_vel[:, self._joint1_idx[0]]
        w2 = j_vel[:, self._joint2_idx[0]]

        j2_error = j2 - self.cfg.target_joint2

        # Stability factor: 1.0 at target, 0.0 at limit
        stability = 1.0 - (j2_error.abs() / self.cfg.max_angle_j2).clamp(0.0, 1.0)

        # 1. Alive bonus
        r_alive = self.cfg.rew_scale_alive

        # 2. Joint2 angle penalty
        r_j2_angle = self.cfg.rew_scale_j2_angle * (1.0 - torch.cos(j2_error))

        # 3. Direction-aware Joint1 reward
        #    Reward joint1 for spinning in the CORRECT direction to correct joint2
        #    If j2_error > 0 (falling right): want w1 < 0 (CCW) → reward -w1
        #    If j2_error < 0 (falling left):  want w1 > 0 (CW)  → reward +w1
        #    This equals: reward = -sign(j2_error) * w1
        #    When near zero error, reward |w1| so it keeps spinning freely
        # correction_spin = -torch.sign(j2_error) * w1        # directional component
        correction_spin = -j2_error * w1

        free_spin       = w1.abs() * (1.0 - j2_error.abs()  # free spinning near center
                          / self.cfg.max_angle_j2).clamp(0.0, 1.0)

        r_j1_vel = self.cfg.rew_scale_j1_vel * (
            correction_spin * j2_error.abs() / self.cfg.max_angle_j2  # direction matters when far
            + free_spin                                                  # free spin when near center
        )

        # 4. Joint1 stop penalty when joint2 near limit
        r_j1_stop = self.cfg.rew_scale_j1_stop * w1.abs() * (1.0 - stability)

        # 5. Joint2 velocity penalty
        r_j2_vel = self.cfg.rew_scale_j2_vel * w2.pow(2)

        # 6. Joint2 action penalty
        r_j2_act = self.cfg.rew_scale_j2_action * self._actions[:, 1].pow(2)

        total_reward = r_alive + r_j2_angle + r_j1_vel + r_j1_stop + r_j2_vel + r_j2_act
        
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # self.joint_pos = self.robot.data.joint_pos
        # self.joint_vel = self.robot.data.joint_vel

        # time_out = self.episode_length_buf >= self.max_episode_length - 1
        # out_of_bounds = torch.any(torch.abs(self.joint_pos[:, self._cart_dof_idx]) > self.cfg.max_cart_pos, dim=1)
        # out_of_bounds = out_of_bounds | torch.any(torch.abs(self.joint_pos[:, self._pole_dof_idx]) > math.pi / 2, dim=1)

        # termination conditions copied from pend_balc_env.py
        j_pos = self._robot.data.joint_pos

        j2 = j_pos[:, self._joint2_idx[0]]

        # Joint1: NEVER terminates — free rotating arm
        # Joint2: terminate if pendulum falls beyond ±90°
        j2_error = (j2 - self.cfg.target_joint2).abs()
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
        
        if env_ids is None or len(env_ids) == 0:
            return

        super()._reset_idx(env_ids)

        noise = 0.1
        n = len(env_ids)

        joint_pos = torch.zeros(n, self._robot.num_joints, device=self.device)
        joint_vel = torch.zeros(n, self._robot.num_joints, device=self.device)

        # Joint1: random start angle + random initial spin (helps discover Furuta behavior)
        #joint_pos[:, self._joint1_idx[0]] = (torch.rand(n, device=self.device) - 0.5) * 2 * math.pi
        #joint_vel[:, self._joint1_idx[0]] = (torch.rand(n, device=self.device) - 0.5) * 4.0  # ±2 rad/s
        
        joint_pos[:, self._joint1_idx[0]] = (torch.rand(n, device=self.device) - 0.5) * 2 * math.pi
        joint_vel[:, self._joint1_idx[0]] = (torch.rand(n, device=self.device) - 0.5) * 6.0  # ±3 rad/s both directions


        # Joint2: near upright ± small noise
        joint_pos[:, self._joint2_idx[0]] = self.cfg.target_joint2 + \
            (torch.rand(n, device=self.device) - 0.5) * 2 * noise
        joint_vel[:, self._joint2_idx[0]] = 0.0

        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self._robot.reset(env_ids)


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