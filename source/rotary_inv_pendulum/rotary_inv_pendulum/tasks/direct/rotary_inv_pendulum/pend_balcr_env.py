# Copyright (c) 2025, Swarm Technologies.
# SPDX-License-Identifier: Apache-2.0

"""Furuta Pendulum (pend_balcr) RL Environment.

Robot structure (from URDF):
  base_link  -- fixed to world (6.207 kg base)
      └── Joint1  (revolute, X-axis, effort=48 Nm, vel=5.02 rad/s)
              Link1 (0.538 kg — rotating arm)
                  └── Joint2  (revolute, X-axis, effort=0 Nm - passive free joint)
                          Link2 (0.418 kg — pendulum rod)

Goal: Swing up and balance the pendulum (Joint2) at 0 rad (vertical up) from an 
      initial downward position (pi rad). The arm (Joint1) must use a "to-and-fro" 
      pumping motion to inject energy and is strictly bounded to a ±30° (pi/6 rad) 
      workspace limit. When balanced, Joint1 should stabilize at the center (0 rad).
Action: Joint1 torque only [τ1], scaled to ±48 Nm.
Obs (8-dim): [sin(J1), cos(J1), sin(J2), cos(J2), ω1, ω2, j1_error, j2_error]
"""



#from __future__ import annotations

import math
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg


'''
# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

@configclass
class PendBalcREnvCfg(DirectRLEnvCfg):
    """Configuration for the pend_balc Furuta Pendulum environment."""

    # -- Simulation --
    sim: SimulationCfg = SimulationCfg(dt=1 / 800, render_interval=4, device="cuda")

    # -- Episode --
    episode_length_s: float = 10.0
    decimation: int = 4

    # -- Spaces --
    action_space: int = 1          # only τ1 (Joint1 motor)
    observation_space: int = 7     # [sin J1, cos J1, sin J2, cos J2, ω1, ω2]
    state_space: int = 0

    # -- Robot asset --
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/hamza/workspace/pend_balcr/urdf/pend_balcr.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=10.0,
                enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=24,
                solver_velocity_iteration_count=24,
                sleep_threshold=0.005,
                stabilization_threshold=0.001,
                fix_root_link=True,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={"Joint1": 0.0, "Joint2": 3.1416},
            joint_vel={"Joint1": 0.0, "Joint2": 0.0},
        ),
        actuators={
            # Joint1 — AK80-64 motor (active, driven)
            "arm_joint": ImplicitActuatorCfg(
                joint_names_expr=["Joint1"],
                stiffness=0.0,
                damping=5.0,
                effort_limit_sim=48.0,
                velocity_limit_sim=5.0,
            ),
            # Joint2 — free pendulum (passive, no motor)
            "pendulum_joint": ImplicitActuatorCfg(
                joint_names_expr=["Joint2"],
                stiffness=0.0,
                damping=0.0,
                effort_limit=0.0,
                velocity_limit_sim=1000.0,
            ),
        },
    )

    # -- Scene --
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=2.0, replicate_physics=True
    )

    # -- Target --
    target_joint1: float = 0.0     # Target angle for the arm (e.g., center)
    target_joint2: float = 0.0     # Joint2 must stay vertical (0 rad)

    # -- Reward scales --
    rew_scale_alive:     float =  1.0   # bonus for keeping Joint2 upright
    rew_scale_j2_angle:  float = -5.0   # penalty for Joint2 deviating from 0
    rew_scale_j2_vel:    float = -0.05  # penalty for fast Joint2 motion
    rew_scale_j1_oppose: float = -2.0   # reward Joint1 moving OPPOSITE to Joint2 fall
    rew_scale_j1_same:   float =  3.0   # heavy penalty for moving SAME direction as fall
    rew_scale_j1_still:  float = -0.3   # penalty for Joint1 moving when Joint2 balanced
    rew_scale_j1_action: float = -0.01  # penalty for large τ1 torques
    rew_scale_j1_pos:    float = -0.5   # Penalty for the Joint1 being away from target

    # -- Disturbance settings --
    enable_disturbance: bool = True
    # Apply disturbance every 5 seconds (5s, 10s, 15s...)
    disturbance_interval_s: float = 5.0
    # How many physics steps should the push last? (40 steps @ 800Hz = 0.05s)
    disturbance_duration_steps: int = 40
    # Range of random force (Nm) applied to Joint2
    disturbance_range_nm: tuple[float, float] = (-2.0, 2.0)


    # -- Termination --
    max_angle_j2: float = math.pi   # ±90° from vertical


# ──────────────────────────────────────────────────────────────────────────────
# Environment
# ──────────────────────────────────────────────────────────────────────────────

class PendBalcREnv(DirectRLEnv):
    cfg: PendBalcREnvCfg

    def __init__(self, cfg: PendBalcREnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._joint1_idx, _ = self._robot.find_joints("Joint1")
        self._joint2_idx, _ = self._robot.find_joints("Joint2")
        self._actions = torch.zeros(self.num_envs, self.cfg.action_space,
                                    device=self.device)
        
        # Calculate how many policy steps make up 5.0 seconds
        # Policy step duration = sim.dt * decimation
        policy_step_dt = self.cfg.sim.dt * self.cfg.decimation
        self._disturbance_interval_policy_steps = int(self.cfg.disturbance_interval_s / policy_step_dt)
        
        # Tracks how many physics steps of "pushing" are left for each env
        self._disturbance_counter = torch.zeros(self.num_envs, device=self.device)
        
        # Holds the randomly sampled torque (-2 to 2) for the current kick
        self._current_disturbance_torque = torch.zeros(self.num_envs, device=self.device)

    # ── Scene ─────────────────────────────────────────────────────────────────

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self._robot
        sim_utils.spawn_ground_plane(prim_path="/World/ground",
                                     cfg=sim_utils.GroundPlaneCfg())
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])

    # ── Actions ───────────────────────────────────────────────────────────────

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)
        tau = torch.zeros(self.num_envs, self._robot.num_joints, device=self.device)

        # 1. Normal Motor Control (Joint1)
        tau[:, self._joint1_idx] = self._actions[:, 0:1] * 48.0

       # 2. Scheduled Recurring Disturbance (Joint2)
        if self.cfg.enable_disturbance:
            # Trigger every N policy steps (and ensure we don't trigger at step 0)
            just_reached = (self.episode_length_buf > 0) & \
                           (self.episode_length_buf % self._disturbance_interval_policy_steps == 0)

            # Start the countdown for triggered envs
            self._disturbance_counter[just_reached] = self.cfg.disturbance_duration_steps

            # Sample random torque between -2 and 2 for the triggered envs
            min_t, max_t = self.cfg.disturbance_range_nm
            random_torques = min_t + (max_t - min_t) * torch.rand(self.num_envs, device=self.device)
            
            # Save the rolled torque so it applies consistently for the duration of the push
            self._current_disturbance_torque[just_reached] = random_torques[just_reached]

            # Apply force to any env where the countdown is > 0
            active = self._disturbance_counter > 0
            if active.any():
                dist_tau = self._current_disturbance_torque * active.float()
                tau[:, self._joint2_idx[0]] += dist_tau

                # Count down one physics step
                self._disturbance_counter[active] -= 1

        self._robot.set_joint_effort_target(tau)

    def _apply_action(self):
        self._robot.write_data_to_sim()

    # ── Observations ──────────────────────────────────────────────────────────

    def _get_observations(self) -> dict:
        j_pos = self._robot.data.joint_pos
        j_vel = self._robot.data.joint_vel

        # 1. Define Joint 1 position and its error
        j1 = j_pos[:, self._joint1_idx[0]]
        diff = j1 - self.cfg.target_joint1
        j1_error = torch.atan2(torch.sin(diff), torch.cos(diff))

        # 2. Define Joint 2 position (wrapped -pi to pi)
        j2_raw = j_pos[:, self._joint2_idx[0]]
        j2_wrapped = torch.atan2(torch.sin(j2_raw), torch.cos(j2_raw))
        
        # 3. Define Velocities
        w1 = j_vel[:, self._joint1_idx[0]]
        w2 = j_vel[:, self._joint2_idx[0]]

        # 4. Stack into observations (Must match observation_space size)
        # Your config says observation_space: 7
        obs = torch.stack([
            torch.sin(j1),          # 1
            torch.cos(j1),          # 2
            torch.sin(j2_wrapped),  # 3
            torch.cos(j2_wrapped),  # 4
            w1,                     # 5
            w2,                     # 6
            j1_error,               # 7
        ], dim=-1)

        return {"policy": obs}

    # ── Rewards ───────────────────────────────────────────────────────────────

    def _get_rewards(self) -> torch.Tensor:
        j_pos = self._robot.data.joint_pos
        j_vel = self._robot.data.joint_vel

        j1 = j_pos[:, self._joint1_idx[0]] 
        j2 = j_pos[:, self._joint2_idx[0]]
        w1 = j_vel[:, self._joint1_idx[0]]
        w2 = j_vel[:, self._joint2_idx[0]]

        # Calculate wrapped error for the reward calculation
        diff = j1 - self.cfg.target_joint1
        j1_error = torch.atan2(torch.sin(diff), torch.cos(diff))

        j2_error = j2 - self.cfg.target_joint2

        # How far from upright: 0 = balanced, 1 = at limit
        normalized_error = (j2_error.abs() / self.cfg.max_angle_j2).clamp(0.0, 1.0)

        # How balanced: 1 = upright, 0 = at limit
        is_balanced = 1.0 - normalized_error

        # 1. Alive bonus
        r_alive = self.cfg.rew_scale_alive

        # 2. Joint2 angle penalty
        r_j2_angle = self.cfg.rew_scale_j2_angle * (1.0 - torch.cos(j2_error))

        # 3. Opposition reward (velocity-based)
        # Joint2 moving CW (+w2) → Joint1 must move CCW (-w1) → -w2*w1 is positive
        # Joint2 moving CCW (-w2) → Joint1 must move CW  (+w1) → -w2*w1 is positive
        # Joint2 still (w2≈0) → signal ≈ 0 → neutral
        opposing_signal = w2 * w1

        # Reward when opposing (positive signal), scale by how much J2 is falling
        r_j1_oppose = self.cfg.rew_scale_j1_oppose * \
            torch.clamp(opposing_signal, min=0.0) * normalized_error

        # Penalize when moving same direction (negative signal)
        r_j1_same = self.cfg.rew_scale_j1_same * \
            torch.clamp(opposing_signal, max=0.0) * normalized_error

        # 4. When Joint2 balanced, Joint1 should stay still
        r_j1_still = self.cfg.rew_scale_j1_still * w1.abs() * is_balanced

        # 5. Joint2 velocity penalty
        r_j2_vel = self.cfg.rew_scale_j2_vel * w2.pow(2)

        # 6. Joint1 action penalty (action_space=1, only index 0)
        r_j1_act = self.cfg.rew_scale_j1_action * self._actions[:, 0].pow(2)

        # 7. Joint1 Position Penalty
        # This forces the arm to find a balance point at exactly target_joint1
        r_j1_pos = self.cfg.rew_scale_j1_pos * j1_error.pow(2)

        return r_alive + r_j2_angle + r_j1_oppose + r_j1_same + r_j1_still + r_j2_vel + r_j1_act + r_j1_pos

    # ── Terminations ──────────────────────────────────────────────────────────

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        j_pos = self._robot.data.joint_pos
        
        # Ensure j1 and j2 are defined here too
        j1 = j_pos[:, self._joint1_idx[0]]
        j2 = j_pos[:, self._joint2_idx[0]]

        # Limit checks
        j1_limit_hit = j1.abs() > 1.047  # 60 degrees
        j2_error = (j2 - self.cfg.target_joint2).abs()
        fell = j2_error > math.pi        # 180 degrees

        timed_out = self.episode_length_buf >= self.max_episode_length

        return (fell | j1_limit_hit), timed_out

    # ── Reset ─────────────────────────────────────────────────────────────────

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == 0:
            return

        super()._reset_idx(env_ids)

        # Reset the disturbance trackers
        self._disturbance_counter[env_ids] = 0.0
        self._current_disturbance_torque[env_ids] = 0.0

        n = len(env_ids)

        # 1. Get the default joint positions from the robot configuration
        # This pulls the 'pos' values from the initial_state you defined in the CFG
        default_joint_pos = self._robot.data.default_joint_pos[env_ids]
        
        # 2. Initialize joint velocity as zeros
        joint_vel = torch.zeros(n, self._robot.num_joints, device=self.device)

        # 3. Write the exact initial state to the simulation
        # Using default_joint_pos ensures Joint1 and Joint2 go to your CFG positions
        self._robot.write_joint_state_to_sim(default_joint_pos, joint_vel, env_ids=env_ids)
        
        # 4. Reset the robot's internal buffers
        self._robot.reset(env_ids)

'''


# Copyright (c) 2025, Swarm Technologies.
# SPDX-License-Identifier: Apache-2.0

import math
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg

'''

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

@configclass
class PendBalcREnvCfg(DirectRLEnvCfg):
    """Configuration for the pend_balc Furuta Pendulum environment."""

    # -- Simulation --
    sim: SimulationCfg = SimulationCfg(dt=1 / 800, render_interval=4, device="cuda")

    # -- Episode --
    episode_length_s: float = 10.0
    decimation: int = 4

    # -- Spaces --
    action_space: int = 1          
    observation_space: int = 7     
    state_space: int = 0

    # -- Robot asset --
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/hamza/workspace/pend_balcr/urdf/pend_balcr.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=10.0,
                enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=24,
                solver_velocity_iteration_count=24,
                fix_root_link=True,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            # Joint2 starts vertically downward (pi rad)
            joint_pos={"Joint1": 0.0, "Joint2": 3.14159},
            joint_vel={"Joint1": 0.0, "Joint2": 0.0},
        ),
        actuators={
            # Joint1 — AK80-64 motor (48Nm, 48RPM limit)
            "arm_joint": ImplicitActuatorCfg(
                joint_names_expr=["Joint1"],
                stiffness=0.0,
                damping=5.0,
                effort_limit_sim=48.0,
                velocity_limit_sim=5.02, # 48 RPM converted to rad/s
            ),
            # Joint2 — free pendulum (passive)
            "pendulum_joint": ImplicitActuatorCfg(
                joint_names_expr=["Joint2"],
                stiffness=0.0,
                damping=0.0,
                effort_limit=0.0,
            ),
        },
    )

    # -- Scene --
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=2.0, replicate_physics=True
    )

    # -- Target --
    target_joint1: float = 0.0     
    target_joint2: float = 0.0     # Upright

    # -- Reward scales --
    rew_scale_swingup:   float = 10.0   # Bonus for high-velocity pumping at the bottom
    rew_scale_balance:   float = 5.0   # Reward for keeping Joint2 near 0 rad
    rew_scale_j1_oppose: float = 10.0    # Reward for Joint1 moving CCW when Joint2 is CW
    rew_scale_j1_pos:    float = -2.0   # Penalty for Joint1 being away from center
    rew_scale_j1_vel:    float = -0.01  # Small penalty for excessive oscillation when balanced
    # New Penalty: Penalty for Joint2 deviating > 60 degrees
    rew_scale_j2_limit:  float = -300.0

    # -- Termination --
    # No termination on Joint2 (allows continuous swing-up training)
    # Terminate only if Joint1 exceeds workspace limits (+/- 60 degrees)
    max_j1_angle: float = 1.047 

# ──────────────────────────────────────────────────────────────────────────────
# Environment
# ──────────────────────────────────────────────────────────────────────────────

class PendBalcREnv(DirectRLEnv):
    cfg: PendBalcREnvCfg

    def __init__(self, cfg: PendBalcREnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._joint1_idx, _ = self._robot.find_joints("Joint1")
        self._joint2_idx, _ = self._robot.find_joints("Joint2")
        self._actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self._robot
        sim_utils.spawn_ground_plane(prim_path="/World/ground", cfg=sim_utils.GroundPlaneCfg())
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)
        tau = torch.zeros(self.num_envs, self._robot.num_joints, device=self.device)
        # Apply scaled torque to Joint1
        tau[:, self._joint1_idx] = self._actions * 48.0
        self._robot.set_joint_effort_target(tau)

    def _apply_action(self):
        self._robot.write_data_to_sim()

    def _get_observations(self) -> dict:
        j_pos = self._robot.data.joint_pos
        j_vel = self._robot.data.joint_vel

        # Map J2: 0 is Up, CCW is positive (+), CW is negative (-)
        # Wrap to [-pi, pi]
        j2 = torch.atan2(torch.sin(j_pos[:, self._joint2_idx[0]]), 
                         torch.cos(j_pos[:, self._joint2_idx[0]]))
        
        j1 = j_pos[:, self._joint1_idx[0]]
        w1 = j_vel[:, self._joint1_idx[0]]
        w2 = j_vel[:, self._joint2_idx[0]]

        obs = torch.stack([
            torch.sin(j1), torch.cos(j1),
            torch.sin(j2), torch.cos(j2),
            w1, w2,
            (j1 - self.cfg.target_joint1) # Position error
        ], dim=-1)

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        j_pos = self._robot.data.joint_pos
        j_vel = self._robot.data.joint_vel

        # State Extraction
        j1 = j_pos[:, self._joint1_idx[0]] 
        w1 = j_vel[:, self._joint1_idx[0]]
        
        # J2: 0 is Up, pi/-pi is Down
        j2 = torch.atan2(torch.sin(j_pos[:, self._joint2_idx[0]]), 
                         torch.cos(j_pos[:, self._joint2_idx[0]]))
        w2 = j_vel[:, self._joint2_idx[0]]

        # ──────────────────────────────────────────────────────────────────────
        # 1. SWING-UP PHASE (The "To-and-Fro" Pumping)
        # ──────────────────────────────────────────────────────────────────────
        # If J2 is > 60 degrees (1.047 rad), we focus entirely on gaining energy.
        is_failing = j2.abs() > 1.047
        
        # We reward the SQUARE of w1 to encourage the motor to hit its 48 RPM limit.
        # This gives the "so much speed" you requested to help Joint2 rise.
        r_speed_pumping = is_failing.float() * w1.pow(2) * self.cfg.rew_scale_swingup

        # ──────────────────────────────────────────────────────────────────────
        # 2. BALANCING PHASE (The "Catch")
        # ──────────────────────────────────────────────────────────────────────
        # Only reward balancing and opposite motion if J2 is within the 60° cone.
        is_in_range = ~is_failing
        
        # Proportional reward for being close to 0 rad
        r_balance = is_in_range.float() * (1.0 - j2.abs() / 1.047) * self.cfg.rew_scale_balance

        # The "Opposite" rule: if J2 leans one way, w1 moves the other.
        # This is critical for catching the pendulum as it arrives from the swing-up.
        r_oppose = (torch.sign(w1) != torch.sign(j2)).float() * self.cfg.rew_scale_j1_oppose

        # ──────────────────────────────────────────────────────────────────────
        # 3. CONSTRAINTS & PENALTIES
        # ──────────────────────────────────────────────────────────────────────
        # Penalty for failing to keep it above 60 degrees
        r_j2_limit = is_failing.float() * self.cfg.rew_scale_j2_limit

        # Joint 1 positional penalty (keep arm centered)
        # Note: We scale this by is_in_range so it doesn't fight the swing-up pumping.
        r_j1_pos = is_in_range.float() * self.cfg.rew_scale_j1_pos * j1.pow(2)

        return r_speed_pumping + r_balance + r_oppose + r_j2_limit + r_j1_pos

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        j_pos = self._robot.data.joint_pos
        j1 = j_pos[:, self._joint1_idx[0]]

        # Terminate only if Joint1 exceeds the allowed workspace (+/- 60 degrees)
        out_of_bounds = j1.abs() > self.cfg.max_j1_angle
        
        timed_out = self.episode_length_buf >= self.max_episode_length

        return out_of_bounds, timed_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == 0:
            return
        super()._reset_idx(env_ids)

        # Reset to initial position defined in cfg (J1=0, J2=180 deg)
        n = len(env_ids)
        default_joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = torch.zeros(n, self._robot.num_joints, device=self.device)

        self._robot.write_joint_state_to_sim(default_joint_pos, joint_vel, env_ids=env_ids)
        self._robot.reset(env_ids)
'''


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

@configclass
class PendBalcREnvCfg(DirectRLEnvCfg):
    """Configuration for the pend_balcr Furuta Pendulum environment.
    
    COORDINATE SYSTEM:
    - Joint2: 0 rad = vertically upright, ±π rad = vertically downward
    - Sign: CCW (counterclockwise) = positive, CW (clockwise) = negative
    - Wrapping: [-π, π] maintains sign convention across full rotation
    
    CONTROL STRATEGY:
    1. SWING-UP PHASE (|J2| > 45°): Motor oscillates at ~48 RPM to pump energy
    2. BALANCE PHASE (|J2| ≤ 45°): Motor applies opposite restoring torque
    3. HOMING PHASE (balanced): Optional slow return to J1 target position
    """

    # -- Simulation --
    sim: SimulationCfg = SimulationCfg(dt=1 / 800, render_interval=10, device="cuda")

    # -- Episode --
    episode_length_s: float = 15.0
    decimation: int = 10  # Policy runs at 100 Hz (800 / 4)

    # -- Spaces --
    action_space: int = 1          # τ1 only
    observation_space: int = 8     # [sin J1, cos J1, sin J2, cos J2, ω1, ω2, J1_error, J2_error]
    state_space: int = 0

    # -- Robot asset --
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/hamza/workspace/pend_balcr/urdf/pend_balcr.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=10.0,
                enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=24,
                solver_velocity_iteration_count=32,
                fix_root_link=True,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            # Joint2 starts vertically downward (π rad, equivalent to 180°)
            joint_pos={"Joint1": 0.0, "Joint2": math.radians(0.0)},
            joint_vel={"Joint1": 0.0, "Joint2": 0.0},
        ),
        actuators={
            # Joint1 — AK80-64 motor (48 Nm, 48 RPM = 5.027 rad/s)
            # damping=5.0 provides mechanical stability and prevents oscillations
            "arm_joint": ImplicitActuatorCfg(
                joint_names_expr=["Joint1"],
                stiffness=0.0,
                damping=3.0,  # ← Strong mechanical damping to absorb oscillations
                effort_limit_sim=48.0,
                velocity_limit_sim=5.027,  # 48 RPM in rad/s
            ),
            # Joint2 — free pendulum (passive, no actuator, fully free)
            "pendulum_joint": ImplicitActuatorCfg(
                joint_names_expr=["Joint2"],
                stiffness=0.0,
                damping=0.0,  # Fully free, no resistance
                effort_limit=0.0,  # No motor torque
            ),
        },
    )

    # -- Scene --
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=2.0, replicate_physics=True
    )

    # -- Targets --
    target_joint1: float = 0.0     # Home position for arm
    target_joint2: float = 0.0     # Upright (balanced)

    # ──────────────────────────────────────────────────────────────────────────
    # MOTION STRATEGY THRESHOLDS (based on J2 angle)
    # ──────────────────────────────────────────────────────────────────────────
    # Critical angle ranges for switching control modes
    angle_90_deg: float = math.radians(60.0)   # 90° (π/2) — intermediate position
    angle_45_deg: float = math.radians(45.0)   # 45° — balance-ready threshold
    angle_10_deg: float = math.radians(10.0)   # 10° — fine balance zone
    
    # ──────────────────────────────────────────────────────────────────────────
    # REWARD STRUCTURE: Hierarchical control with J2-based priorities
    # ──────────────────────────────────────────────────────────────────────────
    
    # PRIORITY 1: BALANCE REWARD (Always active, highest weight)
    # Primary objective: Keep |J2| close to 0 (vertical upright)
    rew_scale_balance: float = 50.0
    
    # PRIORITY 1B: POSITION-BASED J2 SPEED PENALTY
    # Penalize J2 angular velocity to encourage smooth transitions
    rew_scale_j2_speed: float = -10.0 #-2.0
    
    # PRIORITY 2: J1-J2 OPPOSITION REWARD
    # Reward when J1 moves opposite to J2 falling direction
    rew_scale_j1_opposition: float = 25.0
    
    # PRIORITY 2B: J1 CRITICAL OPPOSITION (90°-180° range)
    # Strongly reward J1 moving opposite when J2 in critical swing
    rew_scale_critical_opposition: float = 30.0
    
    # PRIORITY 3: J1 TO-AND-FRO MOTION (Pumping)
    # Reward high J1 velocity for energy transfer during swing-up
    rew_scale_pumping_speed: float = 30.0  # Increased to encourage aggressive pumping
    
    # PRIORITY 3B: J1 VELOCITY PENALTY (Only in fine balance zone)
    # Strong penalty to suppress oscillations when J2 is nearly upright
    # ONLY active during fine balance, NOT during swing-up phases
    rew_scale_j1_velocity: float = -25.0
    
    # PRIORITY 3C: SWING-UP SPEED BOOST (Active in swingup and critical phases)
    # Bonus reward for achieving high velocities during swing-up phases
    # Encourages rapid motor acceleration to pump energy into J2
    rew_scale_swingup_speed: float = 20.0
    
    # PRIORITY 4: J1 POSITION PENALTY (Only when balanced)
    # Strong penalty to keep J1 at target position (zero)
    rew_scale_j1_position: float = -1.0 #-15.0
    
    # PRIORITY 5: EFFORT PENALTY
    # Penalty to discourage unnecessary motor commands, but scaled down during swing-up
    rew_scale_effort: float = -5.1  # Reduced to allow aggressive acceleration during swing-up
    
    # J2 VELOCITY DAMPING (Fine balance)
    # Strong penalty for J2 velocity near upright
    rew_scale_j2_damping: float = -5.0

    # -- Termination --
    max_j1_angle: float = math.radians(60.0)  # ±60° workspace limit for Joint1


# ──────────────────────────────────────────────────────────────────────────────
# Environment
# ──────────────────────────────────────────────────────────────────────────────

class PendBalcREnv(DirectRLEnv):
    """
    Furuta Pendulum Swing-Up and Balance environment.
    Updated to enforce strict directional opposition.
    """
    cfg: PendBalcREnvCfg

    def __init__(self, cfg: PendBalcREnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._joint1_idx, _ = self._robot.find_joints("Joint1")
        self._joint2_idx, _ = self._robot.find_joints("Joint2")
        self._actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._last_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)

    # ── Scene ─────────────────────────────────────────────────────────────────

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self._robot
        sim_utils.spawn_ground_plane(prim_path="/World/ground", cfg=sim_utils.GroundPlaneCfg())
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])

    # ── Actions ───────────────────────────────────────────────────────────────

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)
        tau = torch.zeros(self.num_envs, self._robot.num_joints, device=self.device)
        # Apply 48Nm Torque scale
        tau[:, self._joint1_idx] = self._actions * 48.0
        self._robot.set_joint_effort_target(tau)

    def _apply_action(self):
        self._robot.write_data_to_sim()

    # ── Observations ──────────────────────────────────────────────────────────

    def _get_observations(self) -> dict:
        j_pos = self._robot.data.joint_pos
        j_vel = self._robot.data.joint_vel

        j1 = j_pos[:, self._joint1_idx[0]]
        w1 = j_vel[:, self._joint1_idx[0]]
        
        j2 = torch.atan2(torch.sin(j_pos[:, self._joint2_idx[0]]),
                         torch.cos(j_pos[:, self._joint2_idx[0]]))
        w2 = j_vel[:, self._joint2_idx[0]]

        j1_error = torch.atan2(torch.sin(j1 - self.cfg.target_joint1),
                               torch.cos(j1 - self.cfg.target_joint1))
        
        j2_error = torch.atan2(torch.sin(j2 - self.cfg.target_joint2),
                               torch.cos(j2 - self.cfg.target_joint2))

        obs = torch.stack([
            torch.sin(j1),      # 0
            torch.cos(j1),      # 1
            torch.sin(j2),      # 2
            torch.cos(j2),      # 3
            w1 / 5.027,         # 4: Normalized Arm Vel
            w2 / 10.0,          # 5: Normalized Pendulum Vel
            j1_error,           # 6
            j2_error            # 7
        ], dim=-1)

        return {"policy": obs}

    # ── Rewards ───────────────────────────────────────────────────────────────
    
    def _get_rewards(self) -> torch.Tensor:
        j_pos = self._robot.data.joint_pos
        j_vel = self._robot.data.joint_vel

        # Extract States
        j1 = j_pos[:, self._joint1_idx[0]]
        w1 = j_vel[:, self._joint1_idx[0]]
        j2 = torch.atan2(torch.sin(j_pos[:, self._joint2_idx[0]]),
                         torch.cos(j_pos[:, self._joint2_idx[0]]))
        w2 = j_vel[:, self._joint2_idx[0]]
    
        # NEW: Action Smoothness Reward
        # Penalizes the change in torque between steps.
        # Higher value = smoother motion, but too high will make it "lazy".
        r_action_rate = torch.norm(self._actions - self._last_actions, dim=-1) * -0.05
    
        # Store actions for the next step
        self._last_actions = self._actions.clone()
        
        # ... extraction code ...
        
        # 1. Total Energy (Potential + Kinetic)
        # Reward the agent for increasing the "height" of the pendulum
        # Potential energy is max at top (cos(0)=1), min at bottom (cos(pi)=-1)
        r_height = torch.cos(j2) * 20.0 
        
        # 2. Broad Balance (Keep it simple)
        r_balance = torch.exp(-j2.pow(2) / 0.5) * 100.0
        
        # 3. Effort (Minimal)
        r_effort = self._actions.pow(2).sum(-1) * -0.1
        
        # 4. Limit Penalty (Soft)
        #j1_limit = (j1.abs() / self.cfg.max_j1_angle).pow(4) * -20.0

        # Normalize J1 position (0 to 1)
        j1_ratio = j1.abs() / self.cfg.max_j1_angle

        # A. The Spring (your existing code)
        r_wall_spring = j1_ratio.pow(4) * -20.0

        # B. The Brake (New)
        # If J1 is moving TOWARD the limit (signs match), penalize velocity
        moving_outward = (torch.sign(j1) == torch.sign(w1)).float()
        r_wall_brake = moving_outward * w1.abs() * j1_ratio.pow(2) * -15.0

        j1_limit = r_wall_spring + r_wall_brake

        return r_height + r_balance + r_effort + j1_limit + r_action_rate
    
    # ── Terminations ──────────────────────────────────────────────────────────

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Determine episode termination conditions.
        
        Updated: J1 exceeding bounds no longer terminates the episode.
        It relies on reward penalties to reverse direction.
        """
        # Create a tensor of Falses (no out-of-bounds resets)
        out_of_bounds = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        timed_out = self.episode_length_buf >= self.max_episode_length
        
        return out_of_bounds, timed_out

    # ── Reset ─────────────────────────────────────────────────────────────────

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == 0: 
            return
        super()._reset_idx(env_ids)
    
        n = len(env_ids)
        # Get default positions for the specific envs being reset
        default_joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros(n, self._robot.num_joints, device=self.device)

        # Add +/- 0.1 rad of noise to the starting pendulum angle (Joint2)
        # We use (n, 1) to match the shape of the joint slice [32, 1]
        random_offset = (torch.rand(n, 1, device=self.device) - 0.5) * 0.2
        
        # Apply the offset specifically to Joint2
        default_joint_pos[:, self._joint2_idx] += random_offset

        # Write back to simulation
        self._robot.write_joint_state_to_sim(default_joint_pos, joint_vel, env_ids=env_ids)
        self._robot.reset(env_ids)