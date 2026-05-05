import os
import math
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
# from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

PEND_BALCR_CONFIG = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/fabeha/isaaclab_projects/rotary_inv_pendulum/source/rotary_inv_pendulum/rotary_inv_pendulum/robots/pend_balcr/urdf/pend_balcr.usd",
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
                # sleep_threshold=0.005,
                # stabilization_threshold=0.001,
                fix_root_link=True,    # pins base_link to world
            ),
            ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={"Joint1": 0.0, "Joint2": math.radians(-180.0)},
            joint_vel={"Joint1": 0.0, "Joint2": 0.0},
        ),
        actuators={
            "joint1_actuator": ImplicitActuatorCfg(
                joint_names_expr=["Joint1"],
                stiffness=0.0,
                damping=0.0,  # ← Strong mechanical damping to absorb oscillations
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
        },              # raw torque control
    )

# Check correct file path
# print(os.path.join(os.path.dirname(__file__), "pend_balc", "pend_balc.usd"))