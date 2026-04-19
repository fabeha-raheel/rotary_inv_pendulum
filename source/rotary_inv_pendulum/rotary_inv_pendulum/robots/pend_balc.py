import os
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
# from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

PEND_BALC_CONFIG = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/fabeha/isaaclab_projects/rotary_inv_pendulum/source/rotary_inv_pendulum/rotary_inv_pendulum/robots/pend_balc/urdf/pend_balc.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                max_linear_velocity=1000.0,
                max_angular_velocity=287.0,
                max_depenetration_velocity=100.0,
                enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.001,
                fix_root_link=True,    # pins base_link to world
            ),
            ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={"Joint1": 0.5, "Joint2": 1.5},
            joint_vel={"Joint1": 0.0, "Joint2": 0.0},
        ),
        actuators={
            "joint1_actuator": ImplicitActuatorCfg(
                joint_names_expr=["Joint1"],
                effort_limit_sim=48.0,
                stiffness=20.0,   # IMPORTANT: no position control
                damping=2.0,     # no resistance → free rotation
            ),

            "joint2_passive": ImplicitActuatorCfg(
                joint_names_expr=["Joint2"],
                effort_limit_sim=0.0,   # NO torque allowed → passive
                stiffness=0.0,
                damping=0.0,
    ),},              # raw torque control
    )

# Check correct file path
# print(os.path.join(os.path.dirname(__file__), "pend_balc", "pend_balc.usd"))