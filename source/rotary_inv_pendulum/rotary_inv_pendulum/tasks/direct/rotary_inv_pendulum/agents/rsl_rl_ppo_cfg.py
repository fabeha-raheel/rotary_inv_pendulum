# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # num_steps_per_env = 16
    num_steps_per_env       = 32        # rollout horizon per env per update
    # max_iterations = 150
    max_iterations          = 20000      # total PPO iterations (~25 min on RTX 4070 Ti Super)
    # save_interval = 50
    save_interval           = 20        # checkpoint every 50 iterations
    experiment_name = "rotary_inv_pendulum"
    empirical_normalization = True      # normalize observations online
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        # actor_obs_normalization=False,
        # critic_obs_normalization=False,
        # actor_hidden_dims=[32, 32],
        # critic_hidden_dims=[32, 32],
        actor_hidden_dims  = [128, 64, 32],   # deeper network for Furuta dynamics
        critic_hidden_dims = [128, 64, 32],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        # use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005, # small entropy bonus for exploration
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )