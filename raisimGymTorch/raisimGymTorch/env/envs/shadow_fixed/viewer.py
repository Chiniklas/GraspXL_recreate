#!/usr/bin/python

from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import shadow_demo as mano
from raisimGymTorch.env.RaisimGymVecEnvOther import RaisimGymVecEnvTest as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param
from raisimGymTorch.env.bin.shadow_demo import NormalSampler
from raisimGymTorch.helper.initial_pose_final import get_initial_pose_faive

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from random import choice

from raisimGymTorch.helper import rotations
import raisimGymTorch.algo.ppo.module as ppo_module
import raisimGymTorch.algo.ppo.ppo as PPO

exp_name = "shadow_floating"
weight_saved = 'full_19000_r.pt'


parser = argparse.ArgumentParser(description="Shadow hand viewer restricted to Mug objects.")
parser.add_argument('-c', '--cfg', type=str, default='cfg_reg.yaml', help='config file name in cfgs/')
parser.add_argument('-d', '--logdir', type=str, default=None, help='root dir for temp logs')
parser.add_argument('-e', '--exp_name', type=str, default=exp_name, help='experiment name for logging')
parser.add_argument('-w', '--weight', type=str, default=weight_saved, help='checkpoint to load (relative to data_all unless absolute)')
parser.add_argument('-sd', '--storedir', type=str, default='data_all', help='storage dir under raisimGymTorch/')
parser.add_argument('-seed', '--seed', type=int, default=1)
parser.add_argument('--cat', type=str, default='mixed_train', help='category folder under rsc/ to sample mugs from')
parser.add_argument('--object', type=str, default=None, help='specific Mug object folder (optional)')
parser.add_argument('--fixed-base', action='store_true', help='use fixed-base URDF variant for the object')

args = parser.parse_args()
weight_path = args.weight
cfg_grasp = args.cfg

print(f"Configuration file: \"{args.cfg}\"")
print(f"Experiment name: \"{args.exp_name}\"")

task_name = args.exp_name
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../../.."

if args.logdir is None:
    exp_path = home_path
else:
    exp_path = args.logdir

cfg = YAML().load(open(task_path + '/cfgs/' + args.cfg, 'r'))
cfg['environment']['visualize'] = True

if args.seed != 1:
    cfg['seed'] = args.seed

cat_name = args.cat
cfg['environment']['load_set'] = cat_name
directory_path = home_path + f"/rsc/{cat_name}/"
print(directory_path)

items = os.listdir(directory_path)
mug_folders = [item for item in items
               if os.path.isdir(os.path.join(directory_path, item)) and item.startswith('Mug')]

if len(mug_folders) == 0:
    raise RuntimeError(f"No Mug objects found under {directory_path}")

if args.object:
    if args.object not in mug_folders:
        raise FileNotFoundError(f"Requested mug '{args.object}' not found in {directory_path}")
    obj_list = [args.object]
else:
    obj_list = [choice(mug_folders)]

print(obj_list)

num_envs = len(obj_list)
cfg['environment']['num_envs'] = num_envs
activations = nn.LeakyReLU

print('num envs', num_envs)

env = VecEnv(obj_list, mano.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)),
             cfg['environment'], cat_name=cat_name)

print("initialization finished")

obj_path_list = []
for obj_item in obj_list:
    if args.fixed_base:
        obj_path_list.append(os.path.join(f"{obj_item}/{obj_item}_fixed_base.urdf"))
    else:
        obj_path_list.append(os.path.join(f"{obj_item}/{obj_item}.urdf"))
env.load_multi_articulated(obj_path_list)

ob_dim_r = 258
act_dim = 28
print('ob dim', ob_dim_r)
print('act dim', act_dim)

trail_steps = 50
grasp_steps = 100
lift_step = 50
n_steps_r = grasp_steps + trail_steps + lift_step

actor_r = ppo_module.Actor(
    ppo_module.MLP(cfg['architecture']['policy_net'], activations, ob_dim_r, act_dim),
    ppo_module.MultivariateGaussianDiagonalCovariance(act_dim, num_envs, 1.0, NormalSampler(act_dim)), device)

critic_r = ppo_module.Critic(ppo_module.MLP(cfg['architecture']['value_net'], activations, ob_dim_r, 1), device)

test_dir = True

saver = ConfigurationSaver(log_dir=exp_path + "/raisimGymTorch/" + args.storedir + "/" + task_name,
                           save_items=[], test_dir=test_dir)

if os.path.isabs(weight_path):
    checkpoint_to_load = weight_path
else:
    checkpoint_to_load = saver.data_dir.split('eval')[0] + weight_path

ppo_r = PPO.PPO(actor=actor_r,
                critic=critic_r,
                num_envs=num_envs,
                num_transitions_per_env=n_steps_r,
                num_learning_epochs=4,
                gamma=0.996,
                lam=0.95,
                num_mini_batches=4,
                device=device,
                log_dir=saver.data_dir,
                shuffle_batch=False)

load_param(checkpoint_to_load, env, actor_r, critic_r, ppo_r.optimizer, saver.data_dir, cfg_grasp)

for update in range(1):
    qpos_reset_r = np.zeros((num_envs, 28), dtype='float32')
    qpos_reset_l = np.zeros((num_envs, 28), dtype='float32')
    obj_pose_reset = np.zeros((num_envs, 8), dtype='float32')

    target_center = np.zeros_like(env.affordance_center)
    object_center = np.zeros_like(env.affordance_center)
    fake_non_aff_center = [0.346408, 0.346408, 0.346408]
    contain_non_aff = np.zeros((num_envs, 1), dtype='float32')
    for i in range(num_envs):
        lowest_point = 0.
        txt_file_path = os.path.join(directory_path, obj_list[i]) + "/lowest_point_new.txt"
        with open(txt_file_path, 'r') as txt_file:
            lowest_point = float(txt_file.read())

        obj_pose_reset[i, :] = [1., -0., 0.502, 1., -0., -0., 0., 0.]
        obj_pose_reset[i, 2] -= lowest_point

        qpos_reset_r[i, -4] = 1.7

        if np.linalg.norm(env.non_aff_mesh[i].centroid - fake_non_aff_center) < 0.01:
            non_aff_mesh = None
        else:
            non_aff_mesh = env.non_aff_mesh[i]
            contain_non_aff[i, 0] = 1.

        rot, pos, bias = get_initial_pose_faive(env.aff_mesh[i], non_aff_mesh, "shadow")
        obj_mat = rotations.quat2mat(obj_pose_reset[i, 3:7])
        wrist_pose_obj = rotations.axisangle2euler(rot.reshape(-1, 3)).reshape(1, -1)
        wrist_mat = rotations.euler2mat(wrist_pose_obj)
        wrist_in_world = np.matmul(obj_mat, wrist_mat)
        wrist_pose = rotations.mat2euler(wrist_in_world)
        qpos_reset_r[i, :3] = obj_pose_reset[i, :3] + np.matmul(obj_mat, pos[0, :])
        qpos_reset_r[i, 3:6] = wrist_pose[0, :]

        target_center[i, :] = bias[:]
        object_center[i, :] = env.affordance_center[i]

    env.reset_state(qpos_reset_r,
                    qpos_reset_l,
                    np.zeros((num_envs, 28), 'float32'),
                    np.zeros((num_envs, 28), 'float32'),
                    obj_pose_reset)

    env.set_goals(target_center,
                  object_center,
                  np.zeros((num_envs, 1), 'float32'),
                  np.zeros((num_envs, 1), 'float32'),
                  np.zeros((num_envs, 1), 'float32'),
                  np.zeros((num_envs, 1), 'float32'),
                  np.zeros((num_envs, 1), 'float32'),
                  np.zeros((num_envs, 1), 'float32'),
                  np.zeros((num_envs, 1), 'float32'),
                  np.zeros((num_envs, 1), 'float32'))

    env.turn_on_visualization()

    obs_new_r, _ = env.observe(contain_non_aff, partial_obs=False)
    for step in range(n_steps_r):
        obs_r = obs_new_r[:, :].astype('float32')
        if step == (grasp_steps + trail_steps):
            env.switch_root_guidance(True)

        action_r = actor_r.architecture.architecture(torch.from_numpy(obs_r).to(device))
        action_r = action_r.cpu().detach().numpy()
        action_l = np.zeros_like(action_r)

        frame_start = time.time()

        env.step(action_r.astype('float32'), action_l.astype('float32'))

        obs_new_r, _ = env.observe(contain_non_aff, partial_obs=False)

        frame_end = time.time()
        wait_time = cfg['environment']['control_dt'] - (frame_end - frame_start)
        if wait_time > 0.:
            time.sleep(wait_time)

    env.turn_off_visualization()
