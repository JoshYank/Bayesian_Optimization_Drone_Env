'''
Here we consider a controller trained  for the mountain-car environment in
OpenAI Gym. The controller was taken from the baselines. The controller is
based on ddpg.
'''


import gym
import numpy as np
from baselines.ddpg.ddpg import DDPG
from baselines.ddpg.noise import *
from baselines.ddpg.models import Actor, Critic
from baselines.ddpg.memory import Memory
from baselines.common import set_global_seeds
import baselines.common.tf_util as U
from mpi4py import MPI
from collections import deque


def train_return(env, param_noise, actor, critic, memory,nb_epochs=250, nb_epoch_cycles=20, reward_scale=1.,
                 render=False,normalize_returns=False, normalize_observations=True, critic_l2_reg=1e-2, actor_lr=1e-4,
                 critic_lr=1e-3,
          action_noise=None, popart=False, gamma=0.99, clip_norm=None,nb_train_steps=50, nb_rollout_steps=2048,
          batch_size=64,tau=0.01, param_noise_adaption_interval=50):
    rank = MPI.COMM_WORLD.Get_rank()

    assert (np.abs(env.action_space.low) == env.action_space.high).all()  # we assume symmetric actions.
    max_action = env.action_space.high
    agent = DDPG(actor, critic, memory, env.observation_space.shape, env.action_space.shape,
                 gamma=gamma, tau=tau, normalize_returns=normalize_returns,
                 normalize_observations=normalize_observations,
                 batch_size=batch_size, action_noise=action_noise, param_noise=param_noise, critic_l2_reg=critic_l2_reg,
                 actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart, clip_norm=clip_norm,
                 reward_scale=reward_scale)

    # Set up logging stuff only for a single worker.



    episode_rewards_history = deque(maxlen=100)
    #with U.single_threaded_session() as sess:
    # Prepare everything.
    agent.initialize(sess)
    sess.graph.finalize()

    agent.reset()
    obs = env.reset()
    episode_reward = 0.
    episode_step = 0
    episodes = 0
    t = 0

    epoch_episode_rewards = []
    epoch_episode_steps = []
    epoch_actions = []
    epoch_qs = []
    epoch_episodes = 0
    for epoch in range(nb_epochs):
        print('epoch number:', epoch)
        for cycle in range(nb_epoch_cycles):
            # Perform rollouts.
            for t_rollout in range(nb_rollout_steps):
                # Predict next action.
                action, q = agent.pi(obs, apply_noise=True, compute_Q=True)
                assert action.shape == env.action_space.shape

                # Execute next action.
                if rank == 0 and render:
                    env.render()
                assert max_action.shape == action.shape
                new_obs, r, done, info = env.step(
                    max_action * action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                t += 1
                if rank == 0 and render:
                    env.render()
                episode_reward += r
                episode_step += 1

                # Book-keeping.
                epoch_actions.append(action)
                epoch_qs.append(q)
                agent.store_transition(obs, action, r, new_obs, done)
                obs = new_obs

                if done:
                    # Episode done.
                    epoch_episode_rewards.append(episode_reward)
                    episode_rewards_history.append(episode_reward)
                    epoch_episode_steps.append(episode_step)
                    episode_reward = 0.
                    episode_step = 0
                    epoch_episodes += 1
                    episodes += 1

                    agent.reset()
                    obs = env.reset()

            # Train.
            epoch_actor_losses = []
            epoch_critic_losses = []
            epoch_adaptive_distances = []
            for t_train in range(nb_train_steps):
                # Adapt param noise, if necessary.
                if memory.nb_entries >= batch_size and t % param_noise_adaption_interval == 0:
                    distance = agent.adapt_param_noise()
                    epoch_adaptive_distances.append(distance)

                cl, al = agent.train()
                epoch_critic_losses.append(cl)
                epoch_actor_losses.append(al)
                agent.update_target_net()
    return agent


seed = 8902077161928034768
env = gym.make("MountainCarContinuous-v0")
env.seed(seed)
sess = U.make_session(num_cpu=1).__enter__()
nb_actions = env.action_space.shape[-1]
layer_norm=True
param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(0.2), desired_action_stddev=float(0.2))
memory = Memory(limit=int(1e6), action_shape=env.action_space.shape, observation_shape=env.observation_space.shape)
critic = Critic(layer_norm=layer_norm)
actor = Actor(nb_actions, layer_norm=layer_norm)


agent = train_return(env=env,actor=actor, critic=critic, memory=memory, param_noise=param_noise)
max_action = env.action_space.high

from gym import spaces
def compute_traj(**kwargs):
    env.reset()
    if 'init_state' in kwargs:
        ob = kwargs['init_state']
        env.env.state = ob
    if 'goal_pos' in kwargs:
        gp = kwargs['goal_pos']
        env.env.goal_position = gp
    if 'max_speed' in kwargs:
        ms = kwargs['max_speed']
        env.env.max_speed = ms
        env.env.low_state = \
            np.array([env.env.min_position, - env.env.max_speed])
        env.env.high_state = \
            np.array([env.env.max_position, env.env.max_speed])
        env.env.observation_space = \
            spaces.Box(env.env.low_state, env.env.high_state)
    if 'power' in kwargs:
        pow = kwargs['power']
        env.env.power = pow
    if 'max_steps' in kwargs:
        max_steps = kwargs['max_steps']
    else:
        max_steps = np.inf

    iter_time = 0
    reward = 0
    done=False
    traj = [ob]
    while done==False:
        iter_time += 1
        action, _ = agent.pi(ob, apply_noise=False, compute_Q=True)
        ob, r, done, _ = env.step(max_action*action)
        traj.append(ob)
        reward += r
        done = done or iter_time >= max_steps
        if done:
            break
    return traj, {'reward':reward, 'iter_time': iter_time}

def sut(x0, **kwargs):
    if 'max_steps' in kwargs:
        max_steps = kwargs['max_steps']
    else:
        max_steps = np.inf
    return compute_traj(max_steps=max_steps, init_state=x0[0:2],goal_pos=x0[2],
                        max_speed=x0[3], power=x0[4])

from scipy.stats import chi2, norm

def cost_func(X):
    goal_rv = chi2(5, loc=0.3999999, scale=0.05/3.)
    speed_rv = chi2(5, scale=0.005/3.)
    power_rv = norm(0.0015, 0.00075)
    goal_pdf = goal_rv.pdf(X.T[2]) / goal_rv.pdf(0.45)
    speed_pdf = speed_rv.pdf(0.075- X.T[3]) / speed_rv.pdf(0.005)
    power_pdf = power_rv.pdf(X.T[4]) / power_rv.pdf(0.0015)
    goal_pdf.resize(len(goal_pdf), 1)
    speed_pdf.resize(len(speed_pdf), 1)
    power_pdf.resize(len(power_pdf), 1)
    return goal_pdf*speed_pdf*power_pdf

#------------------------------------------------------------------------------------------------------------------
from adversarial_testing import pred_node, max_node, min_node, test_module
from adversarial_testing.utils import sample_from
rand_nums = [3188388221,
 1954593344,
 2154016205,
 3894811078,
 3493033583,
 3248332584,
 1304673443,
 3857496775,
 2668478815,
 278535713,
 1762150547,
 788841329,
 2525132954,
 677754898,
 754758634]

# Requirement 1: Find the initial configuration that minimizes the reward
# We need only one node for the reward. The reward is a smooth function
# given that the closed loop system is deterministic
bounds = [(-0.6, -0.4)] # Bounds on the position
bounds.append((-0.025, 0.025)) # Bounds on the velocity
bounds.append((0.4, 0.6)) # Bounds on the goal position
bounds.append((0.055, 0.075)) # Bounds on the max speed
bounds.append((0.0005, 0.0025)) # Bounds on the power magnitude

smooth_details_r1 = []
random_details_r1 = []

# This set assumes random sampling and checking
for r in rand_nums:
    np.random.seed(r)
    node0 = pred_node(f=lambda traj: traj[1]['reward'])
    TM = test_module(bounds=bounds, sut=lambda x0: sut(x0),
                     f_tree = node0, init_sample = 60,
                     optimize_restarts=5, exp_weight=2,
                     normalizer=True)
    TM.initialize()

    TM.run_BO(140)
    smooth_vals = np.array(TM.f_acqu.find_GP_func())
    smooth_details_r1.append([np.sum(smooth_vals < 75),
                              np.sum(smooth_vals < 0),
                              TM.smooth_min_x,
                              TM.smooth_min_val])




# With cost function
    np.random.seed(r)
    node0_cf = pred_node(f=lambda traj: traj[1]['reward'])
    TM_cf = test_module(bounds=bounds, sut=lambda x0: sut(x0),
                     f_tree = node0_cf, init_sample = 60,
                     optimize_restarts=5, exp_weight=10,
                     normalizer=True)
    TM_cf.initialize()

    TM_cf.run_BO(30)
    TM_cf.k = 5
    TM_cf.run_BO(40)
    TM_cf.k = 2
    TM_cf.run_BO(70)
    smooth_cf_vals = np.array(TM_cf.f_acqu.find_GP_func())
    smooth_details_r1.append([np.sum(smooth_cf_vals < 75),
                              np.sum(smooth_cf_vals < 0),
                              TM_cf.smooth_min_x,
                              TM_cf.smooth_min_val])
    np.random.seed(r)
    node0 = pred_node(f=lambda traj: traj[1]['reward'])
    TM = test_module(bounds=bounds, sut=lambda x0: sut(x0),
                     f_tree=node0, init_sample=70, with_smooth=False,
                     with_random=True,
                     optimize_restarts=5, exp_weight=10,
                     normalizer=True)
    TM.initialize()
    TM.run_BO(130)
    random_details_r1.append([np.sum(TM.random_Y < 75),
                              np.sum(TM.random_Y < 0),
                              TM.rand_min_x,
                              TM.rand_min_val])
    print(r, smooth_details_r1[-2],  smooth_details_r1[-1], random_details_r1[-1])


# Requirement 2: Find the initial configuration that maximizes the time
# to completion. We need only one node for the time.

smooth_details_r2 = []
random_details_r2 = []

# This set assumes random sampling and checking
for r in rand_nums:
    np.random.seed(r)
    node0 = pred_node(f=lambda traj: -traj[1]['iter_time'])
    TM = test_module(bounds=bounds, sut=lambda x0: sut(x0,max_steps=350),
                     f_tree = node0,  init_sample = 60,
                     optimize_restarts=5, exp_weight=2,
                     normalizer=True)
    TM.initialize()

    TM.run_BO(140)
    smooth_vals = np.array(TM.f_acqu.find_GP_func())
    smooth_details_r2.append([np.sum(smooth_vals < -250),
                              np.sum(smooth_vals < -150),
                              TM.smooth_min_x,
                              TM.smooth_min_val])


    print(smooth_details_r2[-1])
# With cost function
    np.random.seed(r)
    node0_cf = pred_node(f=lambda traj: -traj[1]['iter_time'])
    TM_cf = test_module(bounds=bounds, sut=lambda x0: sut(x0,max_steps=350),
                     f_tree = node0_cf, init_sample = 60,
                     optimize_restarts=5, exp_weight=10,
                     normalizer=True)
    TM_cf.initialize()

    TM_cf.run_BO(30)
    TM_cf.k = 5
    TM_cf.run_BO(40)
    TM_cf.k = 2
    TM_cf.run_BO(70)
    smooth_cf_vals = np.array(TM_cf.f_acqu.find_GP_func())
    smooth_details_r2.append([np.sum(smooth_cf_vals < -250),
                              np.sum(smooth_cf_vals < -150),
                              TM_cf.smooth_min_x,
                              TM_cf.smooth_min_val])

    np.random.seed(r)
    TM = test_module(bounds=bounds, sut=lambda x0: sut(x0, max_steps=350),
                     f_tree=node0, init_sample=70, with_smooth=False,
                     with_random=True,
                     optimize_restarts=5, exp_weight=10,
                     normalizer=True)
    TM.initialize()
    TM.run_BO(130)
    random_details_r2.append([np.sum(TM.random_Y < -250),
                              np.sum(TM.random_Y < -150),
                              TM.rand_min_x,
                              TM.rand_min_val])

    print(r, smooth_details_r2[-2],smooth_details_r2[-1],
          random_details_r2[-1])

# Requirement 3 : Find the initial configuration that minimizes the following
# requirement :
# 1. Either the car remains within the initial condition of state and velocity
# 2. Reaches the goal asap

smooth_details_r3 = []
ns_details_r3 = []
random_details_r3 = []

def pred1(traj):
    traj = traj[0]
    x_s = np.array(traj).T[0]
    init_x = x_s[0]
    dev = init_x - x_s
    dev = np.sum(np.abs(dev))
    return -dev/350.

def pred2(traj):
    iters = traj[1]['iter_time']
    return -iters/350.

def pred3(traj):
    traj=traj[0]
    v_s = np.array(traj).T[1]
    return min(0.025 - np.abs(v_s))

for r in rand_nums:
    np.random.seed(r)
    node0 = pred_node(f=pred1)
    node1 = pred_node(f=pred2)
    node2 = pred_node(f=pred3)
    node3= min_node(children=[node0, node2])
    node4= max_node(children=[node3, node1])
    TM = test_module(bounds=bounds, sut=lambda x0: sut(x0,max_steps=350),
                     f_tree = node4, init_sample = 60,
                     optimize_restarts=5, exp_weight=2,
                     normalizer=True)
    TM.initialize()
    TM.run_BO(140)
    smooth_vals = TM.f_acqu.find_GP_func()
    smooth_details_r3.append([np.sum(smooth_vals < -0.25),
                              np.sum(smooth_vals < -0.30),
                              TM.smooth_min_x,
                              TM.smooth_min_val,
                              TM.smooth_min_loc])


# With cost function
    np.random.seed(r)
    node0_ns = pred_node(f=pred1)
    node1_ns = pred_node(f=pred2)
    node2_ns = pred_node(f=pred3)
    node3_ns = min_node(children=[node0_ns, node2_ns])
    node4_ns = max_node(children=[node3_ns, node1_ns])
    TM_ns = test_module(bounds=bounds, sut=lambda x0: sut(x0,max_steps=350),
                     f_tree = node4_ns,  with_smooth=False,
                     with_ns = True, init_sample = 60,
                     optimize_restarts=5, exp_weight=2, cost_model=cost_func,
                     normalizer=True)
    TM_ns.initialize()
    TM_ns.run_BO(140)
    ns_details_r3.append([np.sum(TM_ns.ns_GP.Y < -0.25),
                          np.sum(TM_ns.ns_GP.Y < -0.3),
                        TM_ns.ns_min_x,
                        TM_ns.ns_min_val,
                        TM_ns.ns_min_loc])

    np.random.seed(r)
    node0 = pred_node(f=pred1)
    node1 = pred_node(f=pred2)
    node2 = pred_node(f=pred3)
    node3 = min_node(children=[node0, node2])
    node4 = max_node(children=[node3, node1])
    TM = test_module(bounds=bounds, sut=lambda x0: sut(x0, max_steps=350),
                     f_tree=node4, init_sample=70, with_smooth=False,
                     with_random=True,
                     optimize_restarts=5, exp_weight=10,
                     normalizer=True)
    TM.initialize()
    TM.run_BO(130)

    random_details_r3.append([np.sum(TM.random_Y < -0.25),
                              np.sum(TM.random_Y < -0.3),
                              TM.rand_min_x,
                              TM.rand_min_val,
                              TM.rand_min_loc])
    print(r, smooth_details_r3[-1], ns_details_r3[-1],
          random_details_r3[-1])
