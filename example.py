#!/usr/bin/env python
# coding: utf-8

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from builtins import input
import deeprl_hw1.lake_envs as lake_env
import gym
import time, sys
sys.path.append('deeprl_hw1')
import numpy as np
import rl

def run_random_policy(env, policy):
    """Run a random policy for the given environment.

    Logs the total reward and the number of steps until the terminal
    state was reached.

    Parameters
    ----------
    env: gym.envs.Environment
      Instance of an OpenAI gym.

    Returns
    -------
    (float, int)
      First number is the total undiscounted reward received. The
      second number is the total number of actions taken before the
      episode finished.
    """
    initial_state = env.reset()
    # env.render()
    # time.sleep(1)  # just pauses so you can see the output

    total_reward = 0
    num_steps = 0
    nextstate = initial_state
    while True:
        nextstate, reward, is_terminal, debug_info = env.step(policy[nextstate])
        # env.render()

        total_reward += reward
        num_steps += 1
        if is_terminal:
            break

        # time.sleep(1)

    return total_reward, num_steps

def print_env_info(env):
    print('Environment has %d states and %d actions.' % (env.nS, env.nA))

def print_model_info(env, state, action):
    transition_table_row = env.P[state][action]
    print(
        ('According to transition function, '
         'taking action %s(%d) in state %d leads to'
         ' %d possible outcomes') % (lake_env.action_names[action],
                                     action, state, len(transition_table_row)))
    for prob, nextstate, reward, is_terminal in transition_table_row:
        state_type = 'terminal' if is_terminal else 'non-terminal'
        print(
            '\tTransitioning to %s state %d with probability %f and reward %f'
            % (state_type, nextstate, prob, reward))

def main():
    from matplotlib import pyplot as plt

    # env_dict = {1:'Deterministic-4x4-FrozenLake-v0', 2:'Deterministic-8x8-FrozenLake-v0'}

    env = gym.make('Deterministic-4x4-FrozenLake-v0')
    gamma_a = 0.9
    t0 = time.time()
    policy_imp, value_function, policy_improvement_idx, policy_eval_idx = rl.policy_iteration(env, gamma_a)
    rl.print_policy(policy_imp.reshape(4,4), lake_env.action_names)
    t1 = time.time()
    print("4*4: policy_improvement_idx {}, policy_eval_idx {}, time {} s".format(policy_improvement_idx, policy_eval_idx, t1-t0))
    total_reward, num_steps = run_random_policy(env, policy_imp)
    # print("4*4 total_reward {}".format(total_reward))
    print('Agent received total reward of: %f' % total_reward)
    print('Agent took %d steps' % num_steps)
    print('discounted reward')
    print(value_function.shape)

    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(value_function.reshape(4,4), cmap=plt.cm.jet, interpolation='nearest')
    fig.colorbar(res)
    plt.savefig('4*4.png', format='png')

    t0 = time.time()
    value_function, iter_idx =  rl.value_iteration(env, gamma_a)
    t1 = time.time()
    print("time :{}, iter_idx {} ".format(t1-t0, iter_idx))
    value_function_print = np.asarray(value_function)
    value_function_print = value_function_print.reshape(4,4).astype(np.float16)
    np.set_printoptions(precision=4)
    print(" \\\\\n".join([" & ".join(map(str,line)) for line in value_function_print]))
    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(value_function.reshape(4,4), cmap=plt.cm.jet, interpolation='nearest')
    fig.colorbar(res)
    plt.savefig('4*4_val_iter.png', format='png')
    policy_from_value = rl.value_function_to_policy(env, gamma_a, value_function)
    rl.print_policy(policy_from_value.reshape(4,4), lake_env.action_names)
    print("\n\n\n\n\n\n\n\n\n")

################################### part b ###################################
    env = gym.make('Deterministic-8x8-FrozenLake-v0')
    gamma_a = 0.9
    t0 = time.time()
    policy_imp, value_function, policy_improvement_idx, policy_eval_idx = rl.policy_iteration(env, gamma_a)
    rl.print_policy(policy_imp.reshape(8,8), lake_env.action_names)
    t1 = time.time()
    print("8*8: policy_improvement_idx {}, policy_eval_idx {}, time {} s".format(policy_improvement_idx, policy_eval_idx, t1-t0))
    total_reward, num_steps = run_random_policy(env, policy_imp)
    # print("8*8 total_reward {}".format(total_reward))
    print('Agent received total reward of: %f' % total_reward)
    print('Agent took %d steps' % num_steps)
    from matplotlib import pyplot as plt
    print(value_function.shape)
    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(value_function.reshape(8,8), cmap=plt.cm.jet, interpolation='nearest')
    fig.colorbar(res)
    plt.savefig('8*8.png', format='png')

    t0 = time.time()
    value_function, iter_idx =  rl.value_iteration(env, gamma_a)
    t1 = time.time()
    print("time :{}, iter_idx {} ".format(t1-t0, iter_idx))
    value_function_print = np.asarray(value_function)
    value_function_print = value_function_print.reshape(8,8).astype(np.float16)
    np.set_printoptions(precision=4)
    print(" \\\\\n".join([" & ".join(map(str,line)) for line in value_function_print]))
    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(value_function.reshape(8,8), cmap=plt.cm.jet, interpolation='nearest')
    fig.colorbar(res)
    plt.savefig('8*8_val_iter.png', format='png')
    policy_from_value = rl.value_function_to_policy(env, gamma_a, value_function)
    rl.print_policy(policy_from_value.reshape(8,8), lake_env.action_names)

    print("\n\n\n\n\n\n\n\n\n")

    env = gym.make('Stochastic-4x4-FrozenLake-v0')
    gamma_a = 0.9

    t0 = time.time()
    value_function, iter_idx =  rl.value_iteration(env, gamma_a)
    t1 = time.time()
    print("time :{}, iter_idx {} ".format(t1-t0, iter_idx))
    value_function_print = np.asarray(value_function)
    value_function_print = value_function_print.reshape(4,4).astype(np.float16)
    np.set_printoptions(precision=4)
    print(" \\\\\n".join([" & ".join(map(str,line)) for line in value_function_print]))
    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(value_function.reshape(4,4), cmap=plt.cm.jet, interpolation='nearest')
    fig.colorbar(res)
    plt.savefig('4*4_stochastic_val_iter.png', format='png')
    policy_from_value = rl.value_function_to_policy(env, gamma_a, value_function)
    rl.print_policy(policy_from_value.reshape(4,4), lake_env.action_names)

    meta_total_reward = 0
    meta_num_steps = 0
    for i in range(100):
        total_reward, num_steps = run_random_policy(env, policy_from_value)
        # print('Agent received total reward of: %f' % total_reward)
        # print('Agent took %d steps' % num_steps)
        meta_total_reward += total_reward
        meta_num_steps += num_steps
    print("meta_total_reward", meta_total_reward)
    print("meta_num_steps", meta_num_steps)

    print("\n\n\n\n\n\n\n\n\n")


    env = gym.make('Stochastic-8x8-FrozenLake-v0')
    gamma_a = 0.9

    t0 = time.time()
    value_function, iter_idx =  rl.value_iteration(env, gamma_a)
    t1 = time.time()
    print("time :{}, iter_idx {} ".format(t1-t0, iter_idx))
    value_function_print = np.asarray(value_function)
    value_function_print = value_function_print.reshape(8,8).astype(np.float16)
    np.set_printoptions(precision=4)
    print(" \\\\\n".join([" & ".join(map(str,line)) for line in value_function_print]))
    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(value_function.reshape(8,8), cmap=plt.cm.jet, interpolation='nearest')
    fig.colorbar(res)
    plt.savefig('8*8_stochastic_val_iter.png', format='png')
    policy_from_value = rl.value_function_to_policy(env, gamma_a, value_function)
    rl.print_policy(policy_from_value.reshape(8,8), lake_env.action_names)
    for i in range(100):
        total_reward, num_steps = run_random_policy(env, policy_from_value)
        # print('Agent received total reward of: %f' % total_reward)
        # print('Agent took %d steps' % num_steps)
        meta_total_reward += total_reward
        meta_num_steps += num_steps
    print("meta_total_reward", meta_total_reward)
    print("meta_num_steps", meta_num_steps)

    print("\n\n\n\n\n\n\n\n\n")

    env = gym.make('Deterministic-4x4-neg-reward-FrozenLake-v0')
    gamma_a = 0.9
    t0 = time.time()
    value_function, iter_idx =  rl.value_iteration(env, gamma_a)
    t1 = time.time()
    print("time :{}, iter_idx {} ".format(t1-t0, iter_idx))
    value_function_print = np.asarray(value_function)
    value_function_print = value_function_print.reshape(4,4).astype(np.float16)
    np.set_printoptions(precision=4)
    print(" \\\\\n".join([" & ".join(map(str,line)) for line in value_function_print]))
    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(value_function.reshape(4,4), cmap=plt.cm.jet, interpolation='nearest')
    fig.colorbar(res)
    plt.savefig('4*4_neg_val_iter.png', format='png')

    policy_from_value = rl.value_function_to_policy(env, gamma_a, value_function)
    rl.print_policy(policy_from_value.reshape(4,4), lake_env.action_names)
    total_reward, num_steps = run_random_policy(env, policy_from_value)
    print('Agent received total reward of: %f' % total_reward)
    print('Agent took %d steps' % num_steps)

if __name__ == '__main__':
    main()
