#!/usr/bin/env python
# coding: utf-8

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from builtins import input
import deeprl_hw1.lake_envs as lake_env
import gym
import time, sys
sys.path.append('deeprl_hw1')
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
    env.render()
    time.sleep(1)  # just pauses so you can see the output

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
    # create the environment
    # env = gym.make('FrozenLake-v0')
    # uncomment next line to try the deterministic version
    # env = gym.make('Deterministic-4x4-FrozenLake-v0')
    env = gym.make('Deterministic-8x8-FrozenLake-v0')

    # print_env_info(env)
    # print_model_info(env, 0, lake_env.DOWN)
    # print_model_info(env, 1, lake_env.DOWN)
    # print_model_info(env, 14, lake_env.RIGHT)

    input('Hit enter to run a policy...')
    import numpy as np
    gamma_a = 0.9
    t0 = time.time()
    policy_imp, value_function, policy_improvement_idx, policy_eval_idx = rl.policy_iteration(env, gamma_a)
    # rl.print_policy(policy_imp.reshape(4,4), lake_env.action_names)
    rl.print_policy(policy_imp.reshape(8,8), lake_env.action_names)
    t1 = time.time()
    print("4*4: policy_improvement_idx {}, policy_eval_idx {}, time {} s".format(policy_improvement_idx, policy_eval_idx, t1-t0))
    total_reward, num_steps = run_random_policy(env, policy_imp)

    from matplotlib import pyplot as plt
    print(value_function.shape)
    # plt.imshow(value_function.reshape(4,4))
    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    # res = ax.imshow(value_function.reshape(4,4), cmap=plt.cm.jet, interpolation='nearest')
    res = ax.imshow(value_function.reshape(8,8), cmap=plt.cm.jet, interpolation='nearest')
    fig.colorbar(res)
    plt.savefig('4*4.png', format='png')
    plt.savefig('8*8.png', format='png')

if __name__ == '__main__':
    main()
