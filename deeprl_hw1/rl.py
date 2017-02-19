# coding: utf-8
from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import numpy as np


def evaluate_policy(env, gamma, policy, max_iterations=int(1e3), tol=1e-3):
	"""Evaluate the value of a policy.

	See page 87 (pg 105 pdf) of the Sutton and Barto Second Edition
	book.

	http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep. 

	Parameters
	----------
	env: gym.core.Environment
	  The environment to compute value iteration for. Must have nS,
	  nA, and P as attributes.
	gamma: float
	  Discount factor, must be in range [0, 1)
	policy: np.array
	  The policy to evaluate. Maps states to actions.
	max_iterations: int
	  The maximum number of iterations to run before stopping.
	tol: float
	  Determines when value function has converged.

	Returns
	-------
	np.ndarray, int
	  The value for the given policy and the number of iterations till
	  the value function converged.
	"""

	value_function = np.zeros(env.nS) # or rand?
	iter_idx = 0
	while iter_idx < max_iterations:
		iter_idx += 1
		delta = 0
		for state_idx in range(env.nS):
			v_state_curr = value_function[state_idx]
			value_function[state_idx] = 0
			action_idx = policy[state_idx]
			for prob_state, next_state_idx, reward, _ in env.P[state_idx][action_idx]:
				value_function[state_idx] += prob_state * (reward + gamma*value_function[next_state_idx])
			delta = max(delta, np.absolute(v_state_curr - value_function[state_idx]))
			# print("iter_idx : {}, delta : {}".format(iter_idx, delta))
		if delta < tol:
			break
	return value_function, iter_idx

def value_function_to_policy(env, gamma, value_function):
	"""Output action numbers for each state in value_function.

	Parameters
	----------
	env: gym.core.Environment
	  Environment to compute policy for. Must have nS, nA, and P as
	  attributes.
	gamma: float
	  Discount factor. Number in range [0, 1)
	value_function: np.ndarray
	  Value of each state.

	Returns
	-------
	np.ndarray
	  An array of integers. Each integer is the optimal action to take
	  in that state according to the environment dynamics and the
	  given value function.
	"""    
	policy = np.zeros(env.nS, dtype='int')
	for state_idx in range(env.nS):
		q_vals = np.zeros(env.nA)
		for action_idx in range(env.nA):
			for prob_state, next_state_idx, reward, _ in env.P[state_idx][action_idx]:
				q_vals[action_idx] += prob_state * (reward + gamma*value_function[next_state_idx])			
		policy[state_idx] = np.argmax(q_vals)

	return policy


def improve_policy(env, gamma, value_func, policy):
	"""Given a policy and value function improve the policy.

	See page 87 (pg 105 pdf) of the Sutton and Barto Second Edition
	book.

	http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

		Parameters
	----------
	env: gym.core.Environment
	  The environment to compute value iteration for. Must have nS,
	  nA, and P as attributes.
	gamma: float
	  Discount factor, must be in range [0, 1)
	value_func: np.ndarray
	  Value function for the given policy.
	policy: dict or np.array
	  The policy to improve. Maps states to actions.

	Returns
	-------
	bool, np.ndarray
	  Returns true if policy changed. Also returns the new policy.
	
	"""
	policy_stable = True
	for state_idx in range(env.nS):
		old_action_idx = policy[state_idx]
		q_vals = np.zeros(env.nA)
		for new_action_idx in range(env.nA):
			for prob_state, next_state_idx, reward, _ in env.P[state_idx][new_action_idx]:
				q_vals[new_action_idx] += prob_state * (reward + gamma*value_func[next_state_idx])			
		policy[state_idx] = np.argmax(q_vals)

		if old_action_idx != policy[state_idx]:
			policy_stable = False
	return not(policy_stable), policy

def policy_iteration(env, gamma, max_iterations=int(1e3), tol=1e-3):
	"""Runs policy iteration.

	See page 87 (pg 105 pdf) of the Sutton and Barto Second Edition
	book.

	http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

	You should use the improve_policy and evaluate_policy methods to
	implement this method.

	Parameters
	----------
	env: gym.core.Environment
	  The environment to compute value iteration for. Must have nS,
	  nA, and P as attributes.
	gamma: float
	  Discount factor, must be in range [0, 1)
	max_iterations: int
	  The maximum number of iterations to run before stopping.
	tol: float
	  Determines when value function has converged.

	Returns
	-------
	(np.ndarray, np.ndarray, int, int)
	   Returns optimal policy, value function, number of policy
	   improvement iterations, and number of value iterations.
	"""
	policy_changed = True
	policy_eval_idx = 0
	improvement_iter = 0
	value_function = np.zeros(env.nS)
	policy = np.zeros(env.nS, dtype='int')

	while (improvement_iter < max_iterations) and policy_changed:
		value_function, eval_iter_curr = evaluate_policy(env, gamma, policy)
		# print("eval value_function ", value_function)
		# print("eval policy ", policy)
		policy_changed, policy = improve_policy(env, gamma, value_function, policy)
		# print("improve value_function ", value_function)
		# print("improve policy ", policy)
		policy_eval_idx += eval_iter_curr
		improvement_iter += 1 
		# print("policy_eval_idx : {}, improvement_iter : {}, policy_stable : {}".format(policy_eval_idx, improvement_iter, policy_changed))
	return policy, value_function, improvement_iter, policy_eval_idx

def value_iteration(env, gamma, max_iterations=int(1e3), tol=1e-3):
	"""Runs value iteration for a given gamma and environment.

	See page 90 (pg 108 pdf) of the Sutton and Barto Second Edition
	book.

	http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

	Parameters
	----------
	env: gym.core.Environment
	  The environment to compute value iteration for. Must have nS,
	  nA, and P as attributes.
	gamma: float
	  Discount factor, must be in range [0, 1)
	max_iterations: int
	  The maximum number of iterations to run before stopping.
	tol: float
	  Determines when value function has converged.

	Returns
	-------
	np.ndarray, iteration
	  The value function and the number of iterations it took to converge.
	"""
	value_function = np.zeros(env.nS)
	iter_idx = 0
	while iter_idx < max_iterations:
		iter_idx += 1
		delta = 0
		for state_idx in range(env.nS):
			q_vals = np.zeros(env.nA)
			v_state_curr = value_function[state_idx]
			for new_action_idx in range(env.nA):
				for prob_state, next_state_idx, reward, is_terminal in env.P[state_idx][new_action_idx]:
					q_vals[new_action_idx] += prob_state * (reward + gamma*(not(is_terminal))*value_function[next_state_idx])
			value_function[state_idx] = max(q_vals)
			delta = max(delta, np.absolute(v_state_curr - value_function[state_idx]))
		if delta < tol:
			break
	return value_function, iter_idx

def print_policy(policy, action_names):
	"""Print the policy in human-readable format.

	Parameters
	----------
	policy: np.ndarray
	  Array of state to action number mappings
	action_names: dict
	  Mapping of action numbers to characters representing the action.
	"""
	str_policy = policy.astype('str')
	for action_num, action_name in action_names.items():
		np.place(str_policy, policy == action_num, action_name)

	print(str_policy)
