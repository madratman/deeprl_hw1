# coding: utf-8
"""Define the Queue environment from problem 3 here."""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from gym import Env, spaces
from gym.envs.registration import register
from copy import deepcopy

class QueueEnv(Env):
    """Implement the Queue environment from problem 3.

    Parameters
    ----------
    p1: float
      Value between [0, 1]. The probability of queue 1 receiving a new item.
    p2: float
      Value between [0, 1]. The probability of queue 2 receiving a new item.
    p3: float
      Value between [0, 1]. The probability of queue 3 receiving a new item.

    Attributes
    ----------
    nS: number of states
    nA: number of actions
    P: environment model
    """
    metadata = {'render.modes': ['human']}

    SWITCH_TO_1 = 0
    SWITCH_TO_2 = 1
    SWITCH_TO_3 = 2
    SERVICE_QUEUE = 3

    def __init__(self, p1, p2, p3):
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.MultiDiscrete(
            [(1, 3), (0, 5), (0, 5), (0, 5)])
        self.nS = 0
        self.nA = 0
        self.P = (p1, p2, p3)
        self.curr_state = ()

    def _reset(self):
        """Reset the environment.

        The server should always start on Queue 1.

        Returns
        -------
        (int, int, int, int)
          A tuple representing the current state with meanings
          (current queue, num items in 1, num items in 2, num items in
          3).
        """
        self.curr_state = (1, np.random.randint(6), np.random.randint(6), np.random.randint(6))
        return self.curr_state

    def _step(self, action):
        """Execute the specified action.

        Parameters
        ----------
        action: int
          A number in range [0, 3]. Represents the action.

        Returns
        -------
        (state, reward, is_terminal, debug_info)
          State is the tuple in the same format as the reset
          method. Reward is a floating point number. is_terminal is a
          boolean representing if the new state is a terminal
          state. debug_info is a dictionary. You can fill debug_info
          with any additional information you deem useful.
        """
        reward = 0
        is_terminal = False
        debug_info = {}

        if action==0: #SWITCH_TO_1
            self.curr_state[0] = 1
        if action==1: #SWITCH_TO_2
            self.curr_state[0] = 2        
        if action==2: #SWITCH_TO_1
            self.curr_state[0] = 3
        if action==3: # service 
            curr_queue = self.curr_state[0]
            if self.curr_state[curr_queue] > 0:
                self.curr_state[curr_queue] -= 1 
                reward = 1
        if not (self.curr_state[1] and self.curr_state[2] and self.curr_state[3]):
            is_terminal = True

        # chance_1 = [1]*int(100*self.P[0])
        # chance_2 = [2]*int(100*self.P[1])
        # chance_3 = [3]*int(100*self.P[2])
        # chance_all = chance_1+chance_2+chance_3
        # queue_to_update = random.choice(chance_all)
        chance_1 = [0]*int(100*self.P[0]) + [1]*int(100*(1-self.P[0]))
        chance_2 = [0]*int(100*self.P[1]) + [1]*int(100*(1-self.P[1]))
        chance_3 = [0]*int(100*self.P[2]) + [1]*int(100*(1-self.P[2]))
        self.curr_state[1] += chance_1
        self.curr_state[2] += chance_2
        self.curr_state[3] += chance_3

        return (self.curr_state, reward, is_terminal, debug_info)

    def _render(self, mode='human', close=False):
        pass

    def _seed(self, seed=None):
        """Set the random seed.

        Parameters
        ----------
        seed: int, None
          Random seed used by numpy.random and random.
        """
        pass

    def query_model(self, state, action):
        """Return the possible transition outcomes for a state-action pair.

        This should be in the same format at the provided environments
        in section 2.

        Parameters
        ----------
        state
          State used in query. Should be in the same format at
          the states returned by reset and step.
        action: int
          The action used in query.

        Returns
        -------
        [(prob, nextstate, reward, is_terminal), ...]
          List of possible outcomes
        """
        outcomes = []
        reward = 0
        is_terminal = False

        if action==0: #SWITCH_TO_1
            nextstate = deepcopy(state)
            nextstate[0] = 1
            outcomes.append(self.P[0], (nextstate[0], nextstate[1]+1, nextstate[2], nextstate[3]), reward, is_terminal)
            outcomes.append(self.P[1], (nextstate[0], nextstate[1], nextstate[2]+1, nextstate[3]), reward, is_terminal)
            outcomes.append(self.P[2], (nextstate[0], nextstate[1], nextstate[2], nextstate[3]+1), reward, is_terminal)

        if action==1: #SWITCH_TO_2
            nextstate = deepcopy(state)
            nextstate[0] = 2
            outcomes.append(self.P[0], (nextstate[0], nextstate[1]+1, nextstate[2], nextstate[3]), reward, is_terminal)
            outcomes.append(self.P[1], (nextstate[0], nextstate[1], nextstate[2]+1, nextstate[3]), reward, is_terminal)
            outcomes.append(self.P[2], (nextstate[0], nextstate[1], nextstate[2], nextstate[3]+1), reward, is_terminal)
       
        if action==2: #SWITCH_TO_3
            nextstate = deepcopy(state)
            nextstate[0] = 3
            outcomes.append(self.P[0], (nextstate[0], nextstate[1]+1, nextstate[2], nextstate[3]), reward, is_terminal)
            outcomes.append(self.P[1], (nextstate[0], nextstate[1], nextstate[2]+1, nextstate[3]), reward, is_terminal)
            outcomes.append(self.P[2], (nextstate[0], nextstate[1], nextstate[2], nextstate[3]+1), reward, is_terminal)
        
        if action==3: # service 
            nextstate = deepcopy(state)
            curr_queue = nextstate[0]
            if nextstate[curr_queue] > 0:
                nextstate[curr_queue] -= 1 
                reward = 1
                if not (nextstate[1] and nextstate[2] and nextstate[3]):
                    is_terminal = True
                outcomes.append(self.P[0], (nextstate[0], nextstate[1]+1, nextstate[2], nextstate[3]), reward, is_terminal)
                outcomes.append(self.P[1], (nextstate[0], nextstate[1], nextstate[2]+1, nextstate[3]), reward, is_terminal)
                outcomes.append(self.P[2], (nextstate[0], nextstate[1], nextstate[2], nextstate[3]+1), reward, is_terminal)
       
        return outcomes

    def get_action_name(self, action):
        if action == QueueEnv.SERVICE_QUEUE:
            return 'SERVICE_QUEUE'
        elif action == QueueEnv.SWITCH_TO_1:
            return 'SWITCH_TO_1'
        elif action == QueueEnv.SWITCH_TO_2:
            return 'SWITCH_TO_2'
        elif action == QueueEnv.SWITCH_TO_3:
            return 'SWITCH_TO_3'
        return 'UNKNOWN'


register(
    id='Queue-1-v0',
    entry_point='deeprl_hw1.queue_envs:QueueEnv',
    kwargs={'p1': .1,
            'p2': .9,
            'p3': .1})

register(
    id='Queue-2-v0',
    entry_point='deeprl_hw1.queue_envs:QueueEnv',
    kwargs={'p1': .1,
            'p2': .1,
            'p3': .1})
