#import dill
from collections import defaultdict
from enum import Enum
import random


# keep a dict that has episode number and rewards

class Agent:
    class UpdateMethod(Enum):
        SARSA = 1
        SARSA_MAX = 2
        EXPECTED_SARSA = 3

    def __init__(self, nA=15, epsilon=0.05, alpha=0.1, gamma=1, update_method=UpdateMethod.SARSA):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: [0.0]*self.nA)
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.update_method = update_method
        self.i_episode = 0
        self.reward_history = {}
        self.state_visit_counter = {}

    def get_policy_probs(self, state):
        """ Given the state, return the probability of each action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - probs: an array, each element corresponds to probability of corresponding action selected
        """
        def argmax(a):
            return max(range(len(a)), key=lambda x: a[x])

        probs = [self.epsilon/self.nA] * self.nA
        probs[argmax(self.Q[state])] += 1 - self.epsilon
        return probs

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        probs = self.get_policy_probs(state)
        return random.choices(list(range(self.nA)), probs, k=1)[0]

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        # Count how often we visit each state
        if not (state in self.state_visit_counter):
            self.state_visit_counter[state] = 1
        else:
            self.state_visit_counter[state] += 1

        if (done == False):
            self.epsilon = 1.0 / (1.0 + self.i_episode)
            if (self.update_method == self.UpdateMethod.SARSA):
                next_action = self.select_action(state)
                self.Q[state][action] += self.alpha * (reward + self.gamma * self.Q[next_state][next_action] - self.Q[state][action])
            elif (self.update_method == self.UpdateMethod.SARSA_MAX):
                self.Q[state][action] += self.alpha * (reward + self.gamma * max(self.Q[next_state]) - self.Q[state][action])
            elif (self.update_method == self.UpdateMethod.EXPECTED_SARSA):
                probs = self.get_policy_probs(state)
                self.Q[state][action] += self.alpha * (reward + self.gamma * sum([q_val * prob for (q_val, prob) in zip(self.Q[next_state], probs)]) - self.Q[state][action])
        else:  # done == True
            self.Q[state][action] += self.alpha * (reward - self.Q[state][action])
            self.i_episode += 1
            print(f"Episode complete, epsilon will now be {1.0 / (1.0 + self.i_episode)}")


            # Pickle/dill the current Q function so that we can restore it if we want
            #print(f"Dilling Q function {dict(self.Q)}")
            #dilled_Q = dill.dumps(self.Q)
