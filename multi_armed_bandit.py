import numpy as np
from math import exp


# Create a class for every iteration initial state
class Board:

    def __init__(self, num_arms, mean, stdev):
        self.narms = num_arms  # Number of options
        self.mean = mean  # Mean for initialize value of options
        self.stdev = stdev  # Standard deviation for initialize value of options

        self.optim = 0  # optimal value of the initial values
        self.action_arr = np.zeros(num_arms)  # List for options
        self.reset()

    def reset(self):
        # Initialize first values of arms with mean and stdev
        self.action_arr = np.random.normal(self.mean, self.stdev, self.narms)
        # Select the otimal value
        self.optim = np.argmax(self.action_arr)


# Create a class for each agent with different ep and te
class Agent:

    def __init__(self, narms, epsilon=0, tempreture=0):
        self.narms = narms  # Number of options
        self.ep = epsilon  # Value of the epsilon
        self.te = tempreture  # Value of the tempreture

        # Initialize lists
        self.action_counter = np.zeros(narms)
        self.sum_reward = np.zeros(narms)
        self.reward_average = np.zeros(narms)

        self.last_action = None

    def __str__(self):
        if self.te:
            return 'Softmax'
        else:
            if self.ep:
                return "Epsilon = " + str(self.ep)
            else:
                return 'Greedy'

    def action(self):
        # If agent use tempreture we should use sofmax selection
        if self.te:
            nominator = np.exp(self.reward_average/self.te)
            exps = nominator/np.sum(nominator)
            action = np.random.choice(len(self.reward_average), p=exps)

        # If the agent doesn't use tempretur we should use epsilon
        else:
            rand = np.random.random()

            # Explore or exploit
            if rand < self.ep:
                action = np.random.choice(len(self.reward_average))  # Select random action

            else:
                # Select an action with maximum average reward
                maxaction = np.argmax(self.reward_average)
                actions = np.where(self.reward_average == maxaction)[0]

                # Choose a random action if values are equal
                if len(actions) == 0:
                    action = maxaction
                else:
                    action = np.random.choice(actions)

        self.last_action = action
        return action

    def updating(self, reward):
        # Reset agents for next iteration
        at = self.last_action

        self.action_counter[at] += 1
        self.sum_reward[at] += reward

        self.reward_average[at] += self.sum_reward[at]/self.action_counter[at]

    def reset(self):
        self.last_action = None
        self.action_counter[:] = 0
        self.sum_reward[:] = 0
        self.reward_average[:] = 0


# Create a class for Running
class RunTest:

    def __init__(self, board, agents, plays, iterations):
        self.board = board              # Our initialized board
        self.agents = agents            # Agents to play
        self.iterations = iterations    # Number of iterations
        self.plays = plays              # Number of plays per itaration

    def play(self):
        # Array to store the rewards
        score_arr = np.zeros((self.plays, len(self.agents)))
        # Array to store optimal choice
        optiml_arr = np.zeros((self.plays, len(self.agents)))

        for iteration in range(self.iterations):
            if (iteration % 100) == 0:
                print("Completed Iterations: ", iteration)

            self.board.reset()

            for agent in self.agents:
                agent.reset()

            for play in range(self.plays):

                for agent_counter, agent in enumerate(self.agents):
                    action = agent.action()

                    reward = np.random.normal(self.board.action_arr[action], 1)
                    agent.updating(reward)

                    score_arr[play, agent_counter] += reward

                    if action == self.board.optim:
                        optiml_arr[play, agent_counter] += 1

        score_avg = score_arr / self.iterations
        optiml_avg = optiml_arr / self.iterations

        return score_avg, optiml_avg
