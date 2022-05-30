import matplotlib.pyplot as plt

from multi_armed_bandit import *

if __name__ == '__main__':
    num_arms = 10
    iterations = 2000
    plays = 1000
    mean = 0
    stdev = 1

    board = Board(num_arms, mean, stdev)
    agents = [Agent(num_arms, 0, 0),
              Agent(num_arms, 0.1, 0),
              Agent(num_arms, 0.01, 0),
              Agent(num_arms, 0, 1)]
    play = RunTest(board, agents, plays, iterations)

    g1_data, g2_data = play.play()

    # Graph 1 - Averate rewards over all plays
    plt.title("Average Rewards")
    plt.plot(g1_data)
    plt.ylabel('Average Reward')
    plt.xlabel('Plays')
    plt.legend(agents, loc=4)
    plt.show()

    # Graph 1 - optimal selections over all plays
    plt.title("Optimal Action")
    plt.plot(g2_data * 100)
    plt.ylim(0, 100)
    plt.ylabel('% Optimal Action')
    plt.xlabel('Plays')
    plt.legend(agents, loc=4)
    plt.show()
