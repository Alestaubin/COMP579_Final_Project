import torch
import os
class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = torch.zeros(shape, 1)
        self.var = torch.ones(shape, 1)
        self.count = epsilon
        self.active = False
        self.x = []

    def update(self):
        # breakpoint()
        x = torch.tensor(self.x)
        batch_mean = torch.mean(x, dim=0)  # Compute mean along the first dimension (batch)
        batch_var = torch.var(x, dim=0)    # Compute variance along the first dimension (batch)
        batch_count = x.size(0)           # Get the number of data points in the batch
        self.update_from_moments(batch_mean, batch_var, batch_count)
        self.active = True
        self.x = []

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + torch.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

    def normalize(self,x):
        self.x.append(x.tolist())
        if self.active:
            return (x - self.mean) / (torch.sqrt(self.var) + 1e-5)
        else:
            return x
class Tester():
    def __init__(self, env, agent=None):
        """
        Overview:
            Initializes the Tester class.

        Arguments:
            - env: (`object`): The environment object.
            - agent: (`object`): The agent object.
        """
        self.env            = env
        self.observation_space_size = env.observation_space.shape[0]
        self.action_space_size = env.action_space.n
        
        self.agent          = agent

    def test(self, step=0):
        """
        Overview:
            Runs the test agent.
        """
        print("Testing agent...")
        state = self.env.reset()
        done = False
        rewards = []
        holdings = []

        while not done:
            state, reward, done, _ = self.agent.play(state, testing=True)
            rewards.append(reward)
            holdings.append(state[0])
        
        # plotting
        plot_dir = "plot"
        os.makedirs(plot_dir, exist_ok=True)

        # Plot rewards
        plt.figure()
        plt.plot(rewards)
        plt.title('Rewards over time')
        plt.xlabel('Time step')
        plt.ylabel('Reward')
        reward_path = os.path.join(plot_dir, f"rewards_{step}.png")
        plt.savefig(reward_path)
        plt.close()

        # Plot holdings
        plt.figure()
        plt.plot(holdings)
        plt.title('Holdings over time') 
        plt.xlabel('Time step')
        plt.ylabel('Holdings')
        holdings_path = os.path.join(plot_dir, f"holdings_{step}.png")
        plt.savefig(holdings_path)
        plt.close()