import gym
import abides_gym

print("Initializing rmsc04 gym environment...")
env = gym.make(
    "markets-daily_investor-v0",
    background_config="rmsc04",
)

env.seed(0)
initial_state = env.reset()
print("Trying to run 5 steps in the environment:")
for i in range(5):
    state, reward, done, info = env.step(0)
    print("state: ", state, "\nreward: ", reward)