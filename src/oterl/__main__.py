import abides_gym
import gym


def main():
  print('Initializing rmsc04 gym environment...')

  env = gym.make(
    'markets-daily_investor-v0',
    background_config='rmsc04',
  )

  env.reset()

  env.seed(0)

  print('Trying to run 5 steps in the environment:')

  for _ in range(5):
    state, reward, _, _ = env.step(0)
    print('state: ', state, '\nreward: ', reward)
