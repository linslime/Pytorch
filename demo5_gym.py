import gym

env = gym.make("CartPole-v1", render_mode = "human")
env = env.unwrapped # 不做这个会有很多限制
env.reset()
for _ in range(100):
    env.render()
    action = env.action_space.sample()
    env.step(action)
env.close()
