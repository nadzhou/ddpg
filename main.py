import gym
import numpy as np

from ddpg import Agent

env = gym.make('LunarLanderContinuous-v2')
agent = Agent(alpha=0.00025, beta=0.00025, input_dims=[8], tau=0.01, env=env, batch_size=64, layer1_size=400,
              layer2_size=300, n_actions=2)

np.random.seed(0)

score_history = []

for i in range(1000):
    done = False
    score = 0
    obs = env.reset()

    while not done:
        act = agent.choose_action(obs)
        new_state, reward, done, info = env.step(act)
        agent.remember(obs, act, reward, new_state, int(done))

        agent.learn()
        score += reward
        obs = new_state

    score_history.append(score)
    print(f'Episode {i}, score: {score}')

    if i % 25 == 0:
        agent.save_model()

    # filename = 'lunar-lander.png'
    # plotLearning(score_history, filename, window=100)