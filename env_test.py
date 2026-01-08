import gymnasium as gym 
import flappy_bird_gymnasium

env = gym.make("FlappyBird-v0", render_mode="human")

obs, _ = env.reset()

while True: 
    # Next Actions:
    # (Feed the observations to your agent)
    action = env.action_space.sample()

    #processing:
    obs, reward, terminated, _, info = env.step(action)

    # Checking if player is still alive 
    if terminated:
        break

env.close()