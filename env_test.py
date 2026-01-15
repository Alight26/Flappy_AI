import gymnasium as gym 
import flappy_bird_gymnasium

env = gym.make("FlappyBird-v0", render_mode="human", use_lidar=False)
step = 0
jump = 0


obs, info = env.reset()
action = 1

# State extractor 



def state_extractor(obs):
    y = obs[9] # players y position
    pipe_top = obs[4] # Next top pipes y position
    pipe_bottom = obs[5] # Next bottom pipes x position
    velocity = obs[10] # The players Velocity 
    # rising, fast_fall and slow_fall will be if statements depending on the velocity 
    if y > pipe_top:
        gap_index = 0 # above the pipe gap
    elif y < pipe_bottom:
        gap_index = 1 # below the pipe gap
    else:
        gap_index = 2

    if velocity > 0:
        velo_index = 0 # Rising 
    elif velocity < -0.6:
        velo_index = 1 # Fast Falling
    else:
        velo_index = 2 # Slow Falling



while True: 
    step += 1
    # Next Actions:
    # (Feed the observations to your agent)

    # Random action step
    #action = env.action_space.sample()


    # jump every 5 time steps
    jump = step % 20
    if (jump == 1):
        action = 1
        obs, reward, terminated, _, info = env.step(action)
        print(f"observations after jump: {obs[:]}")
    else:
        action = 0
        obs, reward, terminated, _, info = env.step(action)
        print(f"observations not jumping: {obs[:]}")

    
    print(f"step: {step}")
    print(f"action: {action}")
    print(f"reward: {reward}")
    print(f"Terminated: {terminated}")
    print(f"Score: {info}")


    
    


    #processing:
    #obs, reward, terminated, _, info = env.step(action)
    """
    print(f"Observation Space: {env.observation_space}")
    print(f"Observation Length: {len(obs)}")

    
    print(f"info: {info}")
    
    
    """
    

    # Checking if player is still alive 
    if terminated:
        #print(env.observation_space)
        #print(env.spec)
        #print(env.unwrapped)
        obs, info = env.reset()

        break

env.close()