import gymnasium as gym 
import flappy_bird_gymnasium
import agent_rl as agent

env = gym.make("FlappyBird-v0", render_mode="human", use_lidar=False)
step = 0
jump = 0


obs, info = env.reset()
# action = 1

# State extractor 



def state_extractor(obs):
    y = obs[9] # players y position
    pipe_top = obs[4] # Next top pipes y position
    pipe_bottom = obs[5] # Next bottom pipes x position
    velocity = obs[10] # The players Velocity 
    # rising, fast_fall and slow_fall will be if statements depending on the velocity 
    gap_min = min(pipe_top, pipe_bottom)
    gap_max = max(pipe_top, pipe_bottom)
    if y < gap_min:
        gap_index = 0 # above the pipe gap
        print(f"Above Gap: {gap_index}")
    elif y > gap_max:
        gap_index = 2 # below the pipe gap
        print(f"Below Gap: {gap_index}")
    else:
        gap_index = 1
        print(f"Between Gap: {gap_index}")
    


    if velocity > 0:
        velo_index = 0 # Rising 
        print(f"Rising: {velo_index}")
    elif velocity < -0.6:
        velo_index = 2 # Fast Falling
        print(f"Fast Falling: {velo_index}")
    else:
        velo_index = 1 # Slow Falling
        print(f"Slow Falling: {velo_index}")



    state_index = gap_index * 3 + velo_index
    print(f"State_Index: {state_index}")
    return state_index


next_state = None



while True: 
    
    # Next Actions:
    # (Feed the observations to your agent)

    # Random action step
    #action = env.action_space.sample()
    #obs, reward, terminated, _, info = env.step(action)
    state = state_extractor(obs)


    # picking the correct action to use 
    action = agent.choose_action(state)


    obs, reward, terminated, truncated, info = env.step(action)
    
    done = terminated or truncated
    agent.update(state, action, reward, next_state, done)
    state = next_state

    
    # jump every 5 time steps
    '''
    jump = step % 20
    if (jump == 1):
        action = 1
        obs, reward, terminated, _, info = env.step(action)
        print(f"observations after jump: {obs[:]}")
    else:
        action = 0
        obs, reward, terminated, _, info = env.step(action)
        print(f"observations not jumping: {obs[:]}")
    '''
    
    
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
    if done:
        #print(env.observation_space)
        #print(env.spec)
        #print(env.unwrapped)
        obs, info = env.reset()

        break

env.close()