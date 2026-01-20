import gymnasium as gym 
import flappy_bird_gymnasium
from agent_rl import Agent

train_env = gym.make("FlappyBird-v0", render_mode=None, use_lidar=False)
demo_env = gym.make("FlappyBird-v0", render_mode="human", use_lidar=False)
step = 0
jump = 0
count = 0


TOTAL_EPISODES = 1000000
RENDER_EVERY = 500


obs, info = train_env.reset()
# action = 1


# Creating my agent 
num_states = 9
num_actions = 2

agent = Agent(
    num_states= num_states,
    num_actions= num_actions,
    alpha = 0.1,
    gamma = 0.9,
    epsilon = 1.0,
    epsilon_decay= 0.995,
    min_epsilon=0.05,
    num_episode=0
)
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

def run_demo_episode():
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0 # temp removes it so I can see what the agent has learned

    demo_obs, demo_info = demo_env.reset()
    demo_done = False 

    while not demo_done: 
        demo_state = state_extractor(demo_obs)
        demo_action = agent.choose_action(demo_state)

        demo_next_obs, demo_reward, demo_terminated, demo_truncated, demo_info, = demo_env.step(demo_action)
        demo_done = demo_terminated or demo_truncated
        demo_obs = demo_next_obs

    agent.epsilon = old_epsilon





while True: 
    count += 1
    
    # Next Actions:
    # (Feed the observations to your agent)

    # Random action step
    #action = env.action_space.sample()
    #obs, reward, terminated, _, info = env.step(action)
    state = state_extractor(obs)


    # picking the correct action to use 
    action = agent.choose_action(state)


    next_obs, reward, terminated, truncated, info = train_env.step(action)
    
    done = terminated or truncated

    next_state = state_extractor(next_obs)

    agent.update(state, action, reward, next_state, done)
    
    obs = next_obs

    
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
    
    
    #print(f"step: {step}")
    count % 50
    if (count == 1):
        print(f"action: {action}")
        print(f"reward: {reward}")
        print(f"Terminated: {terminated}")
        print(f"Score: {info}")
        print(f"Count: {count}")


    
    


    #processing:
    #obs, reward, terminated, _, info = env.step(action)
    """
    print(f"Observation Space: {env.observation_space}")
    print(f"Observation Length: {len(obs)}")

    
    print(f"info: {info}")
    
    
    """
    

    if done: 
        agent.num_episode += 1
        agent.decay_epsilon()

        #Human rendering mode to check progress
        if agent.num_episode % RENDER_EVERY == 0:
            print(f"This is the demo episode{agent.num_episode}")
            run_demo_episode()

        if agent.num_episode >= TOTAL_EPISODES:
            break 

        obs, info = train_env.reset()
        continue

    continue

train_env.close()
demo_env.close()
