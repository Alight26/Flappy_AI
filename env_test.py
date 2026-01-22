import gymnasium as gym 
import flappy_bird_gymnasium
from agent_rl import Agent
import matplotlib.pyplot as plt 

# plt stuff 
episode_rewards = []
episode_lengths = []
epsilon_history = []

'''
(0) the last pipe's horizontal position
(1)the last top pipe's vertical position
(2) the last bottom pipe's vertical position
(3) the next pipe's horizontal position
(4) the next top pipe's vertical position
(5) the next bottom pipe's vertical position
(6) the next next pipe's horizontal position
(7) the next next top pipe's vertical position
(8) the next next bottom pipe's vertical position
(9) player's vertical position
(10) player's vertical velocity
(11) player's rotation
'''

train_env = gym.make("FlappyBird-v0", render_mode=None, use_lidar=False)
demo_env = gym.make("FlappyBird-v0", render_mode="human", use_lidar=False)
step = 0
jump = 0
count = 0


TOTAL_EPISODES = 500000
RENDER_EVERY = 5000


obs, info = train_env.reset()
total_episode_reward = 0
step_in_episode = 0
# action = 1


# Creating my agent 
num_states = 9
num_actions = 2

agent = Agent(
    num_states= num_states,
    num_actions= num_actions,
    alpha = 0.1,
    gamma = 0.95,
    epsilon = 1.0,
    epsilon_decay= 0.00001,
    min_epsilon=0.05,
    num_episode=0
)
# State extractor 



def state_extractor(obs):
    y = obs[9] # players y position
    pipe_top = obs[4] # Next top pipes y position
    pipe_bottom = obs[5] # Next bottom pipes x position
    velocity = obs[10] # The players Velocity 

   # x_var = obs[3] # How far away the next pipe is

    # rising, fast_fall and slow_fall will be if statements depending on the velocity 
    gap_min = min(pipe_top, pipe_bottom)
    gap_max = max(pipe_top, pipe_bottom)
    if y < gap_min:
        gap_index = 0 # above the pipe gap
        #print(f"Above Gap: {gap_index}")
    elif y > gap_max:
        gap_index = 2 # below the pipe gap
        #print(f"Below Gap: {gap_index}")
    else:
        gap_index = 1

        
        #print(f"Between Gap: {gap_index}")
    


    if velocity > 0:
        velo_index = 0 # Rising 
        #print(f"Rising: {velo_index}")
    elif velocity < -0.6:
        velo_index = 2 # Fast Falling
        #print(f"Fast Falling: {velo_index}")
    else:
        velo_index = 1 # Slow Falling
        #print(f"Slow Falling: {velo_index}")



    state_index = gap_index * 3 + velo_index
   # print(f"State_Index: {state_index}")
    return state_index

def gap_idx(obs):
    x_pos = obs[3]
    y = obs[9] # players y position
    pipe_top = obs[4] # Next top pipes y position
    pipe_bottom = obs[5] # Next bottom pipes x position



    # Trying new formula 
    pipe_gap = (pipe_top + pipe_bottom)/2

    centered = y - pipe_gap

    if (centered < -0.15): # very far below gap
        gap_index = 0
    elif (centered < -0.5): # slightly below gap 
        gap_index = 1

    elif (centered == 0): # perfectly centered
        gap_index = 2

    elif (centered > 0.5): # slightly over
        gap_index = 3
    else:
        gap_index = 4 # very over gap

    return gap_index


def run_demo_episode():
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0 # temp removes it so I can see what the agent has learned

    demo_obs, demo_info = demo_env.reset()
    demo_done = False 

    while not demo_done: 
        demo_state = state_extractor(demo_obs)



        demo_action = agent.choose_action(demo_state)

        demo_next_obs, demo_reward, demo_terminated, demo_truncated, demo_info, = demo_env.step(demo_action)
        in_gap = gap_idx(demo_next_obs)
        if (in_gap == 1):
            demo_reward += 0.2
        else: 
            demo_reward -= 0.2




        demo_done = demo_terminated or demo_truncated
        demo_obs = demo_next_obs
        #print(f"demo reward: {demo_reward}")
        #print(f"x position: {demo_obs[3]}")
        


    print(f"demo_reward: {demo_reward}")
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


    next_obs, reward, terminated, truncated, new_info = train_env.step(action)
    in_gap = gap_idx(next_obs)

    if (in_gap == 0):
        reward -= 0.15
    
    if (in_gap == 1):
        reward -= 0.05

    if (in_gap == 2):
        reward += 0.2

    if (in_gap == 3):
        reward -= 0.05

    if (in_gap == 4):
        reward -= 0.15
    
    
    done = terminated or truncated

    next_state = state_extractor(next_obs)
    scored = int(new_info) - int(info)
    if (scored == 1):
        reward += 5

            

    agent.update(state, action, reward, next_state, done)
    total_episode_reward += reward 
    step_in_episode += 1
    
    obs = next_obs
    info = new_info

    
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
    
    
    
    #print(f"step: {step}")
    count % 50
    if (count == 1):
        print(f"action: {action}")
        print(f"reward: {reward}")
        print(f"Terminated: {terminated}")
        print(f"Score: {info}")
        print(f"Count: {count}")
    '''


    
    


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


        # Logging info for matplotlib 
        episode_rewards.append(total_episode_reward)
        episode_lengths.append(step_in_episode)
        epsilon_history.append(agent.epsilon)

        total_episode_reward = 0
        step_in_episode = 0
        #Human rendering mode to check progress
        if agent.num_episode % RENDER_EVERY == 0:
            
            print(f"This is the demo episode{agent.num_episode}")
            print(f"action: {action}")
            print(f"reward: {reward}")
            print(f"Terminated: {terminated}")
            print(f"Score: {info}")
            print(f"Count: {count}")
            
            run_demo_episode()


        if agent.num_episode >= TOTAL_EPISODES:
            break 

        obs, info = train_env.reset()
        total_episode_reward = 0
        step_in_episode = 0

        continue

    continue

train_env.close()
demo_env.close()
episodes = range(1, len(episode_rewards) + 1)

plt.figure(figsize=(15, 4))

# --- Total reward per episode ---
plt.subplot(1, 3, 1)
plt.plot(episodes, episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Episode Reward")

# --- Episode length ---
plt.subplot(1, 3, 2)
plt.plot(episodes, episode_lengths)
plt.xlabel("Episode")
plt.ylabel("Steps Survived")
plt.title("Episode Length")

# --- Epsilon ---
plt.subplot(1, 3, 3)
plt.plot(episodes, epsilon_history)
plt.xlabel("Episode")
plt.ylabel("Epsilon")
plt.title("Exploration Rate")

plt.tight_layout()
plt.show()
