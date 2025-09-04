
import gymnasium as gym # openai gym
import numpy as np 
import argparse
from TD3.Agent import Agent
from gymnasium.wrappers import RecordVideo



class main():
    def __init__(self,args):
        env_name = 'Walker2d-v5'
        env = gym.make(env_name)
        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.shape[0]
        print(num_actions)
        print(num_states)
        
        # args
        args.num_actions = num_actions
        args.num_states = num_states
        args.action_max = env.action_space.high[0]  # Pendulum action space is continuous, so we need to normalize it


        # print args 
        print("---------------")
        for arg in vars(args):
            print(arg,"=",getattr(args, arg))
        print("---------------")

        # create agent
        hidden_layer_num_list = [400,300]
        agent = Agent(args , env , hidden_layer_num_list)

        # trainning
        agent.train() 

        # evaluate 
        render_env = gym.make(env_name, render_mode="rgb_array")  
        render_env = RecordVideo(render_env, video_folder = "Video/"+env_name, episode_trigger=lambda x: True)
        self.agent.evaluate(render_env)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for DDPG")
    parser.add_argument("--d", type=int, default=2, help="Update target network every d step")
    parser.add_argument("--c", type=float, default=0.5, help="Clip range for target policy smoothing")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate of actor")
    parser.add_argument("--var", type=float, default=0.1, help="Normal noise sigma for choose action")
    parser.add_argument("--tau", type=float, default=0.005, help="Parameter for soft update")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--sigma", type=float, default=0.2, help="Sigma for target action noise")
    parser.add_argument("--mem_min", type=float, default=100, help="minimum size of replay memory before updating actor-critic.")

    parser.add_argument("--mini_batch_size", type=int, default=100, help="Mini-Batch size")
    parser.add_argument("--buffer_size", type=int, default=int(10e6), help="Learning rate of actor")
    parser.add_argument("--max_train_steps", type=int, default=int(1e6), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq_steps", type=float, default=5e3, help="Evaluate the policy every 'evaluate_freq_steps' steps")
    args = parser.parse_args()

    main(args)