from collections import deque 
import numpy as np 
import torch




class ReplayBuffer:
    def __init__(self, args):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mini_batch_size = args.mini_batch_size
        self.max_length = args.buffer_size
        self.size = 0       
        self.ptr = 0 
        self.s = np.zeros((self.max_length, args.num_states))
        self.a = np.zeros((self.max_length, args.num_actions))
        self.r = np.zeros((self.max_length, 1))
        self.s_ = np.zeros((self.max_length, args.num_states))
        self.done = np.zeros((self.max_length, 1))

    def store(self, s, a, r, s_, done):
        
        self.s[self.ptr] = s
        self.a[self.ptr] = a
        self.s_[self.ptr] = s_
        self.r[self.ptr] = r
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_length
        self.size = min(self.size + 1, self.max_length)
  
    def sample_minibatch(self):
        index = np.random.choice(self.size , self.mini_batch_size , replace=False)
        s = torch.tensor(self.s[index], dtype=torch.float).to(self.device)
        a = torch.tensor(self.a[index], dtype=torch.float).to(self.device)
        r = torch.tensor(self.r[index], dtype=torch.float).to(self.device)
        s_ = torch.tensor(self.s_[index], dtype=torch.float).to(self.device)
        done = torch.tensor(self.done[index], dtype=torch.float).to(self.device)
        return s, a, r, s_, done

