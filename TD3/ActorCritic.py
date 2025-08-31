import torch.nn.functional as F 
import torch.nn as nn 
import torch

    

class Actor(nn.Module):
    def __init__(self,args,hidden_layers=[64,64]):
        super(Actor, self).__init__()
        self.num_states = args.num_states
        self.num_actions = args.num_actions
        self.action_max = args.action_max 
        # add in list
        hidden_layers.insert(0,self.num_states) # first layer
        hidden_layers.append(self.num_actions) # last layer
        print(hidden_layers)

        # create layers
        layer_list = []
        for i in range(len(hidden_layers)-1):
            input_num = hidden_layers[i]
            output_num = hidden_layers[i+1]
            layer = nn.Linear(input_num,output_num)
            
            layer_list.append(layer)

        # put in ModuleList
        self.layers = nn.ModuleList(layer_list)
        self.tanh = nn.Tanh()

    # when actor(s) will activate the function 
    def forward(self,s):

        for layer in self.layers:
            s = self.tanh(layer(s))
            
        return s * self.action_max


class Critic(nn.Module):
    def __init__(self, args,hidden_layers=[64,64]):
        super(Critic, self).__init__()
        self.num_states = args.num_states
        self.num_actions = args.num_actions
        # add in list
        hidden_layers.insert(0,self.num_states+self.num_actions)
        hidden_layers.append(1)
        print(hidden_layers)

        # create layers
        layer_list = []

        for i in range(len(hidden_layers)-1):
            input_num = hidden_layers[i]
            output_num = hidden_layers[i+1]
            layer = nn.Linear(input_num,output_num)
            
            layer_list.append(layer)
        # put in ModuleList
        self.layers = nn.ModuleList(layer_list)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self,s,a):
        input_data = torch.cat((s,a),dim=1)
        for i in range(len(self.layers)-1):
            input_data = self.relu(self.layers[i](input_data))

        # predicet value
        v_s = self.layers[-1](input_data)
        return v_s