import torch
from torch import nn
import numpy as np
from utils import *
import gymnasium as gym


class Agent:
    def __init__(self,env_name,num_envs):
        self.envs = gym.vector.make(env_name,num_envs=num_envs)
        
        ## convolution_net


    