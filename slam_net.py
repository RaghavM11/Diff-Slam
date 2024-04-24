# This is the architecture of the SLAM network
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

from coordconv import CoordConv

class SlamNet(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass
    
    # This is the visual torso of the transition model
    def mapping_model_latent_local(self, observation_t, observation_t1):
        """
        Take the observations at time t and t-1 as the input and
        output the latent local map.
        """
        diff_observation = observation_t - observation_t1
        
        # Concatenate all the observations as the input
        concat_observations = torch.cat((observation_t, observation_t1, diff_observation), dim=1)
        print("The shape of the concatenated observations is: ", concat_observations.shape)
        _, _, c = concat_observations.shape

        # This layer might have to be changed
        x = CoordConv(c, 32, kernel_size=3, stride=1, padding=1)(concat_observations)
        x = nn.LayerNorm(x)
        x = F.relu(x)

        convs = [
            CoordConv(32, 8, kernel_size=5, stride=1, padding=4)(x),
            CoordConv(32, 8, kernel_size=5, stride=1, padding=2)(x),
            CoordConv(32, 32, kernel_size=5, stride=1, padding=1)(x),
            CoordConv(32, 32, kernel_size=3, stride=1, padding=1)(x)
        ]

        # Concatenate the layers of the convolutions
        x = torch.cat([conv(x) for conv in convs], dim=1)
        x = nn.MaxPool2d(kernel_size=3, stride=2)(x)
        x = nn.LayerNorm(x)
        x = F.relu(x)
        x_store = x.copy() # This is the first storage point
        
        x = CoordConv(128, 128, kernel_size=3, stride=1)(x)
        x = F.relu(x)
        x = CoordConv(128, 64, kernel_size=3, stride=1)(x)
        x = F.relu(x)

        # Add the first storage point to the current point
        x += x_store
        x_store2 = x.copy() # This is the second storage point

        x = CoordConv(128, 128, kernel_size=3, stride=1)(x)
        x = F.relu(x)
        x = CoordConv(128, 64, kernel_size=3, stride=1)(x)
        x = F.relu(x)

        # Add the second storage point to the current point
        x += x_store2

        x = CoordConv(128, 64, kernel_size=4, stride=2)(x)
        x = F.relu(x)
        x = CoordConv(64, 16, kernel_size=4, stride=2)(x)
        x = F.relu(x)
        # Flatten the output
        x = x.view(x.size(0), -1)  

        print("The shape of the output of the mapping model is: ", x.shape)
        return x

    # This is the GMM head of the model
    # TODO: The input must get the valu of 'k'
    def mapping_model_gmm(self, latent_local_map, k):
        """
        Take the latent map as the input and the mean, covariane and mixture log probabilities
        of the GMM as the output.
        """
        x = latent_local_map
        x = nn.Linear(16*16*16, 1024)(x)
        x = F.relu(x)
        x = nn.Linear(1024, 128)(x)
        x = F.relu(x)

        dense_layer = [
            nn.Linear(128, k)(x),
            nn.Linear(128, k)(x),
            nn.Linear(128, k)(x),
        ]

        return dense_layer
    
    # TODO: Make a perspective transformation -- information in the paper
    def perspective_transform(self, x):
        return x

    # Mapping model
    def mapping_model(self, observation_t, N_ch):
        x= observation_t
        x = self.perspective_transform(x)

        convs = [
            CoordConv(3, 8, kernel_size=5, stride=1, padding=4)(x),
            CoordConv(3, 8, kernel_size=5, stride=1, padding=2)(x),
            CoordConv(3, 16, kernel_size=5, stride=1, padding=1)(x),
            CoordConv(3, 32, kernel_size=3, stride=1, padding=1)(x)
        ]

        x = torch.cat([conv(x) for conv in convs], dim=1)
        x = nn.MaxPool2d(kernel_size=3, stride=2)(x)
        x = nn.LayerNorm(x)
        x = F.relu(x)
        
        x_store = x.copy() # This is the first storage point
        x = CoordConv(64, 32, kernel_size=3, stride=1)(x)
        x = nn.LayerNorm(x)
        x = F.relu(x)
        x = CoordConv(32, 64, kernel_size=3, stride=1)(x)
        x = nn.LayerNorm(x)
        x = F.relu(x)

        # Add the first storage point to the current point
        x += x_store
        x = CoordConv(64, N_ch, kernel_size=3, stride=1)(x)

        return x
    
    def transform(self, map_stored, particle_present, particles_stored):
        """
        This must call the spatial transformer network
        """
        pass


    # This is the observation model
    def observation_model(self, present_map, map_stored, particle_present, particles_stored):
        m_combined = self.transform(map_stored, particle_present, particles_stored)

        # Concatenate the present map and the combined map
        x = torch.cat((present_map, m_combined), dim=1)
        x = CoordConv(2, 64, kernel_size=5, stride=1)(x)
        x = nn.Maxpool2d(kernel_size=3, stride=2)(x)
        x = nn.LayerNorm(x)
        x = F.relu(x)
        x = CoordConv(64, 32, kernel_size=3, stride=1)(x)
        
        pools =[
            nn.Maxpool2d(kernel_size=3, stride=2)(x),
            nn.AvgPool2d(kernel_size=3, stride=2)(x)
        ]

        x = torch.cat([pool(x) for pool in pools], dim=1)
        x = nn.LayerNorm(x)
        x = F.relu(x)

        pools = [
            nn.Maxpool2d(kernel_size=5, stride=5)(x),
            nn.AvgPool2d(kernel_size=5, stride=5)(x)
        ]

        # Flatten the output
        x = torch.cat([pool(x) for pool in pools], dim=1)
        x = x.view(x.size(0), -1)
        
        x = nn.Linear(32*16*16, 1)(x)

        return x