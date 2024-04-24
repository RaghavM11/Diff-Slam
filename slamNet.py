import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical, Normal, Independent, MixtureSameFamily

from coordconv import CoordConv2d as CoordConv

class TransitionModel(nn.Module):
    def __init__(self, inputShape, use_cuda):
        super(TransitionModel, self).__init__()
        channels, height, width = inputShape
        self.front = nn.Sequential(
            CoordConv(channels*3, 32, kernel_size=3, stride=1, padding=1, use_cuda=use_cuda),
            nn.LayerNorm([32, height, width]),
            nn.ReLU(),
        )
        self.convs = nn.ModuleList([
            CoordConv(32, 8, kernel_size=5, stride=1, dilation=4, padding=8, use_cuda=use_cuda),
            CoordConv(32, 8, kernel_size=5, stride=1, dilation=2, padding=4, use_cuda=use_cuda),
            CoordConv(32, 16, kernel_size=5, stride=1, dilation=1, padding=2, use_cuda=use_cuda),
            CoordConv(32, 32, kernel_size=3, stride=1, dilation=1, padding=1, use_cuda=use_cuda),
        ])
        self.body_first = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LayerNorm([64, height//2-1, width//2-1]),
            nn.ReLU(),
        )
        self.body_second = nn.Sequential(
            CoordConv(64, 128, kernel_size=3, stride=1, padding=1, use_cuda=use_cuda),
            nn.ReLU(),
            CoordConv(128, 64, kernel_size=3, stride=1, padding=1, use_cuda=use_cuda),
            nn.ReLU(),
        )
        self.body_third = nn.Sequential(
            CoordConv(64, 128, kernel_size=3, stride=1, padding=1, use_cuda=use_cuda),
            nn.ReLU(),
            CoordConv(128, 64, kernel_size=3, stride=1, padding=1, use_cuda=use_cuda),
            nn.ReLU(),
        )
        self.body_fourth = nn.Sequential(
            CoordConv(64, 64, kernel_size=4, stride=2, use_cuda=use_cuda),
            nn.ReLU(),
            CoordConv(64, 16, kernel_size=4, stride=2, use_cuda=use_cuda)
        )

    def forward(self, observation, observationPrev):
        diffObservation = observation - observationPrev

        concatObservations = torch.cat((observation, observationPrev, diffObservation), dim=1)

        x_front = self.front(concatObservations)
        x_conv = []
        for conv in self.convs:
            x_new = conv(x_front)
            x_conv.append(x_new)

        x_cat = torch.cat(x_conv, dim=1)
        #x_conv = torch.cat([conv(x) for conv in self.convs], dim=1)

        xi1 = self.body_first(x_cat)
        xi2 = xi1 + self.body_second(xi1)
        xi3 = xi2 + self.body_third(xi2)
        x_fourth = self.body_fourth(xi3)

        x_final = x_fourth.view(x_fourth.size(0), -1)

        return x_final

class GMModel(nn.Module):
    def __init__(self, num_features, k):
        super(GMModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU(),
        )
        self.mu = nn.Linear(128, k)
        self.sigma = nn.Linear(128, k)
        self.logvar = nn.Linear(128, k)
        

    def forward(self, x):
        xn = self.model(x)
        mu = self.mu(xn)
        sigma = self.sigma(xn)
        logvar = self.logvar(xn)
        return mu, sigma, logvar
       #return {'mu': mu, 'sigma': sigma, 'logvar': logvar}

class MappingModel(nn.Module):

    def __init__(self, N_ch, use_cuda):
        super(MappingModel, self).__init__()
        channels = 1 
        height = 80
        width = 80
        self.convs = nn.ModuleList([
            CoordConv(channels, 8, kernel_size=5, stride=1, dilation=4, padding=8, use_cuda=use_cuda),
            CoordConv(channels, 8, kernel_size=5, stride=1, dilation=2, padding=4, use_cuda=use_cuda),
            CoordConv(channels, 16, kernel_size=5, stride=1, dilation=1, padding=2, use_cuda=use_cuda),
            CoordConv(channels, 32, kernel_size=3, stride=1, dilation=1, padding=1, use_cuda=use_cuda),
        ])
        self.body_first = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.LayerNorm([64, height//2, width//2]),
            nn.ReLU(),
        )
        self.body_second = nn.Sequential(
            CoordConv(64, 32, kernel_size=3, stride=1, padding=1, use_cuda=use_cuda),
            nn.LayerNorm([32, height//2, width//2]),
            nn.ReLU(),
            CoordConv(32, 64, kernel_size=3, stride=1, padding=1, use_cuda=use_cuda),
            nn.LayerNorm([64, height//2, width//2]),
            nn.ReLU(),
        )
        self.body_third = nn.Sequential(
            CoordConv(64, N_ch, kernel_size=3, stride=1, padding=1, use_cuda=use_cuda)
        )

    # Perspective shape is (1, 80, 80)
    @staticmethod
    def try_perspective_transform(observation):
        # Perspective transform is done using torch to aid in backpropagation
        # The perspective transform is done on the CPU

        print("The shape of the observation is: ", observation.shape)
    
        # convert rgb to grayscale
        # Using matplotlib formula
        observation = observation[:, 0, :, :] * 0.2989 + observation[:, 1, :, :] * 0.5870 + observation[:, 2, :, :] * 0.1140
        observation = observation.unsqueeze(1)

        print("The shape of observation after conversion is:", observation.shape)
        fx, fy = 7.188560000000e+02, 7.188560000000e+02
        cx, cy = 6.071928000000e+02, 1.852157000000e+02

        # The intrinsic matrix
        K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32)

        # The extrinsic matrix
        # The rotation matrix
        R = torch.tensor([[9.999239000000e-01, 9.837760000000e-03, -7.445048000000e-03], [-9.869795000000e-03, 9.999421000000e-01, -4.278459000000e-03], [7.402527000000e-03, 4.351614000000e-03, 9.999631000000e-01]], dtype=torch.float32)
        # The translation matrix
        T = torch.tensor([-2.573699000000e-02, -1.199354000000e-01, 1.194591000000e-01], dtype=torch.float32)

        # The extrinsic matrix
        RT = torch.cat((R, T.view(3, 1)), dim=1)

        print("The shape of the RT matrix is: ", RT.shape)

        # The full projection matrix
        P = torch.matmul(K, RT)

        # Perspective transform
        perspective_transform = torch.nn.functional.affine_grid(P.unsqueeze(0), observation.unsqueeze(0).size())
        transformed_image = torch.nn.functional.grid_sample(observation.unsqueeze(0), perspective_transform)

        # Change to the required shape
        #perspective_transform = perspective_transform.permute(0, 3, 1, 2)
        print(perspective_transform.shape)

        return perspective_transform

    def forward(self, observation):
        x = self.try_perspective_transform(observation)
        x = torch.cat([conv(x) for conv in self.convs], dim=1)
        xi = self.body_first(x)
        xi += self.body_second(xi)
        x = self.body_third(xi)
        return x

class ObservationModel(nn.Module):

    def __init__(self, use_cuda):
        super(ObservationModel, self).__init__()
        self.body_first = nn.Sequential(
            CoordConv(2, 64, kernel_size=5, stride=1, use_cuda=use_cuda),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LayerNorm(),
            nn.ReLU(),
            CoordConv(64, 32, kernel_size=3, stride=1, use_cuda=use_cuda),
        )
        self.body_second = nn.ModuleList([
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.AvgPool2d(kernel_size=3, stride=2),
        ])
        self.body_third = nn.Sequential(
            nn.LayerNorm(),
            nn.ReLU(),
        )
        self.body_fourth = nn.ModuleList([
            nn.MaxPool2d(kernel_size=5, stride=5),
            nn.AvgPool2d(kernel_size=5, stride=5),
        ])
        self.body_fifth = nn.Sequential(
            nn.Linear(32*16*16, 1)
        )

    def forward(self, present_map, map_stored, particle_present, particles_stored):
        combined = self.transform(map_stored, particle_present, particles_stored)
        x = torch.cat([present_map, combined], dim=1)
        x = self.body_first(x)
        x = torch.cat([pool(x) for pool in self.body_second], dim=1)
        x = self.body_third(x)
        x = torch.cat([pool(x) for pool in self.body_fourth], dim=1)
        x = x.view(x.size(0), -1)
        x = self.body_fifth(x)
        return x

    @staticmethod
    def transform(map_stored, particle_present, particles_stored):
        pass


# NOTE: In general the model is in testing mode
"""
Apart from that the model can be in one of the following modes:
1. Training mode
2. Pretraining mode -- Transition model
3. Pretraining mode -- Observation and mapping model
"""
class SlamNet(nn.Module):
    # TODO: This function needs to be updated with the training or testing mode parameter
    def __init__(self, inputShape, K, is_training=False, is_pretrain_trans=False, is_pretrain_obs=False,
                 use_cuda=True):
        super(SlamNet, self).__init__()
        # NOTE: The current implementation assumes that the states are always kept with the weight
        self.bs = inputShape[0]
        self.lastStates = [0,0,0]
        self.lastWeights = 1
        self.K = K
        self.trajectory_estimate = [[0, 0, 0]]

        self.is_training = is_training
        self.is_pretrain_trans = is_pretrain_trans
        self.is_pretrain_obs = is_pretrain_obs

        assert(len(inputShape) == 4)
        if self.is_training or self.is_pretrain_obs:
            self.mapping = MappingModel(N_ch=16, use_cuda=use_cuda)
        if self.is_training or self.is_pretrain_trans:
            self.visualTorso = TransitionModel(inputShape[1:], use_cuda=use_cuda)
        if inputShape[1] == 3:
            numFeatures = 2592
        else:
            numFeatures = 2592
        self.gmmX = GMModel(numFeatures, 3)
        self.gmmY = GMModel(numFeatures, 3)
        self.gmmYaw = GMModel(numFeatures, 3)

    def forward(self, observation, observationPrev):
        if self.is_training or self.is_pretrain_obs:
            # Split the observation as rgb or depth data
            rgb_image = observation[:, :3, :, :]
            depth_image = observation[:, 3, :, :]
            print('The shape of rgb_image is: ', rgb_image.shape)
            print('The shape of the depth image is:', depth_image.shape)
            map_t = self.mapping(observation)

        if self.is_training or self.is_pretrain_trans:
            featureVisual = self.visualTorso(observation, observationPrev)
            x = self.gmmX(featureVisual)
            y = self.gmmY(featureVisual)
            yaw = self.gmmYaw(featureVisual)

            if self.is_pretrain_trans:
                new_states, new_weights = self.tryNewState(x, y , yaw)

        #new_states, new_weights = self.resample(new_states, new_weights)
        #print(new_states.shape, new_weights.shape)

        # Calculate the resultant pose estimate
        pose_estimate = self.calc_average_trajectory(new_states, new_weights)

        # TODO: Can return loss instead -- whatever is required for backward pass
        return pose_estimate


    # find the huber loss between the estimated pose and the ground truth pose
    # The value of delta can be altered
    @staticmethod
    def huber_loss(pose_estimated, actual_pose, delta = 0.1):
        residual = torch.abs(pose_estimated - actual_pose)
        is_small_res = residual < delta
        return torch.where(is_small_res, 0.5 * residual ** 2, delta * (residual - 0.5 * delta))

    # Resample the particles based on the weights
    # NOTE: Paper does not mention if the resampling is hard or soft and hence we use soft to avaoid zero gradient
    # NOTE: This function is a PyTorch version of the PFNet implementation
    @staticmethod
    def resample(particle_states, particle_weights, alpha=torch.tensor([0])):
        batch_size, num_particles = particle_states.shape[:2]

        # normalize
        particle_weights = particle_weights - torch.logsumexp(particle_weights, dim=-1, keepdim=True)

        uniform_weights = torch.full((batch_size, num_particles), -torch.log(torch.tensor(num_particles)), dtype=torch.float32)

        # build sampling distribution, q(s), and update particle weights
        if alpha < 1.0:
            # soft resampling
            q_weights = torch.stack([particle_weights + torch.log(alpha), uniform_weights + torch.log(1.0-alpha)], dim=-1)
            q_weights = torch.logsumexp(q_weights, dim=-1, keepdim=False)
            q_weights = q_weights - torch.logsumexp(q_weights, dim=-1, keepdim=True)  # normalized

            particle_weights = particle_weights - q_weights  # this is unnormalized
        else:
            # hard resampling. this will produce zero gradients
            q_weights = particle_weights
            particle_weights = uniform_weights

        # sample particle indices according to q(s)
        indices = torch.multinomial(torch.exp(q_weights), num_particles, replacement=True)  # shape: (batch_size, num_particles)

        # index into particles
        helper = torch.arange(0, batch_size*num_particles, step=num_particles, dtype=torch.int64)  # (batch, )
        indices = indices + helper.view(batch_size, 1).to(indices.device)

        particle_states = particle_states.view(batch_size * num_particles, 3)
        particle_states = particle_states.index_select(0, indices.view(-1)).view(batch_size, num_particles, 3)

        particle_weights = particle_weights.view(batch_size * num_particles)
        particle_weights = particle_weights.index_select(0, indices.view(-1)).view(batch_size, num_particles)

        return particle_states, particle_weights

    def tryNewState(self, x, y, yaw):
        x_mu, x_sigma, x_logvar = x
        y_mu, y_sigma, y_logvar = y
        yaw_mu, yaw_sigma, yaw_logvar = yaw

        mean_val = torch.cat([x_mu, y_mu, yaw_mu], dim=0)
        std_values = torch.cat([x_sigma , y_sigma, yaw_sigma], dim=0)
        # Mean wise multiplication of logvar
        logvar = []
        for i in range(3):
            new_value = x_logvar[:, i] * y_logvar[:, i] * yaw_logvar[:, i]
            logvar.append(new_value)
        logvar = torch.cat(logvar, dim=0)

        std_values = torch.nn.functional.softplus(std_values)
        logvar = torch.nn.functional.softplus(logvar)

        distributions = Independent(Normal(mean_val, std_values), 1)
        mixture_dist = Categorical(logvar)

        gmm_dist = MixtureSameFamily(mixture_dist, distributions) 

        samples = gmm_dist.sample(torch.Size([self.bs, self.K]))

        log_probs = gmm_dist.log_prob(samples)
        weights = torch.exp(log_probs - torch.max(log_probs))
        weights = weights / torch.sum(weights)

        return samples, weights

    def calc_average_trajectory(self, new_states, new_weights):
        # Calculate the average trajectory
        pose_estimate = torch.zeros(self.bs, 3, device=new_states.device)
        for i in range(self.K):
            pose_estimate[:, 0] = pose_estimate[:,0] + new_states[:, i, 0] * new_weights[:, i]
            pose_estimate[:, 1] = pose_estimate[:,1] + new_states[:, i, 1] * new_weights[:, i]
            pose_estimate[:, 2] = pose_estimate[:,2] + new_states[:, i, 2] * new_weights[:, i]
        #self.trajectory_estimate.append(pose_estimate)
        return pose_estimate

