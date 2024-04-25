import torch
from torch.distributions import MultivariateNormal


# Define the mean, covariance, and mixture weights of a GMM
batch_size = 16
num_components = 3
mean = torch.randn(batch_size, num_components, 3)


covariance = torch.stack([torch.eye(3) for _ in range(num_components)])
#print(covariance)
mixture_weights = torch.tensor([0.3, 0.5, 0.2])

# Generate particles from the GMM
num_particles = 100
component_samples = torch.multinomial(mixture_weights, num_particles, replacement=True)
print(component_samples)

particles = []
for component in component_samples:
    mean_component = mean[component]
    #print(mean_component)
    covariance_component = covariance[component]
    #print(covariance_component)
    particle = MultivariateNormal(mean_component, covariance_component).sample()
    particles.append(particle[component])

particles = torch.stack(particles)

print("Particles:", particles.shape)