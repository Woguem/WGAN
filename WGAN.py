"""
@author: Dr Yen Fred WOGUEM 

@description: This script trains a WGAN model to generate image

"""



import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

start_time = datetime.now()  # Start timer



# Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
z_dim = 100
batch_size = 64
lr = 0.00005  
num_epochs = 20
img_size = 28
img_channels = 1
n_critic = 5  # Number of critic iterations per generator iteration
clip_value = 0.01  # Weight clipping value

# Data processing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Generator for WGAN
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, img_size * img_size),
        )
    
    def forward(self, z):
        return self.model(z).view(-1, img_channels, img_size, img_size)

# Critic 
class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(img_size * img_size, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
        )
    
    def forward(self, img):
        flattened = img.view(-1, img_size * img_size)
        return self.model(flattened)

# InitialiZation
G = Generator().to(device)
C = Critic().to(device)

# Optimizers (RMSprop as recommanded in WGAN paper)
optimizer_G = optim.RMSprop(G.parameters(), lr=lr)
optimizer_C = optim.RMSprop(C.parameters(), lr=lr)

# VisualiZation
def save_generated_images(epoch):
    with torch.no_grad():
        z = torch.randn(16, z_dim).to(device)
        samples = G(z).cpu()
        fig, axes = plt.subplots(4, 4, figsize=(8,8))
        for i, ax in enumerate(axes.flatten()):
            ax.imshow(samples[i].permute(1,2,0).numpy()*0.5+0.5, cmap='gray')
            ax.axis('off')
        plt.savefig(f'wgan_samples_epoch_{epoch}.png')
        plt.close()

# Training WGAN 
for epoch in range(num_epochs):
    for i, (real_imgs, _) in enumerate(dataloader):
        real_imgs = real_imgs.to(device)
        batch_size = real_imgs.size(0)
        
        # Training Critic (more than the generator)
        for _ in range(n_critic):
            optimizer_C.zero_grad()
            
            # Noise for generator
            z = torch.randn(batch_size, z_dim).to(device)
            fake_imgs = G(z).detach()
            
            # Scores
            C_real = C(real_imgs)
            C_fake = C(fake_imgs)
            
            # Loss of critic (Wasserstein distance)
            loss_C = -(torch.mean(C_real) - torch.mean(C_fake))
            loss_C.backward()
            optimizer_C.step()
            
            # Clipping Critic weights to meet the 1-Lipschitz condition
            for p in C.parameters():
                p.data.clamp_(-clip_value, clip_value)
        
        # TRaining Generator
        optimizer_G.zero_grad()
        
        z = torch.randn(batch_size, z_dim).to(device)
        fake_imgs = G(z)
        C_fake = C(fake_imgs)
        loss_G = -torch.mean(C_fake)  # We want to maximize the score
        
        loss_G.backward()
        optimizer_G.step()
        
    # Display
    print(f"Epoch [{epoch}/{num_epochs}] Loss C: {loss_C.item():.4f}, Loss G: {loss_G.item():.4f}")
    if epoch % 3 == 0:
        save_generated_images(epoch)

# Save model
torch.save(G.state_dict(), 'wgan_generator.pth')
torch.save(C.state_dict(), 'wgan_critic.pth')


end_time = datetime.now()  # End of timer
execution_time = end_time - start_time
print(f"\nDurée d'exécution : {execution_time}")







