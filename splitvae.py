import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

class VAE(nn.Module):
    def __init__(self, left_input_dim, hidden_dim,hidden_dim2, latent_dim,right_output_dim):
        super(VAE, self).__init__()
        self.encoder1=nn.Linear(left_input_dim,hidden_dim)
        self.fc_mean=nn.Linear(hidden_dim,latent_dim)
        self.fc_logvar=nn.Linear(hidden_dim,latent_dim)
        self.left_decoder1=nn.Linear(latent_dim,hidden_dim)
        self.left_decoder2=nn.Linear(hidden_dim,left_input_dim)
        self.right_decoder1=nn.Linear(latent_dim,hidden_dim)
        self.right_decoder2=nn.Linear(hidden_dim,hidden_dim2)
        self.right_decoder3=nn.Linear(hidden_dim2,right_output_dim)
    def encode(self,x):
        x=torch.relu(self.encoder1(x))
        mean=self.fc_mean(x)
        logvar=self.fc_logvar(x)
        return mean,logvar
    def reparameterize(self,mean,logvar):
        std=torch.exp(0.5*logvar)
        eps=torch.randn_like(std)
        return mean+std*eps
    def decode_left(self,z):
        z=torch.relu(self.left_decoder1(z))
        return self.left_decoder2(z)
    def decode_right(self,z):
        z=torch.relu(self.right_decoder1(z))
        z=torch.relu(self.right_decoder2(z))
        z=torch.sigmoid(self.right_decoder3(z))

        return z
            #floor
#         return (z>0.5).float()
    def forward(self,x):
        mean,logvar=self.encode(x)
        z=self.reparameterize(mean,logvar)
        left_decoded=self.decode_left(z)
        right_decoded=self.decode_right(z)
        return left_decoded,right_decoded,mean,logvar
def recon_kld_loss(ytrue_left,ypred_left,ytrue_right,ypred_right,mean,logvar):
    recon_left = torch.mean((ytrue_left-ypred_left)**2)
    recon_right = torch.mean((ytrue_right-ypred_right)**2)
#     bce_loss_fn=nn.BCELoss()
#     recon_right=bce_loss_fn(ytrue_right.float(),ypred_right.float())
    kld=-.5*torch.sum(1+logvar-mean.pow(2)-logvar.exp())
    return recon_left*100+recon_right*100+kld/1000
def train_step(model,data_left,data_right,optimizer):
    optimizer.zero_grad()
    recon_left,recon_right,mean,logvar=model(data_left)
    loss=recon_kld_loss(data_left,recon_left,data_right,recon_right,mean,logvar)
    loss.backward()
    optimizer.step()
    return loss.item()