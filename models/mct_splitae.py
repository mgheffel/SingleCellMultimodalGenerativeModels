import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


class MCGAE(nn.Module):
    def __init__(self, input_dim, input_dim_right, latent_dim):
        super(MCGAE, self).__init__()
        self.encoder1 = nn.Linear(input_dim, latent_dim)  # Encoder layer
        self.batch_norm1 = nn.BatchNorm1d(latent_dim)  # Batch normalization for encoder
        self.dropout = nn.Dropout(0.2)
        
        self.decoder1 = nn.Linear(latent_dim, input_dim)  # Decoder layer
        self.batch_norm2 = nn.BatchNorm1d(input_dim)  # Batch normalization for decoder
        
#         self.encoder_right1=nn.Linear(input_dim_right, latent_dim)
        
        self.decoder1_right=nn.Linear(latent_dim, input_dim_right)
        self.batch_norm2_right=nn.BatchNorm1d(input_dim_right)

    def encode(self, x):
        # Apply ReLU activation after batch normalization and dropout
        x = torch.relu(self.batch_norm1(self.encoder1(x)))  # Apply BatchNorm after the linear layer
        x = self.dropout(x)
        return x

    def decode(self, z):
        # Decode the latent representation back to input space
        z = self.batch_norm2(self.decoder1(z))  # Apply BatchNorm after the linear layer
        return z
    def decode_right(self, z):
        # Decode the latent representation back to input space
        z = self.batch_norm2_right(self.decoder1_right(z))  # Apply BatchNorm after the linear layer
        return z

    def forward(self, x):
        # Encode the input
        z = self.encode(x)
        # Decode the latent representation
        reconstructed = self.decode(z)
        recon_right=self.decode_right(z)
        return reconstructed, recon_right

    def predict(self, x):
        # Use the model for inference without computing gradients
        with torch.no_grad():
            z = self.encode(x)
            reconstructed= self.decode(z)
            recon_right = self.decode_right(z)
        return reconstructed, recon_right

def compute_loss(true, pred):
    """
    Compute the loss between true and predicted values.
    MSE (Mean Squared Error) is used here.
    """
    loss = torch.mean((true - pred)**2)
    return loss

def train_step(model, data_left,data_right, optimizer):
    """
    Perform one training step for the model.
    """
    model.train()  # Set the model to training mode
    optimizer.zero_grad()  # Clear previous gradients
    recon, rr = model(data_left)  # Forward pass
    loss_left = compute_loss(data_left, recon)  # Compute the loss
    loss_right = compute_loss(data_right, rr)  # Compute the loss
    loss=loss_left+loss_right
    loss.backward()  # Backpropagation
    optimizer.step()  # Update the weights
    return loss.item()