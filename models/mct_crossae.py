import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


class CrossAE(nn.Module):
    def __init__(self, input_dim_left, input_dim_right, latent_dim):
        super(MCGAE, self).__init__()
        #data left encoding to latent
        self.encoder1_left = nn.Linear(input_dim_left, latent_dim)  # Encoder layer
        self.batch_norm1_left = nn.BatchNorm1d(latent_dim)  # Batch normalization for encoder
        self.dropout_left = nn.Dropout(0.2)
        #data right encoding to latent
        self.encoder1_right = nn.Linear(input_dim_right, latent_dim)  # Encoder layer
        self.batch_norm1_right = nn.BatchNorm1d(latent_dim)  # Batch normalization for encoder
        self.dropout_right = nn.Dropout(0.2)
        
        #decode latent to left
        self.decoder1_left = nn.Linear(latent_dim, input_dim_left)  # Decoder layer
        self.batch_norm2_left = nn.BatchNorm1d(input_dim_left)  # Batch normalization for decoder
        #decode latent to right
        self.decoder1_right=nn.Linear(latent_dim, input_dim_right)
        self.batch_norm2_right=nn.BatchNorm1d(input_dim_right)

    def encode_left(self, x):
        # Apply ReLU activation after batch normalization and dropout
        x = torch.relu(self.batch_norm1_left(self.encoder1_left(x)))  # Apply BatchNorm after the linear layer
        x = self.dropout_left(x)
        return x
    def encode_right(self, x):
        # Apply ReLU activation after batch normalization and dropout
        x = torch.relu(self.batch_norm1_right(self.encoder1_right(x)))  # Apply BatchNorm after the linear layer
        x = self.dropout_right(x)
        return x
    def decode_left(self, z):
        # Decode the latent representation back to input space
        z = self.batch_norm2_left(self.decoder1_left(z))  # Apply BatchNorm after the linear layer
        return z
    def decode_right(self, z):
        # Decode the latent representation back to input space
        z = self.batch_norm2_right(self.decoder1_right(z))  # Apply BatchNorm after the linear layer
        return z

    def forward_left_to_both(self, x):
        # Encode the input
        z = self.encode_left(x)
        # Decode the latent representation
        recon_left = self.decode_left(z)
        recon_right=self.decode_right(z)
        return recon_left, recon_right
    def forward_right_to_both(self, x):
        # Encode the input
        z = self.encode_right(x)
        # Decode the latent representation
        recon_left = self.decode_left(z)
        recon_right=self.decode_right(z)
        return recon_left, recon_right
#     def forward

    def predict_from_left(self, x):
        # Use the model for inference without computing gradients
        with torch.no_grad():
            z=torch.relu(self.batch_norm1_left(self.encoder1_left(x)))
            recon_left = self.decode_left(z)
            recon_right=self.decode_right(z)
            return recon_left, recon_right

    def predict_from_right(self, x):
        # Use the model for inference without computing gradients
        with torch.no_grad():
            z=torch.relu(self.batch_norm1_right(self.encoder1_right(x)))
            recon_left = self.decode_left(z)
            recon_right=self.decode_right(z)
            return recon_left, recon_right

    
    def sample_latent_from_left(self, x):
        # Use the model for inference without computing gradients
        with torch.no_grad():
            z=torch.relu(self.batch_norm1_left(self.encoder1_left(x)))
            return z

    def sample_latent_from_right(self, x):
        # Use the model for inference without computing gradients
        with torch.no_grad():
            z=torch.relu(self.batch_norm1_right(self.encoder1_right(x)))
            return z


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
    #get loss for left to right embeddings
    recon_left_ltr, recon_right_ltr = model.forward_left_to_both(data_left)  # Forward pass
    loss_left_ltr = compute_loss(data_left, recon_left_ltr)  # Compute the loss
    loss_right_ltr = compute_loss(data_right, recon_right_ltr)  # Compute the loss
    #get loss for right to left embeddings 
    recon_left_rtl, recon_right_rtl = model.forward_right_to_both(data_right)  # Forward pass
    loss_left_rtl = compute_loss(data_left, recon_left_rtl)  # Compute the loss
    loss_right_rtl = compute_loss(data_right, recon_right_rtl)  # Compute the loss
    loss=loss_left_ltr+loss_right_ltr+loss_left_rtl+loss_right_rtl
    loss.backward()  # Backpropagation
    optimizer.step()  # Update the weights
    return loss.item()