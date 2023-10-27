import torch
import torch.nn as nn
import torch.optim as optim
# from einops.layers.torch import Rearrange
import math

import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, dropout_prob=0.5):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, 512)  # Adjust the input size based on your image dimensions
        self.fc2 = nn.Linear(512, num_classes)
        self.softmax = nn.Softmax(dim=1)  # Softmax activation for multiclass classification
        self.bn1 = nn.BatchNorm2d(16)  # BatchNorm after the first convolutional layer
        self.bn2 = nn.BatchNorm2d(32)  # BatchNorm after the second convolutional layer
        self.dropout = nn.Dropout(dropout_prob)  # Dropout after the first convolutional layer

    def forward(self, x):
        x = self.pool(F.relu(self.dropout(self.bn1(self.conv1(x)))))
        x = self.pool(F.relu(self.dropout(self.bn1(self.conv1(x)))))
        x = x.view(-1, 32 * 56 * 56)  # Adjust the input size based on your image dimensions
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    def predict(self, x):
        return torch.argmax(self.forward(x))
    
class VisionTransformer(nn.Module):
    def __init__(self, num_classes, image_size=224, patch_size=16, num_layers=12, embed_dim=768, num_heads=12, mlp_dim=100):
        super(VisionTransformer, self).__init__()
        assert image_size % patch_size == 0, "Image dimensions must be divisible by the patch size."
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2  # Input image has 3 channels (RGB)


        self.patch_embedding = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        # Positional Embedding
        self.positional_embedding = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.positional_encoding_init()

        encoder_layers = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=mlp_dim,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, num_classes)
    def positional_encoding_init(self):
        """Initialize the positional encoding."""
        max_len = self.positional_embedding.shape[1]
        dim = self.positional_embedding.shape[2]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim))
        pos_enc = torch.zeros(1, max_len, dim)
        pos_enc[:, :, 0::2] = torch.sin(position * div_term)
        pos_enc[:, :, 1::2] = torch.cos(position * div_term)
        
        # Create a new tensor without gradients and assign it to self.positional_embedding
        self.positional_embedding = nn.Parameter(pos_enc, requires_grad=True)


    def forward(self, x):
        x = self.patch_embedding(x)
        x = x.permute(0, 2, 3, 1)  # (batch_size, num_patches, dim)
        x = x.view(x.size(0), -1, x.size(-1))  # (batch_size, num_patches, dim)
        # Add positional embeddings
        x = x + self.positional_embedding

        # Transformer Encoder
        x = self.transformer_encoder(x)  # (batch_size, num_patches, dim)
        # Classification Head
        x = x.mean(dim=1)  # Global average pooling
        x = self.fc(x)  # (batch_size, num_classes)
        return x