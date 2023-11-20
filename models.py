import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, num_conv_layers=2, dropout_prob=0.5):
        super(SimpleCNN, self).__init__()
        self.num_conv_layers = num_conv_layers

        # First Convolutional Block
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=7, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        # Convolutional layers before the first pooling
        self.conv_before_pool1 = nn.ModuleList()
        self.bn_before_pool1 = nn.ModuleList()
        for _ in range(self.num_conv_layers):
            self.conv_before_pool1.append(nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1))
            self.bn_before_pool1.append(nn.BatchNorm2d(16))

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second Convolutional Block
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        # Convolutional layers after the first pooling
        self.conv_after_pool1 = nn.ModuleList()
        self.bn_after_pool1 = nn.ModuleList()
        for _ in range(self.num_conv_layers):
            self.conv_after_pool1.append(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1))
            self.bn_after_pool1.append(nn.BatchNorm2d(32))


        # Second Convolutional Block
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)

        # Third Convolutional Block
        self.bn3 = nn.BatchNorm2d(16)
        self.conv3_layers = nn.ModuleList()
        for _ in range(self.num_conv_layers):
            self.conv3_layers.append(nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1))

        self.conv4 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1)
        # Fourth Convolutional Block
        self.bn4 = nn.BatchNorm2d(8)
        self.conv4_layers = nn.ModuleList()
        for _ in range(self.num_conv_layers):
            self.conv4_layers.append(nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1))

        # Fully Connected Layers
        self.fc1 = nn.Linear(8 * 13 * 13, 512)
        self.fc2 = nn.Linear(512, num_classes)

        # Dropout Layers
        self.dropout1 = nn.Dropout(dropout_prob / 2)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        # Convolutional layers before the first pooling
        x = F.relu(self.dropout(self.bn1(self.conv1(x))))

        residual1 = x
        for i in range(self.num_conv_layers):
            x = F.relu(self.dropout1(self.conv_before_pool1[i](self.bn_before_pool1[i](x))))

        x = x + residual1
        x = F.relu(self.dropout1(self.bn2(self.conv2(x))))
        x = self.pool1(x)

        # Convolutional layers after the first pooling
        residual2 = x
        for i in range(self.num_conv_layers):
            x = F.relu(self.dropout(self.conv_after_pool1[i](self.bn_after_pool1[i](x))))
        x = x + residual2
        x = self.pool1(x)
        x = F.relu(self.dropout(self.conv3(self.bn2(x))))

        # Convolutional layers before the second pooling
        residual3 = x
        for i in range(self.num_conv_layers):
            x = F.relu(self.dropout(self.conv3_layers[i](self.bn3(x))))
        x = x + residual3
        x = self.pool1(x)
        x = F.relu(self.dropout(self.conv4(self.bn3(x))))

        # Convolutional layers before the third pooling
        residual4 = x
        for i in range(self.num_conv_layers):
            x = F.relu(self.dropout(self.conv4_layers[i](self.bn4(x))))
        x = x + residual4
        x = self.pool1(x)

        x = x.view(-1, 8 * 13 * 13)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def predict(self, x):
        out = self.forward(x)
        return torch.argmax(out), torch.max(F.softmax(out))