from train_utils import load_model
import argparse
from models import SimpleCNN, ResNetLight
import torch
from torchvision import models
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Trainer for ASLV Model')
parser.add_argument('--model_type', type=str, default='resnet_light', help='Name of the model')
parser.add_argument('--num_classes', type=int, default=29, help='Number of classes')
parser.add_argument('--checkpoint_name', type=str, default="best_model.pth", help='Model name')
args = parser.parse_args()

model = None
checkpoint_path = ""

if args.model_type == "Simple":
    checkpoint_path = "checkpoints/Simple/Simple"
    model = SimpleCNN(args.num_classes).to(device)
if args.model_type == "pretrained":
    print("Loading Pretrained")
    checkpoint_path = "checkpoints/pretrained"
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, args.num_classes)
if args.model_type == "resnet_light":
    print("loading resnet")
    checkpoint_path = "checkpoints/resnet_light"
    model = ResNetLight(args.num_classes).to(device)
    # Specify the number of classes in your dataset

checkpoint = torch.load(f"{checkpoint_path}/{args.checkpoint_name}")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  # Set the model to evaluation mode


epoch_past = checkpoint['epoch']
train_loss = checkpoint['train_loss']
best_accuracy = checkpoint['accuracy']
train_loss_history = checkpoint['train_loss_history']
train_accuracy_history = checkpoint['train_accuracy_history']
test_loss_history = checkpoint['test_loss_history']
test_accuracy_history = checkpoint['test_accuracy_history']
print(f'Best Iteration at {epoch_past}, Loss: {train_loss:.4f}, Accuracy: {best_accuracy:.2f}%')

# Plot training loss and test loss
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(train_loss_history, label='Training Loss', color='blue')
plt.plot(test_loss_history, label='Test Loss', color='orange')
plt.title('Training and Test Loss History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot training accuracy and test accuracy
plt.subplot(1, 2, 2)
plt.plot(train_accuracy_history, label='Training Accuracy', color='blue')
plt.plot(test_accuracy_history, label='Test Accuracy', color='orange')
plt.title('Training and Test Accuracy History')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()