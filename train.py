import torch
import torch.nn as nn
import torch.optim as optim
import os.path 
from models import SimpleCNN, VisionTransformer
from train_utils import load_data 
from os_util import create_relative_directory
import argparse

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train(self, num_epochs, checkpoint_path, train_accuracy=0):
        best_accuracy = train_accuracy
        self.model.to(device)
        self.model.train()
        for epoch in range(num_epochs):
            train_loss = 0.0
            correct_predictions = 0
            total_samples = 0
            for batch_idx, (data, targets) in enumerate(self.train_loader):
                data, targets = data.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()

                # with torch.cuda.amp.autocast():
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == targets).sum().item()
                total_samples += targets.size(0)

            average_loss = train_loss / len(self.train_loader)
            accuracy = (correct_predictions / total_samples) * 100

            # Save checkpoint after each epoch
            checkpoint_name = f"{accuracy}-{average_loss}.pth"
            self.save_checkpoint(epoch + 1, average_loss, accuracy, f"{checkpoint_path}/{checkpoint_name}")

            val_loss, val_accuracy = self.evaluate()
            # Update best accuracy and save best model checkpoint
            if accuracy > best_accuracy:
                best_accuracy = val_accuracy
                self.save_checkpoint(epoch + 1, average_loss, accuracy,  f"{checkpoint_path}/best_model.pth")

            print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%')

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        train_loss = checkpoint['train_loss']
        accuracy = checkpoint['accuracy']
        print(f'Checkpoint loaded. Resuming training from epoch {epoch}, Loss: {train_loss:.4f}, Accuracy: {accuracy:.2f}%')
        return epoch, train_loss, accuracy
    
    def evaluate(self):
        self.model.eval()
        val_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for data, targets in self.val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == targets).sum().item()
                total_samples += targets.size(0)

        average_loss = val_loss / len(self.val_loader)
        accuracy = (correct_predictions / total_samples) * 100

        print(f'Validation Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%')
        return average_loss, accuracy

    def save_checkpoint(self, epoch, train_loss, accuracy, checkpoint_name):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'accuracy': accuracy
        }
        torch.save(checkpoint, checkpoint_name)
        print(f'Checkpoint saved at: {checkpoint_path}')


parser = argparse.ArgumentParser(description='Your script description here.')

# Define command-line arguments
parser.add_argument('--num_classes', type=int, default=29, help='Number of classes')
parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--data_path', type=str, default='data/asl_alphabet_train/asl_alphabet_train', help='Path to the data')
parser.add_argument('--checkpoint_path', type=str, default='checkpoints', help='Path to save checkpoints')
parser.add_argument('--checkpoint_name', type=str, default='best_model.pth', help='Name of the checkpoint file')
parser.add_argument('--load_checkpoint', action='store_true', help='Load checkpoint if set')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--train_ratio', type=float, default=0.80, help='Training data ratio')
parser.add_argument('--model_type', type=str, default='Simple', help='Name of the model')


# Parse command-line arguments
args = parser.parse_args()
# Access the arguments in your script
num_classes = args.num_classes
num_epochs = args.num_epochs
lr = args.lr
data_path = args.data_path
checkpoint_path = args.checkpoint_path
checkpoint_name = args.checkpoint_name
load_checkpoint = args.load_checkpoint
batch_size = args.batch_size
train_ratio = args.train_ratio
model_type = args.model_type
checkpoint_path = f"{checkpoint_path}/{model_type}/"
create_relative_directory(checkpoint_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using", device)


if model_type == "Simple":
    print("Training Simple")
    model = SimpleCNN(num_classes=num_classes).to(device)# Specify the number of classes in your dataset
elif model_type == "Vision":
    print("Training Vision")
    model = VisionTransformer(num_classes).to(device)# Specify the number of classes in your dataset

train_loader, val_loader = load_data(data_path, batch_size=batch_size, train_ratio=train_ratio, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, device)

if load_checkpoint:
    if os.path.exists(f"{checkpoint_path }/{checkpoint_name}") and checkpoint_name :
        print(f"Loading Checkpoints: {checkpoint_path }/{checkpoint_name}")
        trained_epochs, trained_loss, trained_accuracy = trainer.load_checkpoint(f"{checkpoint_path }/{checkpoint_name}")

trainer.train(num_epochs, checkpoint_path)
trainer.evaluate()