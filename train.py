import torch
import torch.nn as nn
import torch.optim as optim
import os.path 
from models import SimpleCNN, ResNetLight
from train_utils import load_data, create_relative_directory
import argparse
from tqdm import tqdm
from torchvision import models
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.num_epoch = 0
        self.best_accuracy = 0
        self.epoch_past = 0
        self.train_loss_history = []
        self.train_accuracy_history = []
        self.test_loss_history = []
        self.test_accuracy_history = []
    def train_fn(self):
        self.model.train()
        train_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        for data, targets in tqdm(self.train_loader):
            data, targets = data.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            with torch.no_grad(): 
                predicted = torch.argmax(outputs, dim=1)
                correct_predictions += (predicted == targets).sum().item()
                total_samples += targets.size(0)

        loss = train_loss / len(self.train_loader)
        accuracy = (correct_predictions / total_samples) * 100
        return  loss, accuracy
       
    def train(self, num_epochs, checkpoint_path):
        self.num_epoch = num_epochs
        self.model.to(device)
        for epoch in range(num_epochs):
            train_loss, train_accuracy = self.train_fn()
            val_loss, val_accuracy = self.evaluate()

            #Store loss and accuracy
            self.train_loss_history.append(train_loss)
            self.train_accuracy_history.append(train_accuracy)
            self.test_loss_history.append(val_loss)
            self.test_accuracy_history.append(val_accuracy)
            # Save checkpoint after each epoch
            checkpoint_name = f"{train_accuracy}-{train_loss}.pth"
            self.save_checkpoint(epoch + 1, train_loss, train_accuracy, f"{checkpoint_path}/{checkpoint_name}")

            # Update best accuracy and save best model checkpoint
            if val_accuracy > self.best_accuracy:
                self.best_accuracy = val_accuracy
                self.save_checkpoint(epoch + 1, val_loss, val_accuracy,  f"{checkpoint_path}/best_model.pth")

            print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%')
            print(f'Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch_past = checkpoint['epoch']
        train_loss = checkpoint['train_loss']
        self.best_accuracy = checkpoint['accuracy']
        self.train_loss_history = checkpoint['train_loss_history']
        self.train_accuracy_history = checkpoint['train_accuracy_history']
        self.test_loss_history = checkpoint['test_loss_history']
        self.test_accuracy_history = checkpoint['test_accuracy_history']
        print(f'Checkpoint loaded. Resuming training from epoch {self.epoch_past}, Loss: {train_loss:.4f}, Accuracy: {self.best_accuracy:.2f}%')
        return self.epoch_past, train_loss, self.best_accuracy
    
    def evaluate(self):
        self.model.eval()
        val_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        all_targets = []
        all_predictions = []

        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(tqdm(self.val_loader)):
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == targets).sum().item()
                total_samples += targets.size(0)
                # Store targets and predictions for later evaluation
                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        average_loss = val_loss / len(self.val_loader)
        accuracy = (correct_predictions / total_samples) * 100
        precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_predictions, average='weighted')
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
        conf_matrix = confusion_matrix(all_targets, all_predictions)
        # Print or visualize the confusion matrix
        print(f'Validation Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%')
        return average_loss, accuracy, precision, recall, f1, conf_matrix

    def save_checkpoint(self, epoch, train_loss, accuracy, checkpoint_name):
        checkpoint = {
            'epoch': self.epoch_past + epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'accuracy': accuracy,
            'train_loss_history' : self.train_loss_history,
            'train_accuracy_history': self.train_accuracy_history,
            'test_loss_history': self.test_loss_history,
            'test_accuracy_history': self.test_accuracy_history
        }
        torch.save(checkpoint, checkpoint_name)
        print(f'Checkpoint saved at: {checkpoint_path}')


parser = argparse.ArgumentParser(description='Trainer for ASLV Model')

# Define command-line arguments
parser.add_argument('--num_classes', type=int, default=29, help='Number of classes')
parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--data_path', type=str, default='data/asl_alphabet_train/asl_alphabet_train', help='Path to the data')
parser.add_argument('--checkpoint_path', type=str, default='checkpoints', help='Path to save checkpoints')
parser.add_argument('--checkpoint_name', type=str, default='best_model.pth', help='Name of the checkpoint file')
parser.add_argument('--load_checkpoint', action='store_true', help='Load checkpoint if set', default=True)
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--train_ratio', type=float, default=0.80, help='Training data ratio')
parser.add_argument('--model_type', type=str, default='Simple', help='Name of the model')
parser.add_argument('--patch-size', type=int, default=16,
                    help='patch size for images (default : 16)')
parser.add_argument('--latent-size', type=int, default=1024,
                    help='latent size (default : 768)')
parser.add_argument('--num-encoders', type=int, default=12,
                    help='number of encoders (default : 12)')
parser.add_argument('--dropout', type=int, default=0.5,
                    help='dropout value (default : 0.1)')
parser.add_argument('--n-channels', type=int, default=3,
                    help='number of channels in images (default : 3 for RGB)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--num-heads', type=int, default=16,
                    help='(default : 16)')

# Parse command-line arguments
args = parser.parse_args()
# Access the arguments script
num_classes = args.num_classes
num_epochs = args.num_epochs
data_path = args.data_path
checkpoint_path = args.checkpoint_path
checkpoint_name = args.checkpoint_name
load_checkpoint = True
model_type = args.model_type
checkpoint_path = f"{checkpoint_path}/{model_type}/"
create_relative_directory(checkpoint_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using", device)


if model_type == "Simple":
    print("Training Simple")
    model = SimpleCNN(num_classes=num_classes).to(device)# Specify the number of classes in your dataset
if model_type == "pretrained":
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
if model_type == "resnet_light":
    print("Training ResNet Light")
    model = ResNetLight(num_classes).to(device)# Specify the number of classes in your dataset

train_loader, val_loader = load_data(data_path, batch_size=args.batch_size, train_ratio=args.train_ratio)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)
trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, device)

if load_checkpoint:
    if os.path.exists(f"{checkpoint_path }/{checkpoint_name}") and checkpoint_name :
        print(f"Loading Checkpoints: {checkpoint_path }/{checkpoint_name}")
        trained_epochs, trained_loss, trained_accuracy = trainer.load_checkpoint(f"{checkpoint_path }/{checkpoint_name}")

trainer.train(num_epochs, checkpoint_path)
average_loss, accuracy, precision, recall, f1, conf_matrix = trainer.evaluate()


# Visualize confusion matrix as heatmap
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()