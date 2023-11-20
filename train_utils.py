from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def save_images_to_pdf(dataloader, pdf_filename):
    """
    Utility function for visualizing and saving two images from each class in a dataloader to a PDF file.

    Args:
        dataloader (torch.utils.data.DataLoader): Dataloader containing the dataset.
        pdf_filename (str): Filename for the output PDF file.

    Returns:
        None
    """
    # Access the classes from the original dataset, not the Subset
    classes = dataloader.dataset.dataset.classes
    # Calculate the number of rows and columns for the subplot grid
    num_rows = len(classes)
    num_cols = 2
    with PdfPages(pdf_filename) as pdf_pages:
        # Loop through classes
        for class_idx in range(len(classes)):
            # Create a figure and axis for plotting
            fig, axs = plt.subplots(1, num_cols, figsize=(10, 4))
            class_counter = 0

            # Loop through batches in the dataloader
            for batch in dataloader:
                images, labels = batch

                # Loop through each sample in the batch
                for i in range(len(labels)):
                    current_class_idx = labels[i].item()

                    # Check if the current image is from the desired class
                    if current_class_idx == class_idx:
                        img = images[i].numpy()

                        # Plot the image for the current class
                        row_index = class_counter // num_cols
                        axs[class_counter].imshow(img.transpose(1, 2, 0))
                        axs[class_counter].set_title(f'{classes[class_idx]} - Image {class_counter + 1}')
                        axs[class_counter].axis('off')
                        class_counter += 1

                        # Check if we have collected two images for the current class
                        if class_counter >= 2:
                            break

                # Early stopping condition: Check if we have collected two images for the current class
                if class_counter >= 2:
                    break

            # Save the figure to the PdfPages
            plt.tight_layout()
            pdf_pages.savefig(fig)
            plt.close(fig)

def load_data(data_path, batch_size=32, train_ratio=0.8):
    """
    Utility function for loading and preparing image data for training and testing.

    Args:
        data_path (str): Path to the root directory of the image dataset.
        batch_size (int, optional): Number of samples per batch. Default is 32.
        train_ratio (float, optional): Ratio of the dataset used for training. Default is 0.8.
    Returns:
        torch.utils.data.DataLoader: DataLoader for the training set.
        torch.utils.data.DataLoader: DataLoader for the testing set.
    """
    data_augment =  transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.RandomRotation(degrees=(0, 30)),
    transforms.ColorJitter(brightness=0.5, contrast=0.2, saturation=0.2, hue=0.3),
    transforms.RandomAdjustSharpness(sharpness_factor=2),
    ])
    # Load data using ImageFolder
    dataset = datasets.ImageFolder(root=data_path, transform=data_augment)

    # Split the dataset into training and testing subsets
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create data loaders for training and testing
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # # Save images to PDF for visualization
    # save_images_to_pdf(test_loader, 'train_images.pdf')
    return train_loader, test_loader

def dump_labels(dataset):
    """
    Utility function for dumping class-to-index mapping to a JSON file.

    Args:
        dataset (torchvision.datasets.ImageFolder): Dataset containing class information.

    Returns:
        None
    """
    class_to_idx = dataset.class_to_idx
    idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}
    with open('idx_to_class.json', 'w') as json_file:
        json.dump(idx_to_class, json_file)