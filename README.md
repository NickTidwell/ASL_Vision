# ASL_Vision
models.py: This file defines the neural network architectures used for gesture recognition. We use a convolutional neural network (CNN)
train.py: This file trains the models on a dataset of ASL videos and saves the best model weights. We use PyTorch as the deep learning framework and torchvision for data loading and preprocessing.
video.py: This file loads a pretrained model and performs real-time gesture recognition from webcam input. We use OpenCV for video capture and display.
train_utils.py: This file contains some utility functions for training, such as calculating accuracy, saving and loading checkpoints, and plotting learning curves.
idx_to_class.json: This file maps the numerical labels to the corresponding ASL gestures. We use 29 classes, representing the 26 letters of the alphabet and three special symbols: space, delete, and nothing.
report.tex: This file is the LaTeX source code for the project report, which describes the motivation, methodology, results, and discussion of the project.
train_images.pdf: This file is a PDF document that shows some sample images from the training dataset, along with their labels and predictions.
word_buffer.py: This file implements a word buffer class that stores the recognized gestures and converts them to words. The word buffer also handles the special symbols and provides a method to clear the buffer.
LICENSE: This file is the MIT license that grants permission to use, copy, modify, and distribute the project code for any purpose.
.gitignore: This file speci