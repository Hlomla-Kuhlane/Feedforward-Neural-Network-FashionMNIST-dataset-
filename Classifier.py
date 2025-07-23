from torchvision import datasets
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.io
import os

# Define the neural network architecture for classifying fashion items
class FashionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        inputLayer = 784  # 28x28 pixels flattened
        hiddenLayer = [256, 128, 64]  # Three hidden layers for better learning capacity
        outputLayer = 10  # 10 classes in FashionMNIST dataset

        # Define layers of the network
        self.flatten = nn.Flatten()  # Converts image into 1D vector
        self.layers = nn.Sequential(
            nn.Linear(inputLayer, hiddenLayer[0]),
            nn.ReLU(),
            nn.Dropout(p=0.2),  # Helps prevent overfitting
            nn.Linear(hiddenLayer[0], hiddenLayer[1]),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hiddenLayer[1], hiddenLayer[2]),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hiddenLayer[2], outputLayer)  # Final layer outputs class scores
        )

    def forward(self, x):
        x = self.flatten(x)        # Flatten 2D image into 1D
        return self.layers(x)      # Pass input through all layers


# Load the training and testing data
def loadData():
    DATA_DIR = "."
    batch_size = 64  # Number of images per batch

    # Transform: Convert images to tensors and normalize pixel values
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Download and prepare the datasets
    trainSet = datasets.FashionMNIST(DATA_DIR, train=True, download=False, transform=transform)
    testSet = datasets.FashionMNIST(DATA_DIR, train=False, download=True, transform=transform)

    # Create data loaders for batching
    trainLoader = DataLoader(trainSet, batch_size, shuffle=True)
    testLoader = DataLoader(testSet, batch_size, shuffle=False)

    return trainLoader, testLoader


# Train the neural network on the training data
def trainModel(model, trainLoader, num_epochs=30, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()  # Loss function for multi-class classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Optimizer to adjust weights

    print("Pytorch Output...")

    # Run training for specified number of epochs
    for epoch in range(num_epochs):
        TotalTrainingLoss = 0  # Keep track of loss for this epoch

        for images, labels in trainLoader:
            optimizer.zero_grad()         # Reset gradients from previous step
            outputs = model(images)       # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()               # Backward pass (compute gradients)
            optimizer.step()              # Update model parameters
            TotalTrainingLoss += loss.item()  # Accumulate batch loss

        # Print average loss for the epoch
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {TotalTrainingLoss/len(trainLoader):.4f}")


# Evaluate the model on the test data
def evaluateModel(model, testLoader):
    correct_predictions = 0
    total_samples = 0

    # Disable gradient calculation for faster testing
    with torch.no_grad():
        for images, labels in testLoader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)  # Get predicted class
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()  # Count correct predictions

    accuracy = 100 * correct_predictions / total_samples
    print(f"Test Accuracy: {accuracy:.2f}%")
    print("Done!")
    return accuracy


# Predict a single image from a given file path
def predict_image(model, image_path):
    try:
        # Load image in grayscale format
        img = torchvision.io.read_image(image_path, mode=torchvision.io.ImageReadMode.GRAY)

        # Normalize image to match training preprocessing
        img = img.squeeze().float() / 255.0
        img = (img - 0.5) / 0.5

        img = img.unsqueeze(0)  # Add batch dimension

        # Run the image through the model
        with torch.no_grad():
            output = model(img)
            _, predicted_class = torch.max(output.data, 1)

        # Map predicted class index to class name
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

        return class_names[predicted_class.item()]

    except Exception as e:
        print(f"There was a problem with the image: {e}")
        return None


# Main function to train the model and interact with the user
def main():
    model = FashionClassifier()              # Initialize the neural network
    trainLoader, testLoader = loadData()     # Load data

    trainModel(model, trainLoader)           # Train the model
    evaluateModel(model, testLoader)         # Test the model

    print("Please enter a filepath")         # Start prediction loop

    while True:
        user_input = input("> ").strip()

        if user_input.lower() == 'exit':
            print("Exiting...")
            break

        if not os.path.exists(user_input):
            print("File not found. Please try again.")
            continue

        # Predict the class of the given image
        prediction = predict_image(model, user_input)
        if prediction is not None:
            print(f"Classifier: {prediction}")


if __name__ == "__main__":
    main()
