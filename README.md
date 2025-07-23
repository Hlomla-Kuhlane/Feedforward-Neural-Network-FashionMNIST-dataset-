This implements a feedforward neural network using PyTorch to classify images from the FashionMNIST dataset. The network is trained to recognize different categories of fashion items, such as dresses, trousers, and sneakers.

File Description

classifier.py: This file contains the Python code for the neural network.

FashionClassifier class: Defines the architecture of the neural network.

loadData() function: Loads the FashionMNIST dataset.

trainModel() function: Trains the neural network.

evaluateModel() function: Evaluates the trained network.

predict_image() function: Predicts the class of a single image from a file path.

main() function:  Coordinates the execution of the program, including training, evaluation, and prediction.

log.txt: This file contains the output from a previous training run of the model. It includes the training loss after each epoch and the final accuracy of the model on the test set.

ANN Assignment 2025 v1.1 (2).pdf: This file is the assignment brief.

How the Files Fit Together

Data Loading and Preprocessing:  The loadData() function in classifier.py loads the FashionMNIST dataset, which is assumed to be located in a folder named "FashionMNIST" in the same directory as the script. The dataset is preprocessed (normalized) and split into training and testing sets.  The  torchvision  library is used for this.

Model Definition: The  FashionClassifier  class in  classifier.py  defines the neural network architecture. This class inherits from  nn.Module  and consists of several fully connected layers, ReLU activation functions, and dropout layers for regularization.

Training: The  trainModel()  function in  classifier.py  trains the neural network defined in the  FashionClassifier  class.  It uses the training data loaded by  loadData(), the CrossEntropyLoss function, and the Adam optimizer.  The training loss for each epoch is printed to the console, and also saved in log.txt.

Evaluation: The  evaluateModel()  function in  classifier.py  evaluates the trained network on the test data loaded by  loadData().  It calculates the accuracy of the model's predictions and prints the final accuracy to the console, also saved in log.txt.

Prediction: The  predict_image()  function in  classifier.py  allows the user to classify a single image from a file.  The  main()  function prompts the user for the file path, loads the image, and uses the trained model to predict its class.

Main Execution: The  main()  function in  classifier.py  coordinates the entire process. It creates an instance of the  FashionClassifier  model, loads the data, trains the model, evaluates it, and then enters a loop to allow the user to classify individual images.
