# Convolutional Deep Neural Network for Image Classification

## AIM

To Develop a convolutional deep neural network for image classification and to verify the response for new images.

## Problem Statement and Dataset

Include the Problem Statement and Dataset.

## Neural Network Model

<img width="962" height="468" alt="image" src="https://github.com/user-attachments/assets/c1636d7b-518c-4aac-9f24-30d80c389ac3" />

## DESIGN STEPS

### STEP 1: Problem Definition

Define the objective of classifying images into predefined categories using a Convolutional Neural Network.

### STEP 2: Dataset Collection

Use the FashionMNIST dataset containing labeled images for training and testing.

### STEP 3: Data Preprocessing

Convert images into tensors, normalize pixel values, and create DataLoaders for efficient batch processing.

### STEP 4: Model Architecture Design

Design a CNN consisting of convolution layers, ReLU activation, pooling layers, and fully connected layers for classification.

### STEP 5: Model Training

Train the CNN model using CrossEntropyLoss as the loss function and Adam optimizer for multiple epochs.

### STEP 6: Model Evaluation

Evaluate the trained model using test data and compute performance metrics such as accuracy, confusion matrix, and classification report.

### STEP 7: Prediction on New Images

Test the trained CNN model on unseen images and verify predicted outputs.

### STEP 8: Deployment and Visualization

Save the trained model and visualize predictions for sample images.


## PROGRAM

### Name: S.Navadeep
### Register Number: 212224230180
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.FashionMNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = torchvision.datasets.FashionMNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)

        self.pool = nn.MaxPool2d(2,2)

        self.fc1 = nn.Linear(128*3*3,128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x

model = CNNClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, epochs=3):
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print('Name: S.Navadeep')
        print('Register Number: 212224230180')
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {running_loss/len(train_loader):.4f}")

def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    print("\nTest Accuracy:", acc)

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))


def predict_sample(model, test_loader):
    model.eval()

    images, labels = next(iter(test_loader))
    image = images[0].unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    plt.imshow(images[0].squeeze(), cmap='gray')
    plt.title(f"Predicted Label: {predicted.item()}")
    plt.show()


train_model(model, train_loader, epochs=3)
evaluate_model(model, test_loader)
predict_sample(model, test_loader)

```

## OUTPUT
### Training Loss per Epoch

<img width="416" height="221" alt="image" src="https://github.com/user-attachments/assets/001a7303-0a2b-4baf-809b-6cd9c4eca779" />


### Confusion Matrix

<img width="683" height="561" alt="image" src="https://github.com/user-attachments/assets/0102e17b-e090-4c4e-8382-dfdd81323ab8" />

### Classification Report

<img width="490" height="313" alt="image" src="https://github.com/user-attachments/assets/354dbc74-cd9b-4472-91c4-eed599676dc5" />

### New Sample Data Prediction

<img width="417" height="437" alt="image" src="https://github.com/user-attachments/assets/693436d5-1f30-4838-a8f7-5530be1dfccf" />

## RESULT

Thus, a Convolutional Deep Neural Network for image classification was successfully developed and tested using the FashionMNIST dataset, and the model accurately predicted the classes of new images.
