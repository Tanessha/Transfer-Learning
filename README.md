# Implementation-of-Transfer-Learning
## Aim
To Implement Transfer Learning for classification using VGG-19 architecture.
## Problem Statement and Dataset
Develop an image classification model using transfer learning with the pre-trained VGG19 model.

## DESIGN STEPS
### STEP 1:
Import required libraries.Then dataset is loaded and define the training and testing dataset.

### STEP 2:
Initialize the model,loss function,optimizer. CrossEntropyLoss for multi-class classification and Adam optimizer for efficient training.

### STEP 3:
Train the model with training dataset.

### STEP 4:
Evaluate the model with testing dataset.

### STEP 5:
Make Predictions on New Data.

## PROGRAM
```python
# Load Pretrained Model and Modify for Transfer Learning
from torchvision.models import VGG19_Weights
model = models.vgg19(weights=VGG19_Weights.DEFAULT)

# Modify the final fully connected layer to match the dataset classes
num_classes = len(train_dataset.classes)
model.classifier[6] = nn.Linear(in_features=4096, out_features=num_classes)

# Include the Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

# Train the model
def train_model(model, train_loader,test_loader,num_epochs=10):
    train_losses = []
    val_losses = []
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

        # Compute validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_losses.append(val_loss / len(test_loader))
        model.train()

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')

    # Plot training and validation loss
    print("Name: TANESSHA KANNAN        ")
    print("Register Number: 212223040225       ")
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()


```

## OUTPUT
### Training Loss, Validation Loss Vs Iteration Plot
![image](https://github.com/user-attachments/assets/1cdcf383-ea01-45e8-9567-06bbc657d770)


### Confusion Matrix
![image](https://github.com/user-attachments/assets/b579b714-3222-484e-9ae4-c7c04aff1a3b)


### Classification Report
![image](https://github.com/user-attachments/assets/38760b4a-6263-4023-942f-10109259be6b)


### New Sample Prediction
![image](https://github.com/user-attachments/assets/fb6f5c1a-ce89-49ba-8d75-5d3be21ebdaf)


![image](https://github.com/user-attachments/assets/ccc2a41d-f334-4d92-a9e3-c39938a15e81)

## RESULT
Thus, the transfer Learning for classification using VGG-19 architecture has succesfully implemented.
