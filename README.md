# Developing a Neural Network Classification Model

## AIM
To develop a neural network classification model for the given dataset.

## THEORY
An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

<img width="710" height="841" alt="image" src="https://github.com/user-attachments/assets/30d8fc42-9f92-4d46-8542-292ad3b67afa" />


## DESIGN STEPS
### STEP 1: 
Data Collection and Understanding – Load the dataset, inspect features, and identify the target variable.

### STEP 2: 
Data Cleaning and Encoding – Handle missing values and convert categorical data and labels into numerical form.

### STEP 3: 
Feature Scaling and Data Splitting – Normalize features and split data into training and testing sets.

### STEP 4: 
Model Architecture Design – Define the neural network layers, neurons, and activation functions.


### STEP 5: 

Model Training and Optimization – Train the model using a loss function and optimizer through backpropagation.

### STEP 6: 
Model Evaluation and Prediction – Evaluate performance using metrics and make predictions on unseen data.


## PROGRAM

### Name: PRAKASH C

### Register Number: 212223240122

```
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from torch.utils.data import TensorDataset, DataLoader

# Load dataset
data = pd.read_csv('/content/customers.csv')
data.head()

data.columns

# Drop ID column as it's not useful for classification
data = data.drop(columns=["ID"])

# Handle missing values
data.fillna({"Work_Experience": 0, "Family_Size": data["Family_Size"].median()}, inplace=True)

# Encode categorical variables
categorical_columns = ["Gender", "Ever_Married", "Graduated", "Profession", "Spending_Score", "Var_1"]
for col in categorical_columns:
    data[col] = LabelEncoder().fit_transform(data[col])

data.head()

# Encode target variable
label_encoder = LabelEncoder()
data["Segmentation"] = label_encoder.fit_transform(data["Segmentation"])  # A, B, C, D -> 0, 1, 2, 3

# Split features and target
X = data.drop(columns=["Segmentation"])
y = data["Segmentation"].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# Create DataLoader
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# Define Neural Network(Model1)
# Define Neural Network(Model1)
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        self.fc1 =nn.Linear(input_size, 32)
        self.fc2 =nn.Linear(32, 16)
        self.fc3 =nn.Linear(16, 8)
        self.fc4 =nn.Linear(8,4)

    def forward(self, x):
        x = F.relu(self.fc1(x))#activation part,Rectified Linear Unit
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Training Loop
def train_model(model, train_loader, criterion, optimizer, epochs):
  model.train()
  for epoch in range(epochs):
    for inputs,labels in train_loader:
      optimizer.zero_grad()
      outputs=model(inputs)
      loss=criterion(outputs,labels)
      loss.backward()
      optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Initialize model
model =PeopleClassifier(input_size=X_train.shape[1])
criterion =nn.CrossEntropyLoss()
optimizer =optim.Adam(model.parameters(),lr=0.01)

train_model(model,train_loader,criterion,optimizer,epochs=100)

# Evaluation
model.eval()
predictions, actuals = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.numpy())
        actuals.extend(y_batch.numpy())

# Compute metrics
accuracy = accuracy_score(actuals, predictions)
conf_matrix = confusion_matrix(actuals, predictions)
class_report = classification_report(actuals, predictions, target_names=[str(i) for i in label_encoder.classes_])
print("Name: PRAKASH C ")
print("Register No: 212223240122")
print(f'Test Accuracy: {accuracy:.2f}%')
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(conf_matrix, annot=True, cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_,fmt='g')
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

# Prediction for a sample input
sample_input = X_test[12].clone().unsqueeze(0).detach().type(torch.float32)
with torch.no_grad():
    output = model(sample_input)
    # Select the prediction for the sample (first element)
    predicted_class_index = torch.argmax(output[0]).item()
    predicted_class_label = label_encoder.inverse_transform([predicted_class_index])[0]
print("Name: PRAKASH C")
print("Register No: 212223240122")
print(f'Predicted class for sample input: {predicted_class_label}')
print(f'Actual class for sample input: {label_encoder.inverse_transform([y_test[12].item()])[0]}')


```

### Dataset Information

<img width="1192" height="856" alt="Screenshot 2026-02-13 094354" src="https://github.com/user-attachments/assets/93f99094-17a3-4d09-90c7-8a09178a9403" />


### OUTPUT

<img width="1292" height="251" alt="Screenshot 2026-02-13 094118" src="https://github.com/user-attachments/assets/b0bd2896-321c-45a7-8d1c-7523fff9826b" />

<img width="786" height="95" alt="Screenshot 2026-02-13 094131" src="https://github.com/user-attachments/assets/8be360a2-5aa4-4997-9b38-e55ce0619e83" />

<img width="368" height="219" alt="Screenshot 2026-02-13 094143" src="https://github.com/user-attachments/assets/d07b5fed-49e5-4df5-92e3-22bfd127715c" />


## Confusion Matrix

<img width="796" height="565" alt="Screenshot 2026-02-13 094159" src="https://github.com/user-attachments/assets/adad9a0a-003b-43c4-b6a0-fa0f5a15d630" />


## Classification Report

<img width="777" height="439" alt="Screenshot 2026-02-13 094151" src="https://github.com/user-attachments/assets/2119518e-3dd9-410e-bd48-32ac819431cf" />


### New Sample Data Prediction

<img width="450" height="98" alt="Screenshot 2026-02-13 094206" src="https://github.com/user-attachments/assets/1e82df0a-a7c9-4b0d-965e-078c21be8765" />



## RESULT
The neural network model was trained successfully and customer segments were predicted.
