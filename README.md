# DL- Developing a Recurrent Neural Network Model for Stock Prediction

## AIM
To develop a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data.

## Problem Statement and Dataset

### Problem Statement:
To develop an RNN model that predicts the future stock closing price using historical stock closing price data.

### Dataset:
The dataset contains historical stock price information. The Close column is used as the target variable. The training dataset is used to train the RNN model and the testing dataset is used to evaluate the prediction performance.



## DESIGN STEPS
### STEP 1:

Load and preprocess the stock price dataset.
Read the training and testing CSV files and extract the Close price column.

### STEP 2:

Normalize the stock prices.
Apply Min-Max Scaling to convert the closing prices into a range between 0 and 1.

### STEP 3:

Create time-series sequences.
Create sequences of 60 previous closing prices to predict the next closing price.

### STEP 4:

Convert the data into PyTorch tensors.
Create TensorDataset and DataLoader for training the RNN model.

### STEP 5:

Build and train the RNN model.
Create an RNN with hidden layers and a fully connected output layer. Use MSE Loss and Adam optimizer for training.

### STEP 6:

Evaluate and visualize the predictions.
Predict stock prices using the test data, convert the normalized values back to original prices, and compare actual and predicted prices using graphs.




## PROGRAM

### Name:

### Register Number:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

## Step 1: Load and Preprocess Data
# Load training and test datasets
df_train = pd.read_csv('C:\\Users\\vigne\\Downloads\\EX 5 dataset\\trainset.csv')
df_test = pd.read_csv('C:\\Users\\vigne\\Downloads\\EX 5 dataset\\testset.csv')
# Use closing prices
train_prices = df_train['Close'].values.reshape(-1, 1)
test_prices = df_test['Close'].values.reshape(-1, 1)
     
# Normalize the data based on training set only
scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(train_prices)
scaled_test = scaler.transform(test_prices)
# Create sequences
def create_sequences(data, seq_length):
    x = []
    y = []
    for i in range(len(data) - seq_length):
        x.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(x), np.array(y)

seq_length = 60
x_train, y_train = create_sequences(scaled_train, seq_length)
x_test, y_test = create_sequences(scaled_test, seq_length)
x_train.shape, y_train.shape, x_test.shape, y_test.shape
     
# Convert to PyTorch tensors
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Create dataset and dataloader
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
class RNNModel(nn.Module):

    def __init__(self):
        super(RNNModel, self).__init__()

        self.rnn = nn.RNN(
            input_size=1,
            hidden_size=50,
            num_layers=2,
            batch_first=True
        )

        self.fc = nn.Linear(50, 1)

    def forward(self, x):

        out, hidden = self.rnn(x)

        # Take the output from the last time step
        out = out[:, -1, :]

        out = self.fc(out)

        return out
model = RNNModel()

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = model.to(device)

print("Device:", device)
print(model)
!pip install torchinfo
from torchinfo import summary

summary(
    model,
    input_size=(64, 60, 1)
)
criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

print("Loss Function:", criterion)
print("Optimizer:", optimizer)
num_epochs = 20

train_losses = []

for epoch in range(num_epochs):

    model.train()

    running_loss = 0.0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)

    train_losses.append(epoch_loss)

    print(
        f"Epoch [{epoch+1}/{num_epochs}], "
        f"Training Loss: {epoch_loss:.6f}"
    )
print('Name:       Vignesh M          ')
print('Register Number:   212223240176   ')

plt.plot(
    train_losses,
    label='Training Loss'
)

plt.xlabel('Epoch')
plt.ylabel('MSE Loss')

plt.title(
    'Training Loss Over Epochs'
)

plt.legend()
plt.show()
model.eval()

with torch.no_grad():

    predicted = model(
        x_test_tensor.to(device)
    ).cpu().numpy()

    actual = y_test_tensor.cpu().numpy()
# Convert normalized values back to original prices

predicted_prices = scaler.inverse_transform(
    predicted
)

actual_prices = scaler.inverse_transform(
    actual
)

print("Predicted Prices:")
print(predicted_prices[:5])

print("\nActual Prices:")
print(actual_prices[:5])
print('Name:   Vignesh M         ')
print('Register Number:   212223240176   ')

plt.figure(figsize=(10, 6))

plt.plot(
    actual_prices,
    label='Actual Price'
)

plt.plot(
    predicted_prices,
    label='Predicted Price'
)

plt.xlabel('Time')
plt.ylabel('Price')

plt.title(
    'Stock Price Prediction using RNN'
)

plt.legend()
plt.show()
print(
    f"Predicted Price: {predicted_prices[-1][0]:.2f}"
)

print(
    f"Actual Price: {actual_prices[-1][0]:.2f}"
)


```

### OUTPUT

## Training Loss Over Epochs Plot

<img width="1060" height="657" alt="image" src="https://github.com/user-attachments/assets/dc7a5b61-6838-47d7-b5fa-eaebcd036a48" />


## True Stock Price, Predicted Stock Price vs time

<img width="1470" height="772" alt="image" src="https://github.com/user-attachments/assets/0418cd15-81ee-4a82-ac1b-d4120337105d" />

### Predictions
<img width="1547" height="115" alt="image" src="https://github.com/user-attachments/assets/43417a79-367f-4d78-87e3-cf622c5d2ed0" />


## RESULT
Thus, the RNN model was successfully developed and used to predict stock prices from historical data.
