import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, mode):
        super(ConvNet, self).__init__()
        
        # Define various layers here, such as in the tutorial example
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=40, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=40, out_channels=40, kernel_size=5, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        #Model1
        self.fc1_1 = nn.Linear(in_features=784, out_features=100)
        self.out_1 = nn.Linear(in_features=100, out_features=10)
        #Model2 & 3
        self.fc1_2 = nn.Linear(in_features=640, out_features=100)
        self.out_2 = nn.Linear(in_features=100, out_features=10)
        #Model4
        self.fc2_4 = nn.Linear(in_features=100, out_features=100)
        #Model5
        self.fc1_5 = nn.Linear(in_features=640, out_features=1000)
        self.fc2_5 = nn.Linear(in_features=1000, out_features=1000)
        self.out_5 = nn.Linear(in_features=1000, out_features=10)

        # This will select the forward pass function based on mode for the ConvNet.
        # Based on the question, you have 5 modes available for step 1 to 5.
        # During creation of each ConvNet model, you will assign one of the valid mode.
        # This will fix the forward function (and the network graph) for the entire training/testing
        if mode == 1:
            self.forward = self.model_1
        elif mode == 2:
            self.forward = self.model_2
        elif mode == 3:
            self.forward = self.model_3
        elif mode == 4:
            self.forward = self.model_4
        elif mode == 5:
            self.forward = self.model_5
        else: 
            print("Invalid mode ", mode, "selected. Select between 1-5")
            exit(0)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
        
    # Baseline model. step 1
    def model_1(self, X):
        # ======================================================================
        # One fully connected layer.
        #
        # ----------------- YOUR CODE HERE ----------------------
        x = X.view(-1, self.num_flat_features(X))
        x = F.sigmoid(self.fc1_1(x))
        x = self.out_1(x)

        # Uncomment the following return stmt once method implementation is done.
        return x
        # Delete line return NotImplementedError() once method is implemented.
        #return NotImplementedError()

    # Use two convolutional layers.
    def model_2(self, X):
        # ======================================================================
        # Two convolutional layers + one fully connnected layer.
        #
        # ----------------- YOUR CODE HERE ----------------------
        x = F.relu(self.conv1(X))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, self.num_flat_features(x))
        x = F.sigmoid(self.fc1_2(x))
        x = self.out_2(x)

        # Uncomment the following return stmt once method implementation is done.
        return x
        # Delete line return NotImplementedError() once method is implemented.
        #return NotImplementedError()

    # Replace sigmoid with ReLU.
    def model_3(self, X):
        # ======================================================================
        # Two convolutional layers + one fully connected layer, with ReLU.
        #
        # ----------------- YOUR CODE HERE ----------------------
        x = F.relu(self.conv1(X))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1_2(x))
        x = self.out_2(x)
        # Uncomment the following return stmt once method implementation is done.
        return x
        # Delete line return NotImplementedError() once method is implemented.
        #return NotImplementedError()

    # Add one extra fully connected layer.
    def model_4(self, X):
        # ======================================================================
        # Two convolutional layers + two fully connected layers, with ReLU.
        #
        # ----------------- YOUR CODE HERE ----------------------
        x = F.relu(self.conv1(X))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1_2(x))
        x = F.relu(self.fc2_4(x))
        x = self.out_2(x)
        # Uncomment the following return stmt once method implementation is done.
        return x
        # Delete line return NotImplementedError() once method is implemented.
        #return NotImplementedError()

    # Use Dropout now.
    def model_5(self, X):
        # ======================================================================
        # Two convolutional layers + two fully connected layers, with ReLU.
        # and  + Dropout.
        #
        # ----------------- YOUR CODE HERE ----------------------
        x = F.relu(self.conv1(X))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1_5(x))
        x = F.dropout(x, p=0.5)
        x = F.relu(self.fc2_5(x))
        x = F.dropout(x, p=0.5)
        x = self.out_5(x)

        # Uncomment the following return stmt once method implementation is done.
        return x
        # Delete line return NotImplementedError() once method is implemented.
        #return NotImplementedError()