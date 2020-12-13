# Muhtasim Zia (mxz190001)
# Siddhath Sekar (sxs190240)
# Sandeep Shahi (sxs160432)
# Kevin Nguyen (kxn120430)
#
# CS 6375.002 Machine Learning Course Project
# This project aims to implement a recurrent neural network with long short-term memory (LSTM)
# For time-series regression problems such as stock price predictions on the NYSE

import numpy as np
from sklearn.model_selection import train_test_split
from io import StringIO
import pandas as pd
import requests
import logging
import math
np.random.seed(1)

logging.basicConfig(filename="Output.log", level=logging.INFO)


def fetch_and_preprocess_data():
    # This data originates from UCI Machine Learning Repository
    # Get data from url
    url = "https://drive.google.com/file/d/1Lg7cVQw2ub9jdq1wLpNMQr_YPg72-PVG/view?usp=sharing"
    file_id = url.split('/')[-2]
    dwn_url = 'https://drive.google.com/uc?export=download&id=' + file_id
    csv_url = requests.get(dwn_url).text
    csv_raw = StringIO(csv_url)
    # preprocessing data
    # Replace missing values
    missing_values = ["?"]
    data = pd.read_csv(csv_raw, na_values=missing_values)

    data = data.drop(['EMA_10', 'EMA_20', 'EMA_50', 'EMA_200'], axis=1)

    # taking all features and output
    X = (np.array(data.iloc[:, 1:]))
    X = np.round(X**2)
    y = np.array(data.iloc[:, 0])

    return X,y


def find_accuracy(y_test, y_predicted, no_of_samples):
    # Returns accuracy
    get_error = 0

    for i in range(no_of_samples):
        get_error += abs(((y_test[i] - y_predicted[i])/ y_test[i]))
    total_error_percent = (get_error / num_samples) * 100
    return 100 - total_error_percent


def rmse(y_test, y_predicted, no_of_samples):
    # Returns root mean square error
    get_error = 0

    for i in range(no_of_samples):
        get_error += (y_test[i] - y_predicted[i])**2
    return math.sqrt(get_error / (2 * no_of_samples))


class RecurrentNeuralNetworkModel:
    def __init__(self):
        # Initialize with random weights
        self.Weight = [1, 1]
        self.Weight_B = [0.001, 0.001]
        self.Weight_A = [0, 0]

        self.b1 = 1.2
        self.b2 = 0.5

    def forward_cell(self, xt, cell_t):
        # Calculates hidden state to pass forward to next time step
        cell = xt * self.Weight[0] + cell_t * self.Weight[1]

        return cell

    def forward_propagation(self, X):

        cell = np.zeros((X.shape[0], X.shape[1]+1))

        for i in range(0, X.shape[1]):
            next_cell = self.forward_cell(X[:, i], cell[:, i])
            cell[:, i+1] = next_cell
        return cell

    @staticmethod
    def output_gradient(pred, real):

        grad = (pred - real) / num_samples
        grad_output = 2 * grad

        return grad_output

    def backward_propagation(self, X, cell, output_gradient):
        # Update weights to network
        gradient_change = np.zeros(( X.shape[0], X.shape[1]+1 ))
        gradient_change[:,-1] = output_gradient

        weight_X_gradient = 0
        weight_cell_gradient = 0

        for i in range(X.shape[1], 0, -1):

            weight_X_gradient += np.sum( gradient_change[:, i] * X[:, i-1] )
            weight_cell_gradient += np.sum( gradient_change[:, i] * cell[:, i-1] )
            gradient_change[:, i-1] = gradient_change[:, i] * self.Weight[1]

        return (weight_X_gradient, weight_cell_gradient), gradient_change

    def update_rnn(self, X, Y, Weight_A_2, Weight_B):

        initial = self.forward_propagation(X)

        output_gradient = self.output_gradient(initial[:, -1], Y)
        gradient_weight, _ = self.backward_propagation(X, initial, output_gradient)

        self.Weight_A = np.sign(gradient_weight)

        for i, _ in enumerate(self.Weight):
            if self.Weight_A[i] == Weight_A_2[i]:
                Weight_B[i] *= self.b1
            else:
                Weight_B[i] *= self.b2

        self.Weight_B = Weight_B

    def rnn_train(self, X, Y, iterations):
        logging.info('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        logging.info('Recurrent Neural Network')
        print('Recurrent Neural Network')
        logging.info(f'Number of maximum epochs: {iterations}')
        print(f'Number of maximum epochs: {iterations}')
        for epochs in range(iterations):

            self.update_rnn(X, Y, self.Weight_A, self.Weight_B)

            for i, _ in enumerate(self.Weight):
                self.Weight[i] -= self.Weight_A[i] * self.Weight_B[i]


user_input = int(input("Please give the number of iterations for training dataset (3000 is a good default): \n"))
X, y = fetch_and_preprocess_data()

# Splitting for test and train
# Ensure data ordering is kept for time series task
test_size=0.10
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=False, stratify=None)
num_samples = len(X_train)

# Train and calculate accuracy/ error
rnn = RecurrentNeuralNetworkModel()
rnn.rnn_train(X_train, y_train, user_input)
print(f'Train/ Test split: {(1-test_size)*100}/{test_size*100}')
logging.info(f'Train/ Test split: {(1-test_size)*100}/{test_size*100}')
y = rnn.forward_propagation(X_train)[:, -1]
train_accuracy = find_accuracy(y_train, y, num_samples)
print("Training Accuracy: ", train_accuracy)
logging.info(f'Training Accuracy: {train_accuracy}')
logging.info(f'Training RMSE: {rmse(y_train, y, num_samples)}')
print(f'Training RMSE: {rmse(y_train, y, num_samples)}')
# print("Training: \t\t", y_test)


# Test and calculate accuracy/ error
num_samples = len(X_test)
y = rnn.forward_propagation(X_test)[:, -1]

# print("Predicted: \t", y)
accuracy = find_accuracy(y_test, y, num_samples)

print("Testing Accuracy: ", accuracy)
logging.info(f'Testing Accuracy: {accuracy}')
logging.info(f'Testing RMSE: {rmse(y_test, y, num_samples)}')
print(f'Testing RMSE: {rmse(y_test, y, num_samples)}')

