import flwr as fl
import tensorflow as tf
import numpy as np

# Import the model from model.py
from model import generate_ann

# Load and partition the dataset
def load_datasets(cid: int):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    # Normalize data
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train, y_train = x_train[:10_000], y_train[:10_000]
    x_test, y_test = x_test[:1000], y_test[:1000]
    
    # Number of clients
    num_clients = 5

    # Split training set into partitions to simulate the individual dataset
    x_train_splits = np.array_split(x_train, num_clients)
    y_train_splits = np.array_split(y_train, num_clients)

    x_val_splits = np.array_split(x_test, num_clients)
    y_val_splits = np.array_split(y_test, num_clients)

    # Select partition based on client id
    x_train, y_train = x_train_splits[cid], y_train_splits[cid]
    x_val, y_val = x_val_splits[cid], y_val_splits[cid]
    
    # Further split the training data to create validation set
    split_point = int(len(x_train) * 0.9) # Using 90% of training data for actual training and 10% for validation
    x_train, x_val = x_train[:split_point], x_train[split_point:]
    y_train, y_val = y_train[:split_point], y_train[split_point:]

    return (x_train, y_train), (x_val, y_val)

# Create a class to contain the details of the client and be the interface
class MyClient(fl.client.NumPyClient):
    def __init__(self, net, train_dataset, test_dataset):
        self.model = net
        self.trainloader = train_dataset
        self.valloader = test_dataset
        
    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.trainloader[0],self.trainloader[1], epochs=1, batch_size=32, steps_per_epoch=3)
        return self.model.get_weights(), len(self.trainloader[0]), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.valloader[0], self.valloader[1])
        return loss, len(self.valloader[0]), {"accuracy": float(accuracy)}

# The client_id will be an environment variable
import os
cid = int(os.environ["CLIENT_ID"])

# Load the datasets based on the client id
train_dataset, val_dataset = load_datasets(cid)

# Start the client
model = generate_ann()

fl.client.start_numpy_client(server_address="localhost:8080", client=MyClient(model, train_dataset, val_dataset).to_client())