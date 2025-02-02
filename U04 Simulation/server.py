import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.server import ServerConfig
import tensorflow as tf

# Import the model from model.py
from model import generate_ann

# Get model parameters as a list of NumPy ndarrays
def get_initial_parameters():
    model = generate_ann()
    return model.get_weights()

# Create strategy
strategy = FedAvg(
    initial_parameters=fl.common.ndarrays_to_parameters(get_initial_parameters()),
)

# Start the server
fl.server.start_server(
    server_address="localhost:8080",
    config=ServerConfig(num_rounds=3),
    strategy=strategy
)