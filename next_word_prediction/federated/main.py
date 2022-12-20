from federated import train_multiple_federated
from utils.processors import get_local_and_remote_data
from utils.models import bidirectional_LSTM
import torch as th
import pickle
import pandas as pd

if __name__ == "__main__":
    model_file = "data/LSTM_model_top3.pth"
    federated_file = "data/LSTM_model_federated.pth"

    processed_data = "data/processed_data.pkl"
    word2idx_file = "data/tokenizer_keys.pkl"

    min_tweets = 20
    local_share = 0.2
    context_size = 5
    epochs = 5
    D = 300
    n_nodes = 128

    data = pd.read_pickle(processed_data)
    with open(word2idx_file, "rb") as f:
        word2idx = pickle.load(f)

    local_data, remote_data = get_local_and_remote_data(data, local_share)
    model = bidirectional_LSTM(context_size, len(word2idx), D, word2idx, n_nodes)
    model.load_state_dict(th.load(model_file))
    train_multiple_federated(model, remote_data, federated_file, len(word2idx))
