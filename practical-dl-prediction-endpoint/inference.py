import torch as th
import numpy as np
from keras_preprocessing.sequence import pad_sequences
import torch.nn as tnn
import pandas as pd


def split_X_and_Y(sequence):
    X = [0]
    Y = [sequence[0]]
    for idx, token in enumerate(sequence[:-1]):
        X.append(token)
        Y.append(sequence[idx + 1])
    return pd.Series({"X": X, "Y": Y})


def extend_data(X, Y, context_size):
    new_X = []
    new_Y = []
    for idx, x in enumerate(X):
        y = Y[idx]
        for idx2, word in enumerate(x):
            if idx2 < 5:
                new_X.append(pad_sequences([x[:idx2 + 1]], maxlen = context_size)[0])
            else:
                new_X.append(x[idx2 - context_size + 1: idx2 + 1])
            new_Y.append(np.array(y[idx2], dtype = "int32"))
    new_X = np.vstack(new_X)
    new_Y = np.array(new_Y).astype(np.int32)
    return new_X, new_Y


def get_predictions(sentence, model):
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    model = model.to(device)
    sequence = [model.word2idx[word] for word in sentence]
    splitted = split_X_and_Y(sequence)
    X, Y = extend_data([splitted.X], [splitted.Y], model.context_size)
    X, Y = th.tensor(X), th.tensor(Y)
    X = X.to(device)
    Y = Y.to(device)
    model.h_lstm = model._init_hidden(len(X))
    preds = tnn.functional.softmax(model.forward(X), dim = 1).topk(3, dim = 1)[1]
    results = []
    for idx, pred in enumerate(preds[1:-1]):
        word_preds = []
        for idx2 in pred:
            word_preds.append(model.idx2word[int(idx2)])
        prev_word = sentence[idx]
        expected_word = sentence[idx + 1]
        ans = "Previous word: {} \t Expected word: {} \t Predictions: {}\t{}\t{}".format(prev_word, expected_word, *word_preds)
        results.append(ans)
        return ans
