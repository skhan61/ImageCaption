import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class DecoderLSTM(nn.Module):
    def __init__(self, embedding_size, hidden_layer_size, vocabulary_size, num_layers, max_seq_len=20):
        super(DecoderLSTM, self).__init__()
        self.embedding_layer = nn.Embedding(vocabulary_size, embedding_size)
        self.lstm_layer = nn.LSTM(
            embedding_size, hidden_layer_size, num_layers, batch_first=True)
        self.linear_layer = nn.Linear(hidden_layer_size, vocabulary_size)
        self.max_seq_len = max_seq_len

    def forward(self, input_features, captions, lengths):
        embeddings = self.embedding_layer(captions)
        embeddings = torch.cat((input_features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(
            embeddings, lengths, batch_first=True, enforce_sorted=False)
        hidden_variables, _ = self.lstm_layer(packed)
        outputs = self.linear_layer(
            hidden_variables.data)  # Use 'data' attribute
        return outputs

    def sample(self, input_features, states=None):
        sampled_indices = []
        lstm_inputs = input_features.unsqueeze(1)

        for i in range(self.max_seq_len):
            hidden_variables, states = self.lstm_layer(lstm_inputs, states)
            outputs = self.linear_layer(hidden_variables.squeeze(1))
            _, predicted = outputs.max(1)
            sampled_indices.append(predicted)
            lstm_inputs = self.embedding_layer(predicted)
            lstm_inputs = lstm_inputs.unsqueeze(1)

        sampled_indices = torch.stack(sampled_indices, 1)
        return sampled_indices
