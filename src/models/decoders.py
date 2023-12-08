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
        # input_features: [batch_size, embed_size]
        # captions: [batch_size, max_seq_len]
        # lengths: [batch_size]

        embeddings = self.embedding_layer(captions)
        # embeddings: [batch_size, max_seq_len, embed_size]

        embeddings = torch.cat((input_features.unsqueeze(1), embeddings), 1)
        # embeddings: [batch_size, max_seq_len + 1, embed_size]

        # Move lengths to CPU if necessary
        if lengths.is_cuda:
            lengths = lengths.cpu()

        packed = pack_padded_sequence(
            embeddings, lengths, batch_first=True, enforce_sorted=False)
        # packed: PackedSequence object

        packed_hidden_variables, _ = self.lstm_layer(packed)
        # packed_hidden_variables: PackedSequence object

        # Unpack the output (if needed)
        hidden_variables, _ = torch.nn.utils.rnn.pad_packed_sequence(
            packed_hidden_variables, batch_first=True)
        # hidden_variables: [batch_size, max_seq_len + 1, hidden_size]

        # Reshape for linear layer
        outputs = hidden_variables.contiguous().view(-1, hidden_variables.size(2))
        # outputs: [batch_size * (max_seq_len + 1), hidden_size]

        # Pass through linear layer
        outputs = self.linear_layer(outputs)
        # outputs: [batch_size * (max_seq_len + 1), vocab_size]

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


# # Dummy Input
# dummy_features = torch.randn(32, 256)  # CNN output features
# dummy_captions = torch.randint(0, vocabulary_size, (32, 10))  # Random caption indices
# dummy_lengths = torch.tensor([10] * 32)  # Lengths of each caption

# # Instantiate Decoder
# decoder = DecoderLSTM(embedding_size=256, hidden_layer_size=512,
#                       vocabulary_size=vocabulary_size, num_layers=2)

# # Forward pass
# outputs = decoder(dummy_features, dummy_captions, dummy_lengths)

# print("Output size:", outputs.size())  # Expected: [batch_size * seq_len, vocabulary_size]
