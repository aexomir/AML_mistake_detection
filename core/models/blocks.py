import math

import torch
import torch.nn as nn
from torch import Tensor
from constants import Constants as const

# define the transformer backbone here
EncoderLayer = nn.TransformerEncoderLayer
Encoder = nn.TransformerEncoder


def fetch_input_dim(config, decoder=False):
    if config.backbone == const.OMNIVORE:
        return 1024
    elif config.backbone == const.SLOWFAST:
        return 2304
    elif config.backbone == const.X3D:
        return 400
    elif config.backbone == const.RESNET3D:
        return 400
    elif config.backbone == const.IMAGEBIND:
        if decoder is True:
            return 1024
        k = len(config.modality)
        return 1024 * k
    elif config.backbone == const.EGOVLP:
        return 768  # Updated to 768
    elif config.backbone == const.PERCEPTIONENCODER:
        return 768  # VideoMAE-base features
    else:
        raise ValueError(f"Unsupported backbone: {config.backbone}")



class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x


class MLP1(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP1, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size * 8)
        self.layer2 = nn.Linear(hidden_size * 8, hidden_size * 2)
        self.layer3 = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x


class CNN(nn.Module):
    def __init__(self, in_channels, final_width, final_height, num_classes):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * final_width * final_height, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor, indices=None) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        if indices is None:
            x = x + self.pe[:, :x.size(1)]
        else:
            pos = torch.cat([self.pe[:, index] for index in indices])
            x = x + pos
        return self.dropout(x)


class PositionalEncodingLearn(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.embed = nn.Embedding(max_len, d_model)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.embed.weight)

    def forward(self, x, indices=None):
        # x: b, l, d
        r = torch.arange(x.shape[1], device=x.device)
        embed = self.embed(r)  # seq_len, embedding_dim
        return x + embed.repeat(x.shape[0], 1, 1)


class RNNBaseline(nn.Module):
    """
    RNN/LSTM baseline for mistake detection.
    Processes sequences of sub-segment features within each step.
    """
    def __init__(self, config, hidden_size=256, num_layers=2, dropout=0.2, 
                 bidirectional=True, use_attention=False, rnn_type='LSTM'):
        super(RNNBaseline, self).__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.rnn_type = rnn_type
        
        input_dim = fetch_input_dim(config)
        num_directions = 2 if bidirectional else 1
        
        # RNN/LSTM layer
        if rnn_type.upper() == 'LSTM':
            self.rnn = nn.LSTM(
                input_dim, hidden_size, num_layers,
                batch_first=True, dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional
            )
        elif rnn_type.upper() == 'GRU':
            self.rnn = nn.GRU(
                input_dim, hidden_size, num_layers,
                batch_first=True, dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional
            )
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}. Use 'LSTM' or 'GRU'")
        
        # Attention pooling (optional)
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(hidden_size * num_directions, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1)
            )
        
        # Output layer
        self.decoder = MLP(hidden_size * num_directions, 512, 1)
        
    def forward(self, x, lengths=None):
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, feature_dim] (padded)
            lengths: Optional tensor of shape [batch_size] with actual sequence lengths
        Returns:
            output: Tensor of shape [batch_size, 1] with binary logits
        """
        # Check for NaNs in input and replace them with zero
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Pack sequences if lengths are provided
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        # RNN forward pass
        rnn_out, hidden = self.rnn(x)
        
        # Unpack if we packed
        if lengths is not None:
            rnn_out, _ = nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True)
        
        # Extract representation
        if self.use_attention:
            # Attention pooling
            attention_weights = self.attention(rnn_out)  # [batch, seq_len, 1]
            attention_weights = torch.softmax(attention_weights, dim=1)
            # Mask out padding positions if lengths provided
            if lengths is not None:
                mask = torch.arange(rnn_out.size(1), device=rnn_out.device).unsqueeze(0) < lengths.to(rnn_out.device).unsqueeze(1)
                mask = mask.unsqueeze(-1).float()
                attention_weights = attention_weights * mask
                attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdim=True) + 1e-8)
            representation = (rnn_out * attention_weights).sum(dim=1)  # [batch, hidden_size * num_directions]
        else:
            # Use last hidden state
            if self.rnn_type.upper() == 'LSTM':
                # For LSTM, hidden is a tuple (h_n, c_n)
                h_n = hidden[0]  # [num_layers * num_directions, batch, hidden_size]
            else:
                # For GRU, hidden is just h_n
                h_n = hidden  # [num_layers * num_directions, batch, hidden_size]
            
            # Get the last layer's hidden state from all directions
            if self.bidirectional:
                # Concatenate forward and backward
                forward_hidden = h_n[-2]  # [batch, hidden_size]
                backward_hidden = h_n[-1]  # [batch, hidden_size]
                representation = torch.cat([forward_hidden, backward_hidden], dim=1)
            else:
                representation = h_n[-1]  # [batch, hidden_size]
        
        # Decode to binary logits
        output = self.decoder(representation)
        
        return output
