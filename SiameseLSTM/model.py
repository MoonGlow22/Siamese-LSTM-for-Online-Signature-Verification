import torch
import torch.nn as nn
import torch.nn.functional as F
from config import TRAINING_CONFIG

class ContrastiveLoss(nn.Module):
    """Contrastive Loss function"""
    
    def __init__(self, margin=1.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, distance, label):
        """
        distance: model output (euclidean distance)
        label: 0 if same person, 1 if different person
        """
        loss = torch.mean(
            (1- label) * torch.pow(distance, 2) + 
            label * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        )
        return loss


class SiameseLSTM(nn.Module):
    """Enhanced Siamese LSTM with bidirectional layers, dropout, and optional attention"""

    def __init__(self, input_size=3, hidden_size=256, num_layers=3, use_attention=False):
        super(SiameseLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_attention = use_attention
        dropout_rate = TRAINING_CONFIG["dropout_rate"]

        # Bidirectional LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            dropout=dropout_rate, batch_first=True, bidirectional=True)

        # Attention parameters
        if self.use_attention:
            self.attention = nn.Linear(hidden_size * 2, 1)

        self.batch_norm = nn.BatchNorm1d(hidden_size * 2)
        
        # Dense layers
        self.fc1 = nn.Linear(hidden_size * 2, 128)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(64, 32)
        
    def apply_attention(self, lstm_output):
        # lstm_output: (batch, seq_len, hidden_size*2)
        attn_weights = F.softmax(self.attention(lstm_output), dim=1)  # (batch, seq_len, 1)
        context = torch.sum(attn_weights * lstm_output, dim=1)  # (batch, hidden_size*2)
        return context
    
    def forward_one(self, x):
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)

        if self.use_attention:
            out = self.apply_attention(lstm_out)
        else:
            out = lstm_out[:, -1, :]  # last timestep

        out = self.batch_norm(out)
        
        out = F.relu(self.fc1(out))
        out = self.dropout1(out)
        out = F.relu(self.fc2(out))
        out = self.dropout2(out)
        out = F.relu(self.fc3(out))
        
        return out

    def forward(self, sig1, sig2):
        embedding1 = self.forward_one(sig1)
        embedding2 = self.forward_one(sig2)
        
        # Normalize embeddings to unit vectors
        embedding1_norm = F.normalize(embedding1, p=2, dim=1)
        embedding2_norm = F.normalize(embedding2, p=2, dim=1)

        # Cosine similarity: dot product of normalized vectors
        cosine_sim = torch.sum(embedding1_norm * embedding2_norm, dim=1, keepdim=True)

        # Convert to cosine distance
        cosine_distance = 1 - cosine_sim  # range: [0, 2]

        return cosine_distance

