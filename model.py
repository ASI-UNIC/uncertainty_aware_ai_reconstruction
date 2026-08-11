import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super(RotaryPositionalEmbedding, self).__init__()

        # Create a rotation matrix.
        self.rotation_matrix = torch.zeros(d_model, d_model, device=torch.device("cpu"))
        for i in range(d_model):
            for j in range(d_model):
                self.rotation_matrix[i, j] = torch.cos(torch.Tensor([i * j * 0.01]))

        # Create a positional embedding matrix.
        self.positional_embedding = torch.zeros(
            max_seq_len, d_model, device=torch.device("cpu")
        )
        for i in range(max_seq_len):
            for j in range(d_model):
                self.positional_embedding[i, j] = torch.cos(
                    torch.Tensor([i * j * 0.01])
                )

    def forward(self, x):
        """
        Args:
            x: A tensor of shape (batch_size, seq_len, d_model).

        Returns:
            A tensor of shape (batch_size, seq_len, d_model).
        """

        # Add the positional embedding to the input tensor.
        x += self.positional_embedding

        # Apply the rotation matrix to the input tensor.
        x = torch.matmul(x, self.rotation_matrix)

        return x


class ConditionalTimeSeriesModel(nn.Module):
    """
    Hybrid CNN-LSTM model with perturbation conditioning.

    This model accepts both time series data and perturbation values,
    allowing it to predict outcomes based on different perturbation levels.

    Architecture:
    1. Perturbation embedding layer
    2. 1D Convolutional layers for local feature extraction
    3. LSTM layers for temporal dependency modeling
    4. Attention mechanism with perturbation conditioning
    5. Fully connected layers for prediction

    Args:
        input_dim: Number of features per timestep (default: 2000)
        output_dim: Output dimension (default: 2000 for reconstruction, or custom)
        hidden_dim: Hidden dimension for LSTM (default: 256)
        num_lstm_layers: Number of LSTM layers (default: 2)
        perturbation_embed_dim: Dimension for perturbation embedding (default: 32)
        dropout: Dropout rate (default: 0.3)
        task: 'regression' for predicting next timesteps or 'classification' for case classification
    """

    def __init__(
        self,
        input_dim=2000,
        compressed_dim=512,
        output_dim=2000,
        hidden_dim=256,
        num_lstm_layers=2,
        perturbation_embed_dim=32,
        dropout=0.3,
        task="regression",
    ):
        super(ConditionalTimeSeriesModel, self).__init__()

        self.input_dim = input_dim
        self.compressed_dim = compressed_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.perturbation_embed_dim = perturbation_embed_dim
        self.task = task

        # Perturbation embedding network
        self.perturbation_encoder = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, perturbation_embed_dim),
            nn.ReLU(),
        )

        # Compressed embedding network
        self.compressed_encoder = nn.Sequential(
            nn.Linear(input_dim, compressed_dim),
            nn.ReLU(),
            nn.Linear(compressed_dim, compressed_dim),
            nn.ReLU(),
        )

        # 1D Convolutional layers for feature extraction
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(
            in_channels=64, out_channels=128, kernel_size=5, padding=2
        )
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(
            in_channels=128, out_channels=256, kernel_size=3, padding=1
        )
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        # LSTM layers for temporal modeling
        # Concatenate perturbation embedding with conv features
        self.lstm = nn.LSTM(
            input_size=256 + perturbation_embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
            bidirectional=True,
        )

        # Attention mechanism (conditioned on perturbation)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2 + perturbation_embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        # Output layers
        self.dropout = nn.Dropout(dropout)

        if task == "regression":
            # For predicting time series values
            self.fc1 = nn.Linear(hidden_dim * 2 + perturbation_embed_dim, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, output_dim)
        elif task == "classification":
            # For classifying perturbation cases
            self.fc1 = nn.Linear(hidden_dim * 2 + perturbation_embed_dim, 512)
            self.fc2 = nn.Linear(512, 128)
            self.fc3 = nn.Linear(128, 4)  # 4 perturbation classes

    def forward(self, x, perturbation):
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
               e.g., (batch_size, 100, 2000)
            perturbation: Perturbation value tensor of shape (batch_size, 1)
                         e.g., values like [0.09], [0.130], [0.155], [0.190]

        Returns:
            output: Output tensor
                   - For regression: (batch_size, output_dim)
                   - For classification: (batch_size, num_classes)
        """
        batch_size, seq_len, input_dim = x.shape
        compressed_x = self.compressed_encoder(x)

        # Encode perturbation value
        pert_embed = self.perturbation_encoder(
            perturbation
        )  # (batch, perturbation_embed_dim)

        # Reshape for 1D convolution: (batch*seq_len, 1, input_dim)
        x_conv = compressed_x.view(batch_size * seq_len, 1, self.compressed_dim)

        # Convolutional feature extraction
        x_conv = F.relu(self.bn1(self.conv1(x_conv)))
        x_conv = self.pool1(x_conv)

        x_conv = F.relu(self.bn2(self.conv2(x_conv)))
        x_conv = self.pool2(x_conv)

        x_conv = F.relu(self.bn3(self.conv3(x_conv)))
        x_conv = self.pool3(x_conv)

        # Global average pooling across the spatial dimension
        x_conv = F.adaptive_avg_pool1d(x_conv, 1)  # (batch*seq_len, 256, 1)
        x_conv = x_conv.squeeze(-1)  # (batch*seq_len, 256)

        # Reshape back to sequence: (batch_size, seq_len, 256)
        x_conv = x_conv.view(batch_size, seq_len, -1)

        # Concatenate perturbation embedding with each timestep
        pert_embed_expanded = pert_embed.expand(
            -1, seq_len, -1
        )  # (batch, seq_len, pert_embed_dim)
        x_combined = torch.cat(
            [x_conv, pert_embed_expanded], dim=-1
        )  # (batch, seq_len, 256+pert_embed_dim)

        # LSTM layers
        lstm_out, (h_n, c_n) = self.lstm(
            x_combined
        )  # lstm_out: (batch, seq_len, hidden_dim*2)

        # Attention mechanism (conditioned on perturbation)
        pert_embed_for_attn = pert_embed.expand(-1, seq_len, -1)
        lstm_with_pert = torch.cat([lstm_out, pert_embed_for_attn], dim=-1)
        attention_weights = self.attention(lstm_with_pert)  # (batch, seq_len, 1)
        attention_weights = F.softmax(attention_weights, dim=1)

        # Apply attention weights
        context_vector = torch.sum(
            attention_weights * lstm_out, dim=1
        )  # (batch, hidden_dim*2)

        # Concatenate context with perturbation embedding
        final_features = torch.cat([context_vector, pert_embed], dim=-1)

        # Fully connected layers
        x = self.dropout(final_features)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        output = self.fc3(x)

        return output


class ConditionalLSTMModel(nn.Module):
    """
    Simpler LSTM-based model with perturbation conditioning.

    Args:
        input_dim: Number of features per timestep (default: 2000)
        output_dim: Output dimension (default: 2000)
        hidden_dim: Hidden dimension for LSTM (default: 128)
        num_layers: Number of LSTM layers (default: 2)
        perturbation_embed_dim: Dimension for perturbation embedding (default: 32)
        dropout: Dropout rate (default: 0.3)
        task: 'regression' or 'classification'
    """

    def __init__(
        self,
        input_dim=2000,
        compressed_dim=512,
        output_dim=2000,
        hidden_dim=128,
        num_layers=2,
        perturbation_embed_dim=32,
        dropout=0.3,
        task="regression",
    ):
        super(ConditionalLSTMModel, self).__init__()

        self.task = task
        self.perturbation_embed_dim = perturbation_embed_dim

        # Perturbation embedding
        self.perturbation_encoder = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, perturbation_embed_dim),
            nn.ReLU(),
        )

        # Compressed embedding network
        self.compressed_encoder = nn.Sequential(
            nn.Linear(input_dim, compressed_dim),
            nn.ReLU(),
            nn.Linear(compressed_dim, compressed_dim),
            nn.ReLU(),
        )

        # LSTM with perturbation conditioning
        self.lstm = nn.LSTM(
            input_size=compressed_dim + perturbation_embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout)

        if task == "regression":
            self.fc = nn.Linear(hidden_dim * 2 + perturbation_embed_dim, output_dim)
        elif task == "classification":
            self.fc = nn.Linear(hidden_dim * 2 + perturbation_embed_dim, 4)

    def forward(self, x, perturbation):
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            perturbation: Perturbation value tensor of shape (batch_size, 1)

        Returns:
            output: Output tensor
        """
        batch_size, seq_len, _ = x.shape
        compressed_x = self.compressed_encoder(x)

        # Encode perturbation
        pert_embed = self.perturbation_encoder(perturbation)  # (batch, pert_embed_dim)

        # Expand perturbation embedding for each timestep
        pert_embed_expanded = pert_embed.unsqueeze(1).expand(-1, seq_len, -1)

        # Concatenate with input
        x_combined = torch.cat([compressed_x, pert_embed_expanded], dim=-1)

        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x_combined)

        # Use the last hidden state
        forward_hidden = h_n[-2, :, :]
        backward_hidden = h_n[-1, :, :]
        hidden = torch.cat([forward_hidden, backward_hidden, pert_embed], dim=1)

        # Fully connected
        x = self.dropout(hidden)
        output = self.fc(x)

        return output


class ConditionalTransformerModel(nn.Module):
    """
    Transformer-based model with perturbation conditioning.

    Args:
        input_dim: Number of features per timestep (default: 2000)
        output_dim: Output dimension (default: 2000)
        d_model: Dimension of transformer model (default: 256)
        nhead: Number of attention heads (default: 8)
        num_layers: Number of transformer layers (default: 4)
        dim_feedforward: Dimension of feedforward network (default: 1024)
        perturbation_embed_dim: Dimension for perturbation embedding (default: 32)
        dropout: Dropout rate (default: 0.3)
        task: 'regression' or 'classification'
    """

    def __init__(
        self,
        input_dim=2000,
        hidden_dim=512,
        compressed_dim=512,
        output_dim=2000,
        nhead=8,
        num_layers=4,
        seq_len=512,
        step_size=2,
        normalize=True,
        dim_feedforward=512,
        perturbation_embed_dim=64,
        dropout=0.3,
        task="regression",
    ):
        super(ConditionalTransformerModel, self).__init__()

        self.task = task
        self.d_model = hidden_dim
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.step_size = step_size
        self.normalize = normalize
        # Perturbation embedding
        # self.perturbation_encoder = nn.Sequential(
        #     nn.Linear(1, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, perturbation_embed_dim),
        #     nn.ReLU(),
        # )

        # Compressed embedding network
        self.compressed_encoder = nn.Sequential(
            nn.Linear(input_dim, compressed_dim),
            nn.ReLU(),
            nn.Linear(compressed_dim, compressed_dim),
            nn.ReLU(),
        )

        # Input projection (includes perturbation embedding)
        self.input_projection = nn.Linear(compressed_dim, self.d_model)
        self.rotary_positional_embedding = RotaryPositionalEmbedding(
            d_model=self.d_model, max_seq_len=self.seq_len
        )

        # Positional encoding
        # self.pos_encoder = PositionalEncoding(self.d_model, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Classification/Regression head
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.d_model, self.d_model)

        if task == "regression" or task == "interpolation":
            self.fc2 = nn.Linear(self.d_model, output_dim)
        elif task == "classification":
            self.fc2 = nn.Linear(self.d_model, 4)

    def forward(self, x):
        x = self.compressed_encoder(x)
        x = self.input_projection(x)
        # Transformer expects shape: (seq_len, batch, dim)
        # x = self.transformer_encoder(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer.
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


def get_model(model_type="hybrid", **kwargs):
    """
    Factory function to get a conditional model.

    Args:
        model_type: Type of model ('hybrid', 'lstm', 'transformer')
        **kwargs: Additional arguments for the model

    Returns:
        model: PyTorch model
    """
    if model_type == "hybrid":
        return ConditionalTimeSeriesModel(**kwargs)
    elif model_type == "lstm":
        return ConditionalLSTMModel(**kwargs)
    elif model_type == "transformer":
        return ConditionalTransformerModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test the conditional models
    batch_size = 32
    num_samples = 1000
    seq_len = 8
    input_dim = 400
    output_dim = 400

    # Create dummy input
    x = torch.randn(num_samples, seq_len, input_dim)

    # Train the model
    model = ConditionalTransformerModel(
        input_dim=input_dim,
        output_dim=output_dim,
        seq_len=seq_len,
        task="interpolation",
    )
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Dummy training loop
    for epoch in range(500):
        for i in range(num_samples // batch_size):
            optimizer.zero_grad()
            output = model(x[i * batch_size : (i + 1) * batch_size])
            loss = criterion(output, x[i * batch_size : (i + 1) * batch_size])
            loss.backward()
            optimizer.step()

        # Print gradient norms
        total_norm = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                total_norm += grad_norm**2
                # print(f"  {name:30s} grad_norm = {grad_norm:.6f}")
        total_norm = total_norm**0.5
        print(
            f"Epoch {epoch + 1:3d}: Loss = {loss.item():.6f}, Total Grad Norm = {total_norm:.6f}"
        )

        print(f"Epoch {epoch + 1}: Loss = {loss.item()}")
