from typing import Dict, Any, Optional, Union, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from .base import Model


class FeatureSeq2Seq(Model):
    REQUIRED_FIELDS = [
        "hidden_size",
        "num_layers",
        "tr_prediction_length",
        "teacher_forcing_ratio",
    ]

    def __init__(self, work_dir: str):
        super().__init__(work_dir)

    def _build_model(self, config: Dict[str, Any]) -> None:
        """
        Build the feature-enhanced seq2seq model with architecture that scales with feature sizes.

        Uses separate hidden sizes for feature networks, encoder/decoder, and attention.

        Args:
            config: Dictionary containing model configuration parameters.
        """
        # Base configuration parameters
        self.base_hidden_size = config["hidden_size"]
        self.num_layers = config["num_layers"]
        self.prediction_length = config["tr_prediction_length"]
        self.instance_feature_size = config["instance_feature_size"]
        self.time_feature_dim = config["time_feature_size"]

        # Derive component-specific hidden sizes
        # Scale feature networks based on input feature sizes
        self.feature_hidden_size = max(
            self.base_hidden_size, min(128, self.instance_feature_size * 2)
        )
        self.rnn_hidden_size = self.base_hidden_size
        self.attention_hidden_size = self.base_hidden_size // 2
        self.time_embedding_size = self.base_hidden_size // 2

        # Register derived parameters for logging
        self.set_derived_param("feature_hidden_size", self.feature_hidden_size)
        self.set_derived_param("rnn_hidden_size", self.rnn_hidden_size)
        self.set_derived_param("attention_hidden_size", self.attention_hidden_size)
        self.set_derived_param("time_embedding_size", self.time_embedding_size)
        self.set_derived_param("time_feature_dim", self.time_feature_dim)

        # Instance feature processing network
        self.instance_feature_net = nn.Sequential(
            nn.Linear(self.instance_feature_size, self.feature_hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.feature_hidden_size, self.rnn_hidden_size),
            nn.LayerNorm(self.rnn_hidden_size),
        )

        # Time feature processing network - updated to process the actual input size
        self.time_feature_net = nn.Sequential(
            nn.Linear(self.time_feature_dim, self.feature_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.feature_hidden_size // 2, self.time_embedding_size),
            nn.LayerNorm(self.time_embedding_size),
        )

        # Define encoder
        self.encoder = nn.GRU(
            input_size=1 + self.time_embedding_size,
            hidden_size=self.rnn_hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=False,
        )

        # Feature-aware initial hidden state generator
        self.feature_to_hidden = nn.Linear(
            self.rnn_hidden_size, self.num_layers * self.rnn_hidden_size
        )

        # Combined features fusion layer
        self.feature_fusion = nn.Sequential(
            nn.Linear(
                self.rnn_hidden_size + self.time_embedding_size, self.rnn_hidden_size
            ),
            nn.ReLU(),
            nn.LayerNorm(self.rnn_hidden_size),
        )

        # Attention mechanism with separate dimension
        self.attention = nn.Linear(
            self.rnn_hidden_size * 2
            + self.rnn_hidden_size,  # encoder, decoder, instance features
            self.attention_hidden_size,
        )

        self.attention_combine = nn.Linear(
            self.rnn_hidden_size * 2,
            self.rnn_hidden_size,
        )

        # Define decoder
        self.decoder = nn.GRU(
            input_size=self.rnn_hidden_size,
            hidden_size=self.rnn_hidden_size,
            num_layers=self.num_layers,
            batch_first=False,
        )

        # Add decoder input projection - this was missing
        self.decoder_input_proj = nn.Linear(1, self.rnn_hidden_size)

        # Output layer
        self.fc_out = nn.Linear(self.rnn_hidden_size, 1)

        self._initialize_weights()
        self.to(self.device)

    def _initialize_weights(self) -> None:
        """
        Initialize model weights for faster convergence and training stability.

        Different initialization techniques are used based on layer type:
        - Orthogonal for RNN weights: Helps maintain gradient magnitudes through time steps. Prevents vanishing/exploding gradients.
        - Xavier/Glorot for linear layers: Maintains variance across network depth. Keeps activations in good range across layers.
        - Small uniform values for 1D weights: Prevents initial large values, work well as starting point for 1D weights.
        - Zeros for biases: Allows the network to learn biases from scratch, without initial preference.
        """
        for name, param in self.named_parameters():
            # Skip parameters without gradients
            if not param.requires_grad:
                continue

            # Check parameter dimensions
            if len(param.shape) >= 2:  # For parameters with 2 or more dimensions
                if (
                    "gru" in name.lower()
                    or "lstm" in name.lower()
                    or "rnn" in name.lower()
                ):
                    # Orthogonal initialization for recurrent layers - helps with gradient flow
                    nn.init.orthogonal_(param)
                else:
                    # Xavier initialization for fully connected layers - balances activations
                    nn.init.xavier_uniform_(param)
            elif len(param.shape) == 1:  # For 1D parameters (biases, etc.)
                if "bias" in name:
                    # Zero initialization for biases
                    nn.init.zeros_(param)
                else:
                    # Small uniform values for other 1D parameters
                    nn.init.uniform_(param, -0.1, 0.1)

    def _attention_mechanism(self, decoder_hidden, encoder_outputs, instance_embedding):
        """
        Compute feature-aware attention weights and context vector.

        Args:
            decoder_hidden: Current decoder hidden state
            encoder_outputs: All encoder outputs
            instance_embedding: Instance feature embedding to condition attention

        Returns:
            context_vector: Context vector for the current timestep
            attention_weights: Attention weights for visualization
        """
        seq_len = encoder_outputs.size(1)
        batch_size = encoder_outputs.size(0)

        # Reshape the decoder hidden state correctly
        decoder_hidden_for_attn = decoder_hidden[0:1]
        decoder_hidden_expanded = decoder_hidden_for_attn.permute(1, 0, 2).repeat(
            1, seq_len, 1
        )

        # Ensure instance_embedding has shape [batch_size, hidden_size]
        if len(instance_embedding.shape) == 3 and instance_embedding.shape[1] == 1:
            # Handle [batch_size, 1, hidden_size] case by removing the middle dimension
            instance_embedding = instance_embedding.squeeze(1)

        # Expand to [batch_size, seq_len, hidden_size] for attention computation
        # Use expand instead of repeat for better memory efficiency
        feature_expanded = instance_embedding.unsqueeze(1).expand(-1, seq_len, -1)

        # Concatenate encoder outputs, decoder hidden state, and instance features
        energy_input = torch.cat(
            (encoder_outputs, decoder_hidden_expanded, feature_expanded), dim=2
        )

        # Calculate attention energies
        energy = torch.tanh(self.attention(energy_input))
        attention_weights = F.softmax(torch.sum(energy, dim=2), dim=1).unsqueeze(2)

        # Compute context vector
        context_vector = torch.bmm(
            encoder_outputs.transpose(1, 2), attention_weights
        ).squeeze(2)

        return context_vector, attention_weights

    def forward(
        self,
        x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the model with feature conditioning and optional teacher forcing.

        Args:
            x: Tuple of (sequence, instance_features, time_features) where:
               - sequence: tensor of shape (batch_size, sequence_length, input_size)
               - instance_features: tensor of shape (batch_size, feature_size)
               - time_features: tensor of shape (batch_size, time_feature_size, sequence_length)
            target: Target sequence for teacher forcing of shape (batch_size, prediction_length)

        Returns:
            Predictions of shape (batch_size, prediction_length)
        """
        teacher_forcing_ratio = self.config["teacher_forcing_ratio"]
        sequence, instance_features, time_features = x

        batch_size, seq_len, input_size = sequence.size()

        # Process instance features
        instance_embedding = self.instance_feature_net(instance_features)

        feature_hidden = self.feature_to_hidden(
            instance_embedding
        )  # Transforms raw instance features into a dense embedding
        feature_hidden = feature_hidden.view(
            self.num_layers, batch_size, self.rnn_hidden_size
        )  # Reshaped to match the GRU's expected dimensions

        # Handle time features with correct reshaping
        # The time_features shape from dataset is [batch_size, time_feature_size, seq_len]
        # We need to transpose to get [batch_size, seq_len, time_feature_size]
        time_features = time_features.transpose(1, 2)
        time_feat_size = time_features.size(2)

        # Flatten for processing
        reshaped_time_features = time_features.reshape(-1, time_feat_size)
        processed_time_features = self.time_feature_net(reshaped_time_features)

        # Reshape back to [batch_size, seq_len, time_embedding_size]
        processed_time_features = processed_time_features.view(
            batch_size, seq_len, self.time_embedding_size
        )

        # Concatenate processed time features with input sequence
        sequence_with_time = torch.cat([sequence, processed_time_features], dim=2)

        # Encode the input sequence with feature-conditioned initial state
        encoder_outputs, encoder_hidden = self.encoder(
            sequence_with_time, feature_hidden
        )

        # For unidirectional GRU, use the encoder hidden state for the decoder
        decoder_hidden = encoder_hidden

        # Initialize decoder input - Start with a zero vector representing the previous step's output value projection
        # Shape (batch_size, 1) -> projected to (batch_size, rnn_hidden_size) -> unsqueezed to (1, batch_size, rnn_hidden_size)
        # We start with a scalar 0 projected.
        decoder_input = self.decoder_input_proj(
            torch.zeros(batch_size, 1, device=self.device)
        ).unsqueeze(0)

        predictions = torch.zeros(
            batch_size, self.prediction_length, device=self.device
        )

        # Decode sequence
        for t in range(self.prediction_length):
            # Calculate attention weights and context vector using instance features
            context_vector, attention_weights = self._attention_mechanism(
                decoder_hidden, encoder_outputs, instance_embedding
            )

            # Combine the projected previous output/target with the context vector
            decoder_input_combined = torch.cat(
                (decoder_input.squeeze(0), context_vector), dim=1
            )

            # Project combined input to decoder GRU input space
            decoder_input_combined = self.attention_combine(
                decoder_input_combined
            ).unsqueeze(
                0
            )  # Shape: (1, batch_size, rnn_hidden_size)

            # Decoder forward pass
            # Input to GRU is the combined context and projected previous step info
            decoder_output, decoder_hidden = self.decoder(
                decoder_input_combined, decoder_hidden
            )  # decoder_output shape: (1, batch_size, rnn_hidden_size)

            # Generate prediction for current timestep from the GRU output
            output = self.fc_out(decoder_output.squeeze(0))  # Shape: (batch_size, 1)
            predictions[:, t] = output.squeeze(1)

            use_teacher_forcing = False
            if target is not None and t < self.prediction_length - 1:
                use_teacher_forcing = random.random() < teacher_forcing_ratio

            if use_teacher_forcing and target is not None:
                next_val = target[:, t].unsqueeze(1)  # Shape: (batch_size, 1)
            else:
                # Use the model's own prediction from the current step as input for the next step
                # Detach to prevent gradients from flowing back through this path during prediction use
                next_val = output.detach()  # Shape: (batch_size, 1)

            # Project the chosen value (target or prediction) to the required dimension for the next decoder input
            decoder_input = self.decoder_input_proj(next_val).unsqueeze(
                0
            )  # Shape: (1, batch_size, rnn_hidden_size)

        return predictions
