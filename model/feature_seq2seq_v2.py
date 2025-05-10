from typing import Dict, Any, Optional, Union, Tuple, List
import torch
import torch.nn as nn

from .base import Model


class FeatureSeq2Seq_v2(Model):
    """
    An improved sequence-to-sequence model that explicitly incorporates:
    1. Instance-level features (static attributes of each entity)
    2. Historical values (the actual sequence of previous values)

    The model uses these two sources of information to predict future values.
    """

    REQUIRED_FIELDS = [
        "hidden_size",
        "num_layers",
        "tr_prediction_length",
        "instance_feature_size",
        "dropout_rate",
    ]

    def __init__(self, work_dir: str):
        super().__init__(work_dir)

    def _build_model(self, config: Dict[str, Any]) -> None:
        """
        Build the model components according to configuration.

        The architecture separates processing for:
          - Instance features (static information per entity)
          - Value sequence (historical values)

        These are combined in a principled way for prediction.
        """
        # Extract config parameters
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_layers"]
        self.prediction_length = config["tr_prediction_length"]
        self.instance_feature_size = config["instance_feature_size"]
        self.dropout_rate = config.get("dropout_rate", 0.2)

        # Save derived parameters for documentation
        self.set_derived_param("encoder_input_size", 1)  # Raw value only

        # === Feature Processing Networks ===

        # Instance Feature Network - processes static features of each entity
        self.instance_feature_net = nn.Sequential(
            nn.Linear(self.instance_feature_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
        )

        # === Sequence Processing Components ===

        # Encoder takes raw sequence values as input
        self.encoder = nn.GRU(
            input_size=1,  # Raw value only
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout_rate if self.num_layers > 1 else 0,
        )

        # === Integration Components ===

        # Context Integration (combines instance features with encoded sequence)
        self.context_integration = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
        )

        # === Attention Mechanism ===

        # Multi-head attention for focusing on relevant parts of input sequence
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=4,
            dropout=self.dropout_rate,
            batch_first=True,
        )

        # === Decoder Components ===

        # Decoder processes the integrated context to generate predictions
        self.decoder = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout_rate if self.num_layers > 1 else 0,
        )

        # Output projection to convert hidden states to predictions
        self.output_projection = nn.Linear(self.hidden_size, 1)

        # Move model to appropriate device
        self.to(self.device)

    def forward(
        self,
        x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x: Tuple containing (sequence, instance_features)
                - sequence: Historical values [batch_size, seq_len, 1]
                - instance_features: Entity attributes [batch_size, instance_feature_size]
            target: Optional target values for teacher forcing (not used in this implementation)

        Returns:
            Predictions tensor [batch_size, prediction_length]
        """
        # Unpack input tuple
        sequence, instance_features, _ = x
        batch_size, seq_len, _ = sequence.size()

        # === Step 1: Process instance features (static entity attributes) ===
        # Transform to [batch_size, hidden_size]
        instance_embedding = self.instance_feature_net(instance_features.squeeze(1))

        # === Step 2: Encode sequence ===
        # Pass through encoder to get outputs and final hidden state
        encoder_outputs, encoder_hidden = self.encoder(sequence)

        # === Step 3: Integrate instance context ===
        # Expand instance embedding to match encoder output's time dimension
        instance_context = instance_embedding.unsqueeze(1).expand(-1, seq_len, -1)

        # Concatenate instance embeddings with encoder outputs along feature dimension
        combined_context = torch.cat([encoder_outputs, instance_context], dim=2)

        # Process combined context
        integrated_context = self.context_integration(combined_context)

        # === Step 4: Apply attention mechanism ===
        # Self-attention to focus on relevant parts of the sequence
        attn_output, _ = self.attention(
            integrated_context, integrated_context, integrated_context
        )

        # === Step 5: Initialize decoder hidden state ===
        # Use encoder's final hidden state
        decoder_hidden = encoder_hidden

        # === Step 6: Prepare decoder input ===
        # We use the attended context from the last timestep as initial input
        decoder_input = attn_output[:, -1:, :]  # Shape: [batch_size, 1, hidden_size]

        # === Step 7: Generate future predictions ===
        # Container for predictions
        predictions = torch.zeros(
            batch_size, self.prediction_length, device=self.device
        )

        # Autoregressive decoding loop
        for t in range(self.prediction_length):
            # Run one step of the decoder
            output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

            # Project to scalar prediction
            pred = self.output_projection(output.squeeze(1))
            predictions[:, t] = pred.squeeze(-1)

            # Prepare input for next timestep (use current output)
            decoder_input = output

        return predictions
