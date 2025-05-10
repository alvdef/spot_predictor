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
        "instance_feature_size",
        "time_feature_size",
    ]

    def __init__(self, work_dir: str):
        super().__init__(work_dir)

    def _build_model(self, config: Dict[str, Any]) -> None:
        self.base_hidden_size = config["hidden_size"]
        self.num_layers = config["num_layers"]
        self.prediction_length = config["tr_prediction_length"]
        self.instance_feature_size = config["instance_feature_size"]
        self.time_feature_dim = config[
            "time_feature_size"
        ]  # Matches variable name used later

        self.feature_hidden_size = max(
            self.base_hidden_size, min(128, self.instance_feature_size * 2)
        )
        self.rnn_hidden_size = self.base_hidden_size
        self.attention_hidden_size = self.base_hidden_size // 2
        self.time_embedding_size = self.base_hidden_size // 2

        self.set_derived_param("feature_hidden_size", self.feature_hidden_size)
        self.set_derived_param("rnn_hidden_size", self.rnn_hidden_size)
        self.set_derived_param("attention_hidden_size", self.attention_hidden_size)
        self.set_derived_param("time_embedding_size", self.time_embedding_size)
        # self.set_derived_param("time_feature_dim", self.time_feature_dim) # Already a primary config

        self.instance_feature_net = nn.Sequential(
            nn.Linear(self.instance_feature_size, self.feature_hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.feature_hidden_size, self.rnn_hidden_size),
            nn.LayerNorm(self.rnn_hidden_size),
        )

        self.time_feature_net = nn.Sequential(
            nn.Linear(self.time_feature_dim, self.feature_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.feature_hidden_size // 2, self.time_embedding_size),
            nn.LayerNorm(self.time_embedding_size),
        )

        self.encoder = nn.GRU(
            input_size=1
            + self.time_embedding_size,  # Assuming input sequence value is 1D
            hidden_size=self.rnn_hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=False,
        )

        # Input to this Linear is rnn_hidden_size (from processed instance_embedding)
        self.feature_to_hidden = nn.Linear(
            self.rnn_hidden_size, self.num_layers * self.rnn_hidden_size
        )

        # Attention mechanism
        # Input: encoder_outputs (H_rnn) + decoder_hidden (H_rnn) + instance_features (H_rnn)
        # Total = 3 * H_rnn
        self.attention = nn.Linear(
            self.rnn_hidden_size * 3,
            self.attention_hidden_size,
        )

        self.attention_combine = nn.Linear(
            self.rnn_hidden_size
            * 2,  # prev_output_projected (H_rnn) + context_vector (H_rnn)
            self.rnn_hidden_size,
        )

        self.decoder = nn.GRU(
            input_size=self.rnn_hidden_size,  # Output of attention_combine
            hidden_size=self.rnn_hidden_size,
            num_layers=self.num_layers,
            batch_first=False,  # Expects (seq_len=1, batch, feature)
        )

        self.decoder_input_proj = nn.Linear(
            1, self.rnn_hidden_size
        )  # Projects scalar prev_output/target
        self.fc_out = nn.Linear(
            self.rnn_hidden_size, 1
        )  # Maps decoder GRU output to scalar prediction

        self.to(self.device)

    def _attention_mechanism(
        self, decoder_hidden_all_layers, encoder_outputs, instance_embedding_2d
    ):
        """
        Compute feature-aware attention weights and context vector.
        Assumes instance_embedding_2d is already [batch_size, rnn_hidden_size].
        """
        batch_size, seq_len, _ = encoder_outputs.size()

        # Use the hidden state of the first layer of the decoder for attention
        # decoder_hidden_all_layers shape: (num_layers, batch_size, rnn_hidden_size)
        decoder_hidden_first_layer = decoder_hidden_all_layers[
            0
        ]  # Shape: (batch_size, rnn_hidden_size)

        # Expand decoder hidden state to match encoder output sequence length
        # (batch_size, rnn_hidden_size) -> (batch_size, 1, rnn_hidden_size) -> (batch_size, seq_len, rnn_hidden_size)
        decoder_hidden_expanded = decoder_hidden_first_layer.unsqueeze(1).expand(
            -1, seq_len, -1
        )

        # Expand instance embedding to match encoder output sequence length
        # instance_embedding_2d shape: (batch_size, rnn_hidden_size)
        # (batch_size, rnn_hidden_size) -> (batch_size, 1, rnn_hidden_size) -> (batch_size, seq_len, rnn_hidden_size)
        feature_expanded = instance_embedding_2d.unsqueeze(1).expand(-1, seq_len, -1)

        # Concatenate for attention energy calculation
        # Shapes: encoder_outputs (B,S,H), decoder_hidden_expanded (B,S,H), feature_expanded (B,S,H)
        energy_input = torch.cat(
            (encoder_outputs, decoder_hidden_expanded, feature_expanded), dim=2
        )  # Shape: (batch_size, seq_len, 3 * rnn_hidden_size)

        # Calculate attention energies and weights
        # self.attention is Linear(3 * rnn_hidden_size, attention_hidden_size)
        energy = torch.tanh(
            self.attention(energy_input)
        )  # Shape: (batch_size, seq_len, attention_hidden_size)
        # Sum over the attention_hidden_size dimension to get a score per encoder timestep
        attention_scores = torch.sum(energy, dim=2)  # Shape: (batch_size, seq_len)
        attention_weights = F.softmax(attention_scores, dim=1).unsqueeze(
            2
        )  # Shape: (batch_size, seq_len, 1)

        # Compute context vector
        # encoder_outputs.transpose(1,2) shape: (batch_size, rnn_hidden_size, seq_len)
        # attention_weights shape: (batch_size, seq_len, 1)
        # context_vector shape: (batch_size, rnn_hidden_size, 1) -> then squeezed
        context_vector = torch.bmm(
            encoder_outputs.transpose(1, 2), attention_weights
        ).squeeze(
            2
        )  # Shape: (batch_size, rnn_hidden_size)

        return context_vector, attention_weights

    def forward(
        self,
        x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        teacher_forcing_ratio = self.config["teacher_forcing_ratio"]
        sequence, instance_features, time_features_original_shape = x

        batch_size, seq_len, _ = sequence.size()  # Assuming sequence input_size is 1

        # 1. Process Instance Features
        # Assuming instance_feature_net might produce (B,1,1,H_rnn) based on prior issues,
        # or (B,H_rnn) if instance_features are "cleaner".
        # We ensure it becomes (B, H_rnn).
        raw_instance_embedding = self.instance_feature_net(instance_features)

        # Reshape raw_instance_embedding to (batch_size, rnn_hidden_size)
        # This handles cases like (B,H), (B,1,H), (B,1,1,H) etc., as long as the
        # total number of elements per batch item is rnn_hidden_size.
        instance_embedding_2d = raw_instance_embedding.reshape(
            batch_size, self.rnn_hidden_size
        )

        # Create initial hidden state for encoder GRU from instance features
        # feature_to_hidden input: (B, rnn_hidden_size)
        # flat_initial_hidden output: (B, num_layers * rnn_hidden_size)
        flat_initial_hidden = self.feature_to_hidden(instance_embedding_2d)
        # Reshape to (num_layers, batch_size, rnn_hidden_size) for GRU
        encoder_initial_hidden = (
            flat_initial_hidden.view(batch_size, self.num_layers, self.rnn_hidden_size)
            .permute(1, 0, 2)
            .contiguous()
        )

        # 2. Process Time Features
        # time_features_original_shape: (batch_size, time_feature_dim, seq_len)
        time_features_transposed = time_features_original_shape.transpose(
            1, 2
        )  # (B, seq_len, time_feature_dim)

        # Flatten for time_feature_net: (B * seq_len, time_feature_dim)
        reshaped_time_features = time_features_transposed.reshape(
            -1, self.time_feature_dim
        )
        processed_time_features_flat = self.time_feature_net(
            reshaped_time_features
        )  # (B*S, time_embedding_size)

        # Reshape back to (batch_size, seq_len, time_embedding_size)
        time_embedding = processed_time_features_flat.view(
            batch_size, seq_len, self.time_embedding_size
        )

        # Concatenate sequence values with time embeddings for encoder input
        sequence_with_time_embedding = torch.cat([sequence, time_embedding], dim=2)

        # 3. Encode
        encoder_outputs, encoder_last_hidden = self.encoder(
            sequence_with_time_embedding, encoder_initial_hidden
        )  # encoder_outputs: (B,S,H_rnn), encoder_last_hidden: (L,B,H_rnn)

        # 4. Decode
        decoder_hidden = encoder_last_hidden  # Initial hidden state for decoder

        # Initial projected input for the decoder (based on a zero value)
        # This represents the embedding of y_{t-1} for the first step.
        # Shape: (1, batch_size, rnn_hidden_size)
        projected_prev_output = self.decoder_input_proj(
            torch.zeros(batch_size, 1, device=self.device)
        )  # (B, rnn_hidden_size)

        predictions = torch.zeros(
            batch_size, self.prediction_length, device=self.device
        )

        for t in range(self.prediction_length):
            # Attention: decoder_hidden is (L,B,H), encoder_outputs is (B,S,H), instance_embedding_2d is (B,H)
            (
                context_vector,
                _,
            ) = self._attention_mechanism(  # We don't need attention_weights here
                decoder_hidden, encoder_outputs, instance_embedding_2d
            )  # context_vector: (B, rnn_hidden_size)

            # Combine projected previous output and current context vector
            # projected_prev_output: (B, rnn_hidden_size), context_vector: (B, rnn_hidden_size)
            combined_input_for_gru_proj = torch.cat(
                (projected_prev_output, context_vector), dim=1
            )  # (B, 2 * rnn_hidden_size)

            # Project this combination to the decoder GRU's expected input size
            # self.attention_combine: Linear(2*H_rnn, H_rnn)
            decoder_gru_input = self.attention_combine(
                combined_input_for_gru_proj
            ).unsqueeze(
                0
            )  # Shape: (1, batch_size, rnn_hidden_size) for GRU

            # Decoder GRU step
            # decoder_gru_input: (1,B,H_rnn), decoder_hidden: (L,B,H_rnn)
            decoder_output, decoder_hidden = self.decoder(
                decoder_gru_input, decoder_hidden
            )  # decoder_output: (1,B,H_rnn), decoder_hidden: (L,B,H_rnn)

            # Generate scalar prediction for the current timestep
            # self.fc_out: Linear(H_rnn, 1)
            # decoder_output.squeeze(0): (B, H_rnn)
            current_prediction_scalar = self.fc_out(decoder_output.squeeze(0))  # (B, 1)
            predictions[:, t] = current_prediction_scalar.squeeze(1)

            # Prepare for next iteration: select value for y_t (teacher forcing or model's prediction)
            use_teacher_forcing = False
            if (
                target is not None and t < self.prediction_length - 1
            ):  # Only apply for relevant steps
                use_teacher_forcing = random.random() < teacher_forcing_ratio

            if use_teacher_forcing:
                next_scalar_input = target[:, t].unsqueeze(1)  # (B, 1)
            else:
                next_scalar_input = current_prediction_scalar.detach()  # (B, 1)

            # Project this scalar value for the next iteration's projected_prev_output
            projected_prev_output = self.decoder_input_proj(
                next_scalar_input
            )  # (B, rnn_hidden_size)

        return predictions
