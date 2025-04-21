from typing import Dict, Any, Optional, Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from .base import Model


class Seq2Seq(Model):
    REQUIRED_FIELDS = [
        "hidden_size",
        "input_size",
        "num_layers",
        "tr_prediction_length",
        "teacher_forcing_ratio",
    ]

    def __init__(self, work_dir: str):
        super().__init__(work_dir)

    def _build_model(self, config: Dict[str, Any]) -> None:
        """
        Build the seq2seq model with attention mechanism.

        Args:
            config: Dictionary containing model configuration parameters.
        """
        # Extract configuration parameters
        self.base_hidden_size = config["hidden_size"]
        self.input_size = config["input_size"]
        self.num_layers = config["num_layers"]
        self.prediction_length = config["tr_prediction_length"]
        
        self.rnn_hidden_size = self.base_hidden_size
        self.attention_hidden_size = self.base_hidden_size // 2

        # Define encoder (unidirectional for simplicity)
        self.encoder = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.rnn_hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=False,  # Changed to unidirectional
        )

        # Attention mechanism for unidirectional encoder
        self.attention = nn.Linear(
            self.rnn_hidden_size * 2,  # encoder, decoder
            self.attention_hidden_size,
        )

        self.attention_combine = nn.Linear(
            self.rnn_hidden_size * 2,
            self.rnn_hidden_size,
        )

        # Define decoder (unidirectional)
        self.decoder = nn.GRU(
            input_size=self.rnn_hidden_size,
            hidden_size=self.rnn_hidden_size,
            num_layers=self.num_layers,
            batch_first=False,
        )

        # Define output layers
        self.fc_out = nn.Linear(self.rnn_hidden_size, 1)

        self._initialize_weights()
        self.to(self.device)

    def _initialize_weights(self) -> None:
        """
        Initialize model weights for faster convergence.
        Uses Xavier/Glorot initialization for linear layers,
        orthogonal initialization for recurrent layers, and zeros for biases.
        """
        for name, param in self.named_parameters():
            if "weight" in name:
                if isinstance(param, nn.Parameter) and isinstance(
                    param.data, torch.Tensor
                ):
                    if isinstance(self.encoder, nn.GRU) or isinstance(
                        self.decoder, nn.GRU
                    ):
                        nn.init.orthogonal_(param)
                    elif isinstance(param, nn.Linear):
                        nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def _attention_mechanism(self, decoder_hidden, encoder_outputs):
        """
        Compute attention weights and context vector.

        Args:
            decoder_hidden: Current decoder hidden state
            encoder_outputs: All encoder outputs

        Returns:
            context_vector: Context vector for the current timestep
            attention_weights: Attention weights for visualization
        """
        seq_len = encoder_outputs.size(1)
        batch_size = encoder_outputs.size(0)

        # Reshape the decoder hidden state correctly
        # Take just the first layer of the decoder hidden state, but keep all features
        decoder_hidden_for_attn = decoder_hidden[0:1]
        
        # Use expand instead of repeat for better memory efficiency
        decoder_hidden_expanded = decoder_hidden_for_attn.permute(1, 0, 2).expand(-1, seq_len, -1)

        # Concatenate encoder outputs and decoder hidden state
        energy_input = torch.cat((encoder_outputs, decoder_hidden_expanded), dim=2)

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
        Forward pass through the model with optional teacher forcing.

        Args:
            x: Input sequence of shape (batch_size, sequence_length, features)
            target: Target sequence for teacher forcing of shape (batch_size, prediction_length)

        Returns:
            Predictions of shape (batch_size, prediction_length)
        """
        sequence = x[0]
        teacher_forcing_ratio = self.config["teacher_forcing_ratio"]
        batch_size = sequence.size(0)

        # Encode the input sequence
        encoder_outputs, encoder_hidden = self.encoder(sequence)

        # For unidirectional GRU, we can directly use the encoder hidden state for the decoder
        decoder_hidden = encoder_hidden

        # Initialize decoder input
        decoder_input = torch.zeros(1, batch_size, self.rnn_hidden_size, device=self.device)

        # Store predictions
        predictions = torch.zeros(
            batch_size, self.prediction_length, device=self.device
        )
        
        # Pre-allocate tensor for teacher forcing to avoid repeated allocations
        zero_input = torch.zeros(1, batch_size, self.rnn_hidden_size, device=self.device)

        # Decode sequence
        for t in range(self.prediction_length):
            # Calculate attention weights and context vector
            context_vector, attention_weights = self._attention_mechanism(
                decoder_hidden, encoder_outputs
            )

            # Combine context with decoder input
            decoder_input_combined = torch.cat(
                (decoder_input.squeeze(0), context_vector), dim=1
            )
            decoder_input_combined = self.attention_combine(
                decoder_input_combined
            ).unsqueeze(0)

            # Decoder forward pass
            decoder_output, decoder_hidden = self.decoder(
                decoder_input_combined, decoder_hidden
            )

            # Generate prediction for current timestep
            output = self.fc_out(decoder_output.squeeze(0))
            predictions[:, t] = output.squeeze(1)

            # Teacher forcing: decide whether to use real target or prediction
            if target is not None and t < self.prediction_length - 1:
                use_teacher_forcing = random.random() < teacher_forcing_ratio
                if use_teacher_forcing:
                    # Convert target value to appropriate decoder input format
                    next_input = target[:, t].unsqueeze(1).unsqueeze(0)
                    # Use pre-allocated tensor and in-place operations for better efficiency
                    zero_input.zero_()
                    decoder_input = zero_input.scatter_(2, next_input.long(), 1.0)
                else:
                    # Use prediction as next input
                    decoder_input = decoder_output
            else:
                # Use prediction as next input
                decoder_input = decoder_output

        return predictions
