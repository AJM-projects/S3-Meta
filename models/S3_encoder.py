# variational_mamba_encoder.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Type

import torch
import torch.nn as nn


@dataclass
class S3BlockConfig:
    """
    Configuration container for a Mamba block.

    :param d_model: Dimension of the model (input and output features).
    :param d_state: Dimension of the internal state per channel. Larger values allow the SSM to capture longer dependencies.
    :param d_conv: Width of the optional depthwise convolution. Set to zero to disable.
    :param expand: Expansion factor for intermediate projections. Intermediate dimension becomes ``expand * d_model``.
    :param include_conv: Whether to include the depthwise convolution. Default is True.
    :param layer_norm: If True, applies a LayerNorm to the input before the block.
    :param residual: If True, adds a skip connection from the input to the output.
    """

    d_model: int
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    include_conv: bool = True
    layer_norm: bool = True
    residual: bool = True


class S3Block(nn.Module):
    """
    Selective state‑space model block implementing the Mamba architecture.

    This block maintains a per‑channel hidden state of shape ``(B, d_model, d_state)``
    and processes an input sequence ``x`` of shape ``(L, B, d_model)``.
    """

    def __init__(self, config: S3BlockConfig) -> None:
        """
        Initialise the S3Block.

        :param config: Configuration specifying dimensions and optional components.
        """
        super().__init__()
        self.config = config

        d_model = config.d_model
        expand_dim = config.expand * d_model

        # normalisation layer
        self.norm = nn.LayerNorm(d_model) if config.layer_norm else None

        # depthwise convolution
        self.include_conv = config.include_conv and config.d_conv > 0
        if self.include_conv:
            self.conv = nn.Conv1d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=config.d_conv,
                padding=config.d_conv // 2,
                groups=d_model,
            )

        # projections for state update
        # combined in one linear for efficiency: out_dim = expand_dim * 2 (u, v) + d_model (dt)
        self.in_proj = nn.Linear(d_model, expand_dim * 2 + d_model)
        # state projection matrices (B1, C, D) and dt multiplier
        self.B1 = nn.Parameter(torch.randn(config.d_state))
        # projection from hidden state to output
        self.C = nn.Linear(config.d_state, 1, bias=False)
        # skip connection
        self.D = nn.Linear(d_model, d_model, bias=False)

        # register state buffer; will be initialised on first forward call
        self.register_buffer("_state", None, persistent=False)

    def reset(self) -> None:
        """
        Reset the internal state to ``None``.

        Calling this method before processing an independent sequence ensures
        that hidden states from previous sequences are discarded.
        """
        self._state = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process a sequence through the Mamba block.

        :param x: Input of shape ``(L, B, d_model)``.
        :return: Output of shape ``(L, B, d_model)``.
        """
        L, B, D = x.shape
        assert D == self.config.d_model, (
            f"Expected input dimension {self.config.d_model}, got {D}"
        )

        # Apply optional layer norm
        if self.norm is not None:
            x = self.norm(x)

        # Initialise hidden state on first call: shape (B, D, d_state)
        if self._state is None:
            self._state = torch.zeros(
                B,
                self.config.d_model,
                self.config.d_state,
                dtype=x.dtype,
                device=x.device,
            )

        # Optionally apply depthwise conv: reshape to (B, D, L) for Conv1d
        if self.include_conv:
            # (L, B, D) -> (B, D, L)
            x_conv = (
                self.conv(x.transpose(0, 1).transpose(1, 2))
                .transpose(1, 2)
                .transpose(0, 1)
            )
            x = x + x_conv

        # Run through sequence
        outputs: list[torch.Tensor] = []
        state = self._state  # shape (B, D, d_state)
        for t in range(L):
            x_t = x[t]  # (B, D)

            # in_proj yields (B, expand*2*D + D)
            proj = self.in_proj(x_t)
            u, v, dt = torch.split(
                proj,
                [self.config.expand * D, self.config.expand * D, D],
                dim=-1,
            )
            # Reshape u and v to (B, D, expand)
            u = u.view(B, D, self.config.expand)
            v = v.view(B, D, self.config.expand)

            # Compute discretisation parameter; ensure positivity via sigmoid
            dt = torch.sigmoid(dt).unsqueeze(-1)  # (B, D, 1)

            # Broadcast B1 and compute input-to-state projection
            B1 = self.B1.view(1, 1, -1)  # (1,1,d_state)
            u_state = torch.sum(u, dim=-1, keepdim=True)  # (B, D, 1)
            # update hidden state: (1 - dt) * h + dt * (u_state + B1 * x_t)
            h_t = (1.0 - dt) * state + dt * (u_state + B1 * x_t.unsqueeze(-1))

            # compute output: (v * (C @ h_t)) + D @ x_t
            # C @ h_t: apply linear across d_state -> 1, so result shape (B, D, 1)
            c_h = self.C(h_t).squeeze(-1)  # (B, D)
            # gating via v: sum over expansion dimension
            v_gate = torch.sum(v, dim=-1)  # (B, D)
            y_t = v_gate * c_h + self.D(x_t)  # (B, D)

            outputs.append(y_t.unsqueeze(0))
            state = h_t  # update state

        # detach state to avoid BPTT across sequences
        self._state = state.detach()
        return torch.cat(outputs, dim=0)  # (L, B, D)


class S3Encoder(nn.Module):
    """
    A stack of Mamba blocks with optional residual connections.
    """

    def __init__(
        self,
        num_layers: int,
        config: S3BlockConfig,
        block_class: Type[nn.Module] = S3Block,
    ) -> None:
        """
        Initialise the S3Encoder.

        :param num_layers: Number of blocks to stack.
        :param config: Configuration for each block.
        :param block_class: The block class to instantiate. Defaults to S3Block.
        """
        super().__init__()
        self.layers = nn.ModuleList(block_class(config) for _ in range(num_layers))
        self.config = config

    def reset(self) -> None:
        """Reset all internal states in the stacked blocks."""
        for layer in self.layers:
            if hasattr(layer, "reset"):
                layer.reset()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process input through the stack of Mamba blocks.

        :param x: Input of shape ``(L, B, d_model)``.
        :return: Output of shape ``(L, B, d_model)``.
        """
        for layer in self.layers:
            x_res = x
            x = layer(x)
            # residual connection if configured
            if self.config.residual:
                x = x + x_res
        return x


class VariationalS3Encoder(nn.Module):
    """
    Variational encoder using a Mamba stack to infer latent MDP parameters.
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        latent_dim: int,
        encoder: S3Encoder,
    ) -> None:
        """
        Initialise the VariationalS3Encoder.

        :param input_dim: Dimension of the input features (concatenated action, state and reward).
        :param embed_dim: Dimension of the embedding used as input to the Mamba encoder.
        :param latent_dim: Dimension of the latent variable representing the MDP.
        :param encoder: Instance of a Mamba encoder that processes embedded sequences.
        """
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.encoder = encoder

        # input projection
        self.embed = nn.Linear(input_dim, embed_dim)
        # output projections for mean and log‑variance
        self.fc_mu = nn.Linear(embed_dim, latent_dim)
        self.fc_logvar = nn.Linear(embed_dim, latent_dim)

    def reset(self) -> None:
        """Reset the internal states of the underlying Mamba encoder."""
        self.encoder.reset()

    def _encode_sequence(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Helper to embed and encode a sequence.

        :param x: Input sequence of shape ``(L, B, input_dim)``.
        :return: A tuple containing the latent mean and log‑variance of shape ``(L, B, latent_dim)``.
        """
        # project to embed_dim
        h = self.embed(x)  # (L, B, embed_dim)
        # process through Mamba stack
        h = self.encoder(h)  # (L, B, embed_dim)
        # map to latent parameters
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def _sample_gaussian(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Draw samples from a Gaussian using the reparameterisation trick.

        :param mu: Mean of the Gaussian.
        :param logvar: Log-variance of the Gaussian.
        :return: A sampled tensor.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def prior(
        self,
        batch_size: int,
        sample: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        """
        Compute the prior latent distribution before seeing any data.

        :param batch_size: Number of parallel latent priors to generate.
        :param sample: Whether to draw a sample from the prior distribution. Default is True.
        :return: A tuple ``(z, mu, logvar, None)`` where each tensor has shape ``(1, B, latent_dim)``.
        """
        # reset encoder to guarantee clean state
        self.reset()
        zeros = torch.zeros(
            1,
            batch_size,
            self.input_dim,
            dtype=next(self.parameters()).dtype,
            device=next(self.parameters()).device,
        )
        mu, logvar = self._encode_sequence(zeros)  # (1, B, latent_dim)
        z = self._sample_gaussian(mu, logvar) if sample else mu
        # reset again so prior call does not affect future calls
        self.reset()
        return z, mu, logvar, None

    def forward(
        self,
        actions,
        states,
        rewards,
        hidden_state=None,
        return_prior=False,
        sample=True,
        detach_every=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        """
        Encode a sequence of (action, state, reward) tuples to latent codes.

        :param actions: Sequence of actions taken.
        :param states: Sequence of states visited.
        :param rewards: Sequence of rewards received.
        :param hidden_state: Unused (maintained for API compatibility).
        :param return_prior: If True, prepend the prior distribution to the outputs. Default is False.
        :param sample: If True, return samples from the posterior. If False, return means. Default is True.
        :param detach_every: Optional frequency to detach gradients for long sequences.
        :return: A tuple ``(z, mu, logvar, None)`` containing the latents.
        """

        actions_states_rewards = torch.cat((actions, states, rewards), dim=-1)
        L, B, _ = actions_states_rewards.shape
        if return_prior:
            z0, mu0, logvar0, _ = self.prior(batch_size=B, sample=sample)

        # encode posterior
        mu, logvar = self._encode_sequence(actions_states_rewards)
        z = self._sample_gaussian(mu, logvar) if sample else mu

        if return_prior:
            mu = torch.cat([mu0, mu], dim=0)
            logvar = torch.cat([logvar0, logvar], dim=0)
            z = torch.cat([z0, z], dim=0)
        return z, mu, logvar, None
