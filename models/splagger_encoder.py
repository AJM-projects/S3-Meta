import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from utils import helpers as utl

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SplAggerEncoder(nn.Module):
    """
    SplAgger task encoder: GRU + elementwise-max aggregation + skip connection.

    The hidden state passed between calls packs both the GRU internal state and
    the running max into a single tensor of shape
    (num_gru_layers + 1, batch, hidden_size), where the last slice stores the
    running max of all GRU outputs seen so far in the current episode.

    No stochasticity: latent_sample == latent_mean, logvar is always zeros.
    """

    def __init__(
        self,
        args,
        layers_before_gru=(),
        hidden_size=256,
        layers_after_gru=(),
        latent_dim=12,
        action_dim=2,
        action_embed_dim=10,
        state_dim=2,
        state_embed_dim=10,
        reward_size=1,
        reward_embed_size=5,
    ):
        super().__init__()

        self.args = args
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.num_gru_layers = args.num_gru_layers

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reward_size = reward_size

        self.state_encoder = utl.FeatureExtractor(state_dim, state_embed_dim, F.relu)
        self.reward_encoder = utl.FeatureExtractor(
            reward_size, reward_embed_size, F.relu
        )
        self.action_encoder = utl.FeatureExtractor(action_dim, action_embed_dim, F.relu)

        curr_input_dim = state_embed_dim + reward_embed_size + action_embed_dim

        self.fc_before_gru = nn.ModuleList()
        for size in layers_before_gru:
            self.fc_before_gru.append(nn.Linear(curr_input_dim, size))
            curr_input_dim = size

        self.gru = nn.GRU(
            input_size=curr_input_dim,
            hidden_size=hidden_size,
            num_layers=self.num_gru_layers,
        )
        for name, param in self.gru.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param)

        # After skip-concat the input doubles (gru_out || running_max)
        skip_dim = hidden_size * 2
        curr_input_dim = skip_dim
        self.fc_after_gru = nn.ModuleList()
        for size in layers_after_gru:
            self.fc_after_gru.append(nn.Linear(curr_input_dim, size))
            curr_input_dim = size

        self.fc_mu = nn.Linear(curr_input_dim, latent_dim)

    # ------------------------------------------------------------------
    # helpers

    def _embed(self, actions, states, rewards):
        ha = self.action_encoder(actions)
        hs = self.state_encoder(states)
        hr = self.reward_encoder(rewards)
        return torch.cat((ha, hs, hr), dim=-1)

    def _pack_hidden(self, gru_hidden, running_max):
        """Pack (num_gru_layers, B, H) + (1, B, H) → (num_gru_layers+1, B, H)."""
        return torch.cat((gru_hidden, running_max), dim=0)

    def _unpack_hidden(self, hidden_state):
        """Unpack packed hidden state → (gru_hidden, running_max)."""
        return hidden_state[: self.num_gru_layers], hidden_state[[self.num_gru_layers]]

    def _latent_from_skip(self, gru_out, running_max):
        """
        Apply skip connection then FC layers to produce latent mean.

        gru_out:     (T, B, hidden_size)
        running_max: (T, B, hidden_size)  — already aligned with gru_out
        """
        h = torch.cat((gru_out, running_max), dim=-1)  # (T, B, 2*hidden_size)
        for fc in self.fc_after_gru:
            h = F.relu(fc(h))
        return self.fc_mu(h)  # (T, B, latent_dim)

    def reset_hidden(self, hidden_state, done):
        if done:
            return torch.zeros_like(hidden_state)
        return hidden_state

    # ------------------------------------------------------------------

    def prior(self, batch_size, sample=True):
        h = self._embed(
            torch.zeros(1, batch_size, self.action_dim),
            torch.zeros(1, batch_size, self.state_dim),
            torch.zeros(1, batch_size, self.reward_size),
        )
        for fc in self.fc_before_gru:
            h = F.relu(fc(h))

        gru_hidden = torch.zeros(self.num_gru_layers, batch_size, self.hidden_size).to(
            device
        )
        gru_out, gru_hidden = self.gru(h, gru_hidden)  # (1, B, H)

        running_max = gru_out  # first step: running max = first output
        latent_mean = self._latent_from_skip(gru_out, running_max)  # (1, B, latent_dim)
        latent_logvar = torch.zeros_like(latent_mean)

        hidden_state = self._pack_hidden(gru_hidden, running_max)
        return latent_mean, latent_mean, latent_logvar, hidden_state

    def forward(
        self,
        actions,
        states,
        rewards,
        hidden_state,
        return_prior=False,
        sample=True,
        detach_every=None,
    ):
        actions = actions.reshape((-1, *actions.shape[-2:]))
        states = states.reshape((-1, *states.shape[-2:]))
        rewards = rewards.reshape((-1, *rewards.shape[-2:]))

        if return_prior:
            prior_sample, prior_mean, prior_logvar, prior_hidden = self.prior(
                actions.shape[1]
            )
            gru_hidden, running_max = self._unpack_hidden(prior_hidden)
        else:
            gru_hidden, running_max = self._unpack_hidden(hidden_state)

        h = self._embed(actions, states, rewards)
        for fc in self.fc_before_gru:
            h = F.relu(fc(h))

        if detach_every is None:
            gru_out, gru_hidden = self.gru(h, gru_hidden)
        else:
            chunks = []
            for i in range(int(np.ceil(h.shape[0] / detach_every))):
                chunk_out, gru_hidden = self.gru(
                    h[i * detach_every : (i + 1) * detach_every], gru_hidden
                )
                chunks.append(chunk_out)
                gru_hidden = gru_hidden.detach()
            gru_out = torch.cat(chunks, dim=0)  # (T, B, H)

        # Compute running max over the sequence, accounting for any prior running_max
        # running_max: (1, B, H); gru_out: (T, B, H)
        cummax = torch.cummax(gru_out, dim=0).values  # (T, B, H)
        # Incorporate the pre-existing running max from hidden state
        cummax = torch.max(running_max.expand_as(cummax), cummax)
        new_running_max = cummax[[-1]]  # (1, B, H)

        latent_mean = self._latent_from_skip(gru_out, cummax)  # (T, B, latent_dim)
        latent_logvar = torch.zeros_like(latent_mean)

        if return_prior:
            latent_mean = torch.cat((prior_mean, latent_mean), dim=0)
            latent_logvar = torch.cat((prior_logvar, latent_logvar), dim=0)

        new_hidden = self._pack_hidden(gru_hidden, new_running_max)

        # Squeeze time dim for single-step inference (matches RNNEncoder convention)
        if latent_mean.shape[0] == 1:
            latent_mean = latent_mean[0]
            latent_logvar = latent_logvar[0]

        return latent_mean, latent_mean, latent_logvar, new_hidden
