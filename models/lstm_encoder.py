import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

from utils import helpers as utl


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class LSTMEncoder(nn.Module):
    def __init__(
        self,
        args,
        # network size
        layers_before_gru=(),
        hidden_size=64,
        layers_after_gru=(),
        latent_dim=32,
        action_dim=2,
        action_embed_dim=10,
        state_dim=2,
        state_embed_dim=10,
        reward_size=1,
        reward_embed_size=5,
    ):
        super(LSTMEncoder, self).__init__()

        self.args = args
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.reparameterise = self._sample_gaussian
        self.encode_action = (
            self.args.encode_action if hasattr(self, "encode_action") else True
        )

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reward_size = reward_size

        # embed action, state, reward
        self.state_encoder = utl.FeatureExtractor(state_dim, state_embed_dim, F.relu)
        self.reward_encoder = utl.FeatureExtractor(
            reward_size, reward_embed_size, F.relu
        )
        self.action_encoder = None

        curr_input_dim = state_embed_dim + reward_embed_size

        if self.encode_action:
            self.action_encoder = utl.FeatureExtractor(
                action_dim, action_embed_dim, F.relu
            )
            curr_input_dim += action_embed_dim

        # fully connected layers before the recurrent cell
        self.fc_before_gru = nn.ModuleList([])
        for i in range(len(layers_before_gru)):
            self.fc_before_gru.append(nn.Linear(curr_input_dim, layers_before_gru[i]))
            curr_input_dim = layers_before_gru[i]

        # recurrent unit (LSTM instead of GRU)
        self.lstm = nn.LSTM(
            input_size=curr_input_dim,
            hidden_size=hidden_size,
            num_layers=self.args.num_gru_layers,
        )

        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param)

        # fully connected layers after the recurrent cell
        curr_input_dim = hidden_size
        self.fc_after_gru = nn.ModuleList([])
        for i in range(len(layers_after_gru)):
            self.fc_after_gru.append(nn.Linear(curr_input_dim, layers_after_gru[i]))
            curr_input_dim = layers_after_gru[i]

        # output layer
        self.fc_mu = nn.Linear(curr_input_dim, latent_dim)
        self.fc_logvar = nn.Linear(curr_input_dim, latent_dim)

        self.truncate_size = (
            args.truncate_size if hasattr(args, "truncate_size") else None
        )

    def _sample_gaussian(self, mu, logvar, num=None):
        std = torch.exp(0.5 * logvar)

        if num is None:
            eps = torch.randn_like(std)
            return eps * std + mu

        else:
            mu = mu.unsqueeze(0).expand(num, *mu.shape)
            std = std.unsqueeze(0).expand(num, *std.shape)
            eps = torch.randn_like(std)
            return eps * std + mu

    def reset_hidden(self, hidden_state, done):
        """Reset the hidden state where the BAMDP was done (i.e., we get a new task)"""
        if done:
            if isinstance(hidden_state, tuple):
                h, c = hidden_state
                return (
                    torch.zeros_like(h),
                    torch.zeros_like(c),
                )
            return (
                torch.zeros_like(hidden_state),
                torch.zeros_like(hidden_state),
            )
        return hidden_state

    def prior(self, batch_size, sample=True):
        # TODO: add option to incorporate the initial state

        hs = self.state_encoder(torch.zeros(batch_size, self.state_dim))
        hr = self.reward_encoder(torch.zeros(batch_size, self.reward_size))
        if self.encode_action:
            ha = self.action_encoder(torch.zeros(batch_size, self.action_dim))
            h = torch.cat((ha, hs, hr), dim=-1).unsqueeze(0)
        else:
            h = torch.cat((hs, hr), dim=-1).unsqueeze(0)
        # forward through fully connected layers before LSTM
        for i in range(len(self.fc_before_gru)):
            h = F.relu(self.fc_before_gru[i](h))

        # we start out with a hidden state of zero
        h0 = torch.zeros(
            (self.args.num_gru_layers, batch_size, self.hidden_size), requires_grad=True
        ).to(device)
        c0 = torch.zeros(
            (self.args.num_gru_layers, batch_size, self.hidden_size), requires_grad=True
        ).to(device)

        h, (h_n, c_n) = self.lstm(h, (h0, c0))
        # forward through fully connected layers after LSTM
        for i in range(len(self.fc_after_gru)):
            h = F.relu(self.fc_after_gru[i](h))

        # outputs
        latent_mean = self.fc_mu(h)
        latent_logvar = self.fc_logvar(h)
        if sample:
            latent_sample = self.reparameterise(latent_mean, latent_logvar)
        else:
            latent_sample = latent_mean

        return latent_sample, latent_mean, latent_logvar, (h_n, c_n)

    def forward(
        self,
        actions,
        states,
        rewards,
        hidden_state,
        return_prior,
        sample=True,
        detach_every=None,
    ):
        """
        Actions, states, rewards should be given in form [sequence_len * batch_size * dim].
        For one-step predictions, sequence_len=1 and hidden_state!=None.
        For feeding in entire trajectories, sequence_len>1 and hidden_state=None.
        In the latter case, we return embeddings of length sequence_len+1 since they include the prior.

        :param actions: Action tensor of shape [T, B, action_dim]
        :param states: State tensor of shape [T, B, state_dim]
        :param rewards: Reward tensor of shape [T, B, 1]
        :param hidden_state: Hidden state tuple (h, c) for LSTM or None when using prior
        :param return_prior: Whether to prepend prior embedding
        :param sample: Whether to sample from posterior
        :param detach_every: Truncated BPTT length if provided
        """

        # shape should be: sequence_len x batch_size x hidden_size
        actions = actions.reshape((-1, *actions.shape[-2:]))
        states = states.reshape((-1, *states.shape[-2:]))
        rewards = rewards.reshape((-1, *rewards.shape[-2:]))
        if return_prior:
            # if hidden state is none, start with the prior
            prior_sample, prior_mean, prior_logvar, prior_hidden_state = self.prior(
                actions.shape[1]
            )
            hidden_state = (
                prior_hidden_state[0].clone(),
                prior_hidden_state[1].clone(),
            )

        hs = self.state_encoder(states)
        hr = self.reward_encoder(rewards)
        if self.encode_action:
            ha = self.action_encoder(actions)
            h = torch.cat((ha, hs, hr), dim=-1)
        else:
            h = torch.cat((hs, hr), dim=-1)

        # fully connected layers before LSTM
        for i in range(len(self.fc_before_gru)):
            h = F.relu(self.fc_before_gru[i](h))

        if detach_every is None:
            output, hidden_state = self.lstm(h, hidden_state)
        else:
            output = []
            for i in range(int(np.ceil(h.shape[0] / detach_every))):
                curr_input = h[i * detach_every : i * detach_every + detach_every]
                curr_output, hidden_state = self.lstm(curr_input, hidden_state)
                output.append(curr_output)
                # detach hidden state; useful for BPTT when sequences are very long
                h_n, c_n = hidden_state
                hidden_state = (h_n.detach(), c_n.detach())
            output = torch.cat(output, dim=0)
        lstm_h = output.clone()

        # fully connected layers after LSTM
        for i in range(len(self.fc_after_gru)):
            lstm_h = F.relu(self.fc_after_gru[i](lstm_h))

        # outputs
        latent_mean = self.fc_mu(lstm_h)
        latent_logvar = self.fc_logvar(lstm_h)
        if sample:
            latent_sample = self.reparameterise(latent_mean, latent_logvar)
        else:
            latent_sample = latent_mean

        if return_prior:
            latent_sample = torch.cat((prior_sample, latent_sample))
            latent_mean = torch.cat((prior_mean, latent_mean))
            latent_logvar = torch.cat((prior_logvar, latent_logvar))
            # For consistency with the GRU encoder implementation
            # prepend the prior hidden state along the time dimension
            output = torch.cat((prior_hidden_state[0], output))

        if latent_mean.shape[0] == 1:
            latent_sample, latent_mean, latent_logvar = (
                latent_sample[0],
                latent_mean[0],
                latent_logvar[0],
            )

        return latent_sample, latent_mean, latent_logvar, hidden_state
