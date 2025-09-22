from agents.base_agent import BaseAgent
from models.lstm_encoder import LSTMEncoder


class HumplikAgent(BaseAgent):
    def __init__(self, config):
        super(HumplikAgent, self).__init__(config)
        assert not self.decode_reward
        assert self.decode_task
        assert not self.contrastive_task_loss
        assert self.use_kl_loss

    def initialise_encoder(self):
        encoder = LSTMEncoder(
            args=self.args,
            layers_before_gru=self.args.encoder_layers_before_gru,
            hidden_size=self.args.encoder_gru_hidden_size,
            layers_after_gru=self.args.encoder_layers_after_gru,
            latent_dim=self.args.latent_dim,
            action_dim=self.args.action_dim,
            action_embed_dim=self.args.action_embedding_size,
            state_dim=self.args.state_dim,
            state_embed_dim=self.args.state_embedding_size,
            reward_size=1,
            reward_embed_size=self.args.reward_embedding_size,
        )
        return encoder
