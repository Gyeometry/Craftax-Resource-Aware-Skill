# 기존 Phase42 실행 네트워크의 두 클래스만 추출한 추론 전용 어댑터.
import functools
from typing import Sequence,Dict
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant,orthogonal
import distrax


class ScannedRNN(nn.Module):

    @functools.partial(nn.scan, variable_broadcast='params', in_axes=0, out_axes=0, split_rngs={'params': False})
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(resets[:, np.newaxis], self.initialize_carry(ins.shape[0], ins.shape[1]), rnn_state)
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return (new_rnn_state, y)

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))

class ActorCriticRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x[:2]
        action_mask = x[2] if len(x) > 2 else jnp.ones(obs.shape[:-1] + (self.action_dim,), bool)
        embedding = nn.Dense(self.config['LAYER_SIZE'], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(obs)
        embedding = nn.relu(embedding)
        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)
        actor_mean = nn.Dense(self.config['LAYER_SIZE'], kernel_init=orthogonal(2), bias_init=constant(0.0))(embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(self.config['LAYER_SIZE'], kernel_init=orthogonal(2), bias_init=constant(0.0))(actor_mean)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)
        pi = distrax.Categorical(logits=jnp.where(action_mask, actor_mean, -1000000000.0))
        critic = nn.Dense(self.config['LAYER_SIZE'], kernel_init=orthogonal(2), bias_init=constant(0.0))(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(self.config['LAYER_SIZE'], kernel_init=orthogonal(2), bias_init=constant(0.0))(critic)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)
        return (hidden, pi, jnp.squeeze(critic, axis=-1))
