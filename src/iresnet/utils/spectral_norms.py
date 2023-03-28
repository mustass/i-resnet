# Spectral norm for Feed Forward layers
# Based on: Spectral Normalization for Generative Adversarial Networks
#     (Miyato et al. 2018)
#     https://arxiv.org/abs/1802.05957
#
import jax
import jax.numpy as jnp

from equinox.nn import Linear, Conv2d


def spectral_norm_linear(linear, key, coeff=1.0, power_iter=1, eps=1e-12):
    if isinstance(linear, Linear):
        key_u, key_v = jax.random.split(key, 2)
        u = jax.random.normal(jax.random.PRNGKey(key_u), (1, linear.out_features))
        v = jax.random.normal(jax.random.PRNGKey(key_v), (1, linear.in_features))
        for _ in range(power_iter):
            v = jnp.matmul(linear.weight, u.T).T
            v = v / (jnp.linalg.norm(v) + eps)
            u = jnp.matmul(linear.weight.T, v.T).T
            u = u / (jnp.linalg.norm(u) + eps)
        sigma = jnp.matmul(jnp.matmul(v, linear.weight), u.T)
        linear.weight = linear.weight / jnp.max(1, sigma / coeff)
        return linear.weight
    else:
        raise ValueError("linear must be an instance of Linear")


def spectral_norm_conv2d(conv, key, coeff=1.0, power_iter=1, eps=1e-12):
    if isinstance(conv, Conv2d):
        key_u, key_v = jax.random.split(key, 2)
        u = jax.random.normal(jax.random.PRNGKey(key_u), (1, conv.out_channels))
        v = jax.random.normal(jax.random.PRNGKey(key_v), (1, conv.in_channels))
        for _ in range(power_iter):
            v = jnp.matmul(conv.weight.reshape(conv.out_channels, -1), u.T).T
            v = v / (jnp.linalg.norm(v) + eps)
            u = jnp.matmul(conv.weight.reshape(conv.out_channels, -1).T, v.T).T
            u = u / (jnp.linalg.norm(u) + eps)
        sigma = jnp.matmul(
            jnp.matmul(v, conv.weight.reshape(conv.out_channels, -1)), u.T
        )
        conv.weight = conv.weight / jnp.max(1, sigma / coeff)
        return conv.weight
    else:
        raise ValueError("conv must be an instance of Conv2d")
