"""
Code is inspired by "Invertible Residual Networks"
http://proceedings.mlr.press/v97/behrmann19a.html
ICML, 2019

And re-implemented using Jax.
"""

import equinox as eqx
import jax
from equinox import nn
from jax import nn as jnn
from jax import numpy as jnp
import equinox as eqx
import equinox.experimental as eqxe
import jax.random as jr
import jax.tree_util as jtu
import functools as ft


class conv_iresnet_block(eqx.Module):
    conv1: nn.Conv2d
    conv2: nn.Conv2d
    conv3: nn.Conv2d

    def __init__(self, key: jax.random.PRNGKeyArray):
        self.conv1 = nn.Conv2d(
            3, 8, 3, stride=1, padding=1, key=key
        )  # let's figure the dimensions out later
        self.conv2 = nn.Conv2d(
            32, 16, 1, stride=1, padding=0, key=key
        )  # let's figure the dimensions out later
        self.conv3 = nn.Conv2d(
            32, 32, 3, stride=1, padding=1, key=key
        )  # let's figure the dimensions out later

    def __call__(
        self, x: jnp.ndarray, key: jax.random.PRNGKeyArray, inference: bool = False
    ) -> jnp.ndarray:
        k1, k2, k3, k4 = (
            jax.random.split(key, 4) if key is not None else (None, None, None, None)
        )
        x = self.conv1(x)

        x = jnn.elu(x)
        x = self.conv2(x)
        x = jnn.elu(x)
        x = self.conv3(x)
        return x


key = jr.PRNGKey(0)

model_key, spectral_key = jr.split(key)

SN = ft.partial(eqxe.SpectralNorm, key=spectral_key)


def _is_linear(leaf):
    return isinstance(leaf, eqx.nn.Linear)


def _is_conv2d(leaf):
    return isinstance(leaf, eqx.nn.Conv2d)


def _apply_sn_to_linear(module):
    if _is_linear(module):
        module = eqx.tree_at(lambda m: m.weight, module, replace_fn=SN)
    return module


def _apply_sn_to_conv2d(module):
    if _is_conv2d(module):
        module = eqx.tree_at(lambda m: m.weight, module, replace_fn=SN)
    return module


def apply_sn(model):
    return jtu.tree_map(_apply_sn_to_linear, model, is_leaf=_is_linear)


def apply_sn_conv(model):
    return jtu.tree_map(_apply_sn_to_conv2d, model, is_leaf=_is_conv2d)


model = eqx.nn.MLP(2, 2, 2, 2, key=model_key)
model_with_sn = apply_sn(model)
