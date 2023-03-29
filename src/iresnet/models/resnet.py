import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PyTree

class resnet_block_conv_cifar10(eqx.Module):
    layers: list

    def __init__(self, key):
        key1, key2, key3 = jax.random.split(key, 3)

        self.layers = [
            eqx.nn.Conv2d(3,6,5,1,1, key=key1),
            jax.nn.relu,
            eqx.nn.MaxPool2d(2,2),
            eqx.nn.Conv2d(6,16,5,1,1, key=key2),
            jax.nn.relu,
            eqx.nn.MaxPool2d(2,2),
            eqx.nn.ConvTranspose2d(in_channels = 16,out_channels = 3,kernel_size = 9,
                                   stride = 3, output_padding = 2, padding = 1, 
                                   dilation=2 , key=key3),
            jax.nn.relu,
        ]
    
    def __call__(self, x: Float[Array, "3 32 32"]) ->  Float[Array, "3 32 32"]:
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x

class resnet_skip_connection_cifar10(eqx.Module):
    layers: list

    def __init__(self, key):
        self.layers = [
            resnet_block_conv_cifar10(key),
        ]

    def __call__(self, x: Float[Array, "3 32 32"]) ->  Float[Array, "3 32 32"]:
            for layer in self.layers:
                y = layer(x)
            return x + y


class resnet_cifar10(eqx.Module):
    layers: list

    def __init__(self, key):
        key1, key2, key3, key4 = jax.random.split(key, 4)
        self.layers = [
            resnet_skip_connection_cifar10(key1),
            resnet_skip_connection_cifar10(key2),
            resnet_skip_connection_cifar10(key3),
            jnp.ravel,
            eqx.nn.MLP(3*32*32, 10,10,2, key=key4),
        ]

    def __call__(self, x: Float[Array, "3 32 32"]) -> Float[Array, "10"]:
        for layer in self.layers:
            x = layer(x)
        return x
    



