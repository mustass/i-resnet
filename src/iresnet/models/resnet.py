import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PyTree

class BasicBlock(eqx.Module):
    layers:list
    skip: list
    
    def __init__(self, in_planes, planes, key, stride = 1) -> None:
        super(BasicBlock).__init__()
        key1, key2 = jax.random.split(key, 2)
        self.layers = [
         eqx.nn.Conv2d(in_channels=in_planes, out_channels=planes,  kernel_size=3, stride=stride, padding=1, use_bias=False, key=key1),
         eqx.experimental.BatchNorm(input_size=planes, axis_name="batch"),
         jax.nn.relu,
         eqx.nn.Conv2d(in_channels=planes, out_channels=planes,  kernel_size=3, stride= 1,padding=1, use_bias=False,key=key2),
         eqx.experimental.BatchNorm(input_size=planes, axis_name="batch")
        ]
        self.skip = []
        if stride != 1 or (in_planes != planes):
            self.skip = [
                eqx.nn.Conv2d(in_channels=in_planes, out_channels=planes,  kernel_size=1, stride=stride ,padding=0, use_bias=False,key=key1),
                eqx.experimental.BatchNorm(input_size=planes, axis_name="batch"),
            ]

    def __call__(self, x: Float[Array, "3 32 32"]) ->  Float[Array, "3 32 32"]:
        y = x
        for i, layer in enumerate(self.layers):
            y = layer(y)
        y +=  self.skip[-1](self.skip[-2](x)) if len(self.skip)>0 else x
        y = jax.nn.relu(y)
        return y



def make_layer(in_planes, planes, num_blocks, stride_in, key) -> None:
        keys = jax.random.split(key, num_blocks)
        strides = [stride_in] + [1]*(num_blocks-1)

        layers = []
        for stride in strides:
            key = keys[stride]
            layers.append(BasicBlock(in_planes,planes,key,stride=stride))
            in_planes = planes
        return layers

class ResNet18(eqx.Module):
    layers: list

    def __init__(self,key) -> None:
        super(ResNet18).__init__()
        key1, key2, key3, key4, key5, key6 = jax.random.split(key, 5)
        self.layers = [
            eqx.nn.Conv2d(in_channels=3, out_channels=64,  kernel_size=3, stride= 1,padding=1, use_bias=False, key=key1),
            eqx.experimental.BatchNorm(input_size=64, axis_name="batch"),
            jax.nn.relu]
        self.layers.extend(make_layer(64, 64,2,1,key2))
        self.layers.extend(make_layer(64,128,2,2,key3))
        self.layers.extend(make_layer(128,256,2,2,key4))
        self.layers.extend(make_layer(256,512,2,2,key5))
        
        output_lyrs = [eqx.nn.AvgPool2d(4,1),
            jnp.ravel,
            eqx.nn.Linear(512,10,key=key6),
            jax.nn.log_softmax
        ]
        self.layers.extend(output_lyrs)

    def __call__(self, x: Float[Array, "3 32 32"]) ->  Float[Array, "10"]:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            #print(f'Output Shape of layer {i} is:  {x.shape}')
        return x 