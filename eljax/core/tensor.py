import jax
import jax.numpy as jnp
from jax import device_put

class Tensor:

    def __init__(self, value, requires_grad=False):

        self.value = device_put(jnp.asarray(value, dtype=jnp.float32), jax.devices("cpu")[0])
        self.requires_grad = requires_grad

        if self.requires_grad:
            self.grad = 0
        else:
            self.grad = None
    
    def __repr__(self):
        return f"Tensor({self.value})"

    def astype(self, dtpe):
        self.value = self.value.astype(dtpe)
        
    def dtype(self):
        return self.value.dtype

    def to(self, device):
        self.value = device_put(self.value, device)

    


if __name__ == '__main__':
    
    a = Tensor(jnp.ones(1000000))
    print(a.value.device_buffer.device())

    
    for device in jax.devices("gpu"):
        a.to(device)
        print(a.value.device_buffer.device())

    a.to(jax.devices("cpu")[0])
    print(a.value.device_buffer.device())
    
