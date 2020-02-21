import jax
import jax.numpy as np
from jax.ops import index_update

from truss import truss_mass_jax
from truss import truss_stress_jax

grad_mass = jax.grad(truss_mass_jax)
grad_stress = jax.jacfwd(truss_stress_jax)

A: np.DeviceArray = np.ones(10)

dm = grad_mass(A)
ds = grad_stress(A)

print(f'\ndm: {dm}')
print(f'\nds: {ds}')