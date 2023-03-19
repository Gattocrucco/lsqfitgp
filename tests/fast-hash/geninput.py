import numpy as np

gen = np.random.default_rng(202303181456)

def genint(dtype, size=()):
    return gen.integers(np.iinfo(dtype).min, np.iinfo(dtype).max, endpoint=True, dtype=dtype, size=size)

ninputs = 20

inputs = [genint('u1', size) for size in range(ninputs)]
seed32 = genint("u4")
seed64 = genint("u8")

print('\nC CODE:\n')
for size in range(ninputs):
    print(f'    uint8_t input{size}[] = {{{", ".join(map(str, inputs[size]))}}};')
print(f'    uint8_t *inputs[] = {{{", ".join(f"input{size}" for size in range(ninputs))}}};')
print(f'    uint32_t seed32 = {seed32}U;')
print(f'    uint64_t seed64 = {seed64}ULL;')

print('\nPYTHON CODE:\n')
print('    inputs = [')
for size in range(ninputs):
    print(f'        jnp.array([{", ".join(map(str, inputs[size]))}], dtype=jnp.uint8),')
print('    ]')
print(f'    seed32 = {seed32}')
print(f'    seed64 = {seed64}')
