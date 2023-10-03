import timeit
import numpy as np
from cython_sdft_functions import cython_sdft  # Import the compiled Cython function
from real_time_audio_visualization import sdft  # Import the normal SDFT function

# Generate sample data
n = 1000
signal = np.random.rand(n) + 1j * np.random.rand(n)

# Number of loops for benchmarking
num_loops = 1000

# Benchmark the Cython SDFT function
cython_time = timeit.timeit("cython_sdft(signal, n)", globals=globals(), number=num_loops)
cython_per_loop_time = cython_time / num_loops
print(f"Time taken for {num_loops} runs (Cython): {cython_time} seconds")
print(f"Per loop time (Cython): {cython_per_loop_time} seconds")

# Benchmark the normal SDFT function
normal_time = timeit.timeit("sdft(signal, n)", globals=globals(), number=num_loops)
normal_per_loop_time = normal_time / num_loops
print(f"Time taken for {num_loops} runs (Normal): {normal_time} seconds")
print(f"Per loop time (Normal): {normal_per_loop_time} seconds")

