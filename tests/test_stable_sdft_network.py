import timeit
import numpy as np
from cython_sdft_functions import cython_stable_sdft
from real_time_audio_visualization import stable_sdft 

# Set the value of N, k 
N = 10000
k = 5

# Generate sample data
signal = np.random.rand(N) + 1j * np.random.rand(N)  # Creating a complex signal

# Number of loops for benchmarking
num_loops = 1000000

# Benchmark the cython_stable_sdft function
cython_total_time = timeit.timeit("cython_stable_sdft(signal, N, k)", globals=globals(), number=num_loops)
cython_time_per_loop = cython_total_time / num_loops

print(f"Cython Stable SDFT - Time taken for {num_loops} runs: {cython_total_time} seconds")
print(f"Cython Stable SDFT - Time per loop: {cython_time_per_loop} seconds")

# Benchmark the stable_sdft function
stable_total_time = timeit.timeit("stable_sdft(signal, N, k)", globals=globals(), number=num_loops)
stable_time_per_loop = stable_total_time / num_loops

print(f"Stable SDFT - Time taken for {num_loops} runs: {stable_total_time} seconds")
print(f"Stable SDFT - Time per loop: {stable_time_per_loop} seconds")

# Compare the performance of both functions
if cython_time_per_loop < stable_time_per_loop:
    print("Cython Stable SDFT is faster.")
elif cython_time_per_loop > stable_time_per_loop:
    print("Stable SDFT is faster.")
else:
    print("Both functions have similar performance.")

