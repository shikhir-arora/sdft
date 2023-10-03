# TODO Docs/FAQ/Analysis


![Figure_1](https://github.com/shikhir-arora/sdft/assets/8797918/ce7b2478-01fa-43af-a802-f9888f3f01e0) 


https://github.com/shikhir-arora/sdft/assets/8797918/591677fe-249e-44c7-af59-7d7a7164eace


https://github.com/shikhir-arora/sdft/assets/8797918/59e88348-0fc7-4a0d-b8e1-c1bd7446b26f


---


### Optimization Benchmarks

**See the test files in the `tests` folder in the repo.**

---

First, for SDFT (`sdft` -> `cython_sdft`):

Parameters: n = 1000 samples and the number of runs specified below.

```
Time taken for 1000 runs (Cython): 0.007804499997291714 seconds
Per loop time (Cython): 7.804499997291713e-06 seconds

Time taken for 1000 runs (Normal): 0.5207489170134068 seconds
Per loop time (Normal): 0.0005207489170134067 seconds
```
**Cython is approximately 66.75 times faster in this example.**

```
Time taken for 100000 runs (Cython): 0.783888083009515 seconds
Per loop time (Cython): 7.83888083009515e-06 seconds

Time taken for 100000 runs (Normal): 47.388167375000194 seconds
Per loop time (Normal): 0.0004738816737500019 seconds
```
**Cython is approximately 60.48 times faster in this case.**


*yes, it really is that much faster! 🙂* 

---

Secondly, for "Stable" SDFT (`stable_sdft` -> `cython_stable_sdft`)  moving filter (IIR), kth frequency bin, N samples, fs sampling frequency:

> **Note:** See https://www.dsprelated.com/showarticle/796.php and https://www.dsprelated.com/showarticle/1533.php by Rick Lyons, the original author of these algorithms and a wealth of insight!



Parameters: N = 10000 (large!) and the number of runs specified below, k = 5. 

```
Cython Stable SDFT - Time taken for 10000 runs: 1.7372043330105953 seconds
Cython Stable SDFT - Time per loop: 0.00017372043330105954 seconds

Stable SDFT - Time taken for 10000 runs: 2.27197125001112 seconds
Stable SDFT - Time per loop: 0.000227197125001112 seconds

Cython Stable SDFT is faster.
```
**Cython Stable SDFT is approximately 1.31 times faster in this example.**


```
Cython Stable SDFT - Time taken for 1000000 runs: 170.44354233302874 seconds
Cython Stable SDFT - Time per loop: 0.00017044354233302874 seconds

Stable SDFT - Time taken for 1000000 runs: 220.6045451249811 seconds
Stable SDFT - Time per loop: 0.00022060454512498107 seconds

Cython Stable SDFT is faster.
```
**Cython Stable SDFT is approximately 1.29 times faster in this example.**


```
Cython Stable SDFT - Time taken for 100000 runs: 17.084494707989506 seconds
Cython Stable SDFT - Time per loop: 0.00017084494707989505 seconds

Stable SDFT - Time taken for 100000 runs: 22.193888457957655 seconds
Stable SDFT - Time per loop: 0.00022193888457957655 seconds

Cython Stable SDFT is faster.
```
**Cython Stable SDFT is approximately 1.30 times faster in this example.**


With a lot of cythoning, we've accomplished the speedups.


