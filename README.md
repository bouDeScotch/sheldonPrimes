# Sheldon Primes Search (OEIS A069469)

High-performance C implementation for finding terms in [OEIS A069469](https://oeis.org/A069469): numbers $k$ such that $\pi(\text{reverse}(k)) = \text{reverse}(\pi(k))$.

These numbers are related to the famous "Sheldon Conjecture" (named after Sheldon Cooper's character from *The Big Bang Theory*, who highlighted the unique properties of $73 = \text{prime}(21)$
).

## 🚀 Key Result & Breakthrough

* **New Lower Bound:** **$a(10) > 10^{11}$**
* **Previous Known Bound:** $a(10) > 10^9$ (Giovanni Resta, 2017)

By applying tight asymptotic prime bounds and in-memory filtering, this repository exhaustively searched all **100,000,000,000 primes** up to $2.76 \times 10^{12}$ and confirmed that **no new Sheldon primes exist under $10^{11}$**.

## 💡 Algorithmic Improvements (v4 vs Older Versions)

Earlier brute-force approaches generated massive **35 GB disk sieves** and were heavily bottlenecked by SSD I/O. The current architecture (`prime_reverse.c`) introduces a multi-stage filtering pipeline that operates entirely in RAM:

1. **Lazy Digit Filter ($O(1)$):** Instantly eliminates indices ending in `0` and candidates whose reversed prime is divisible by 2, 3, or 5 before heavy calculations.
2. **Axler (2017) Asymptotic Bounds:** Uses ultra-tight prime bounds based on $p_n$ expansions ($\ln n + \ln \ln n - 1 + \dots$) to reject **>99.99%** of non-candidate primes instantly without executing expensive primality tests.
3. **Deterministic Miller-Rabin Pre-Filter:** Reduces candidates down to a microscopic pool using 4 proven bases for numbers under $3.4 \times 10^{14}$.
4. **Zero-Disk In-Memory Verification:** The candidate pool is so small (~273,000 items out of $10^{11}$ primes) that it consumes **< 9 MB of RAM**, allowing a fast, single-pass verification loop without any file I/O.

### Performance Milestone (single-threaded)
- **100 Billion Primes Processed** in **~61 minutes** on a single thread (~27.2 million primes/sec).
- **RAM Usage:** ~9 MB total (no disk writing required).

## ⚡ Multi-threading (v5)

The candidate-generation loop is now parallelized with OpenMP. `primesieve_iterator` can only generate primes in sequential order on a single thread, so the value range $[3, \text{LARGEST\ PRIME\ POSSIBLE}]$ is split into one chunk per thread, and each thread gets its **own** iterator.

The one piece of state a sequential loop gets "for free" that a chunked loop doesn't is `a`, the *rank* of the current prime (1st, 2nd, 3rd prime, ...), which the digit-reversal logic depends on. Each thread recovers its starting rank with `primecount_pi(chunk_start - 1) + 1` — `primecount`'s prime counting function is itself multi-threaded and asymptotically much faster than counting sequentially.

Each thread accumulates candidates in its own growable buffer (no locking), and all buffers are concatenated after the parallel region — safe to do in any order since the result gets sorted by `reverse_a` immediately afterward anyway.

```bash
OMP_NUM_THREADS=20 ./output/prime_reverse.exe   # defaults to all available cores if unset
```

### Empirical scaling law

Benchmarking the parallelized loop (20 threads) for $k = 6 \ldots 10$ ($A_{MAX} = 10^k$) and fitting a single-parameter model $t_1 = c \cdot N \cdot \ln(\ln(N))$ gives $R^2 = 0.99999$ (fit dominated by, and most accurate at, the largest tested points — $k=9$ and $k=10$ are within ~3% and ~0.03% of the model respectively). Extrapolating:

| $k$ | $A_{MAX}$ | Estimated time (20 threads) |
|---|---|---|
| 11 | $10^{11}$ | ~8 min |
| 12 | $10^{12}$ | ~81 min (~1.5h) |

## 🛠️ Dependencies

This project relies on `primesieve` for fast prime generation and `primecount` for exact prime counting.

On **MSYS2 (UCRT64)** or **Arch Linux**:
```bash
pacman -S mingw-w64-ucrt-x86_64-primesieve mingw-w64-ucrt-x86_64-primecount
```
On **Ubuntu/Debian**:
```bash
sudo apt update
sudo apt install libprimesieve-dev libprimecount-dev
```

## 💻 Building and Running

Compile with maximum gcc optimizations (-O3) and OpenMP enabled:
```bash
gcc -Wall -Wextra -O3 -fopenmp prime_reverse.c -o output/prime_reverse.exe -lprimesieve -lprimecount -lm
./output/prime_reverse.exe
```

## The next step

I am still trying to find the next number of this sequence. Using the parallelized version on my 20-core machine, increasing the lower bound to $10^{12}$ (or finding $a(10)$) is estimated to take around **~1.5 hours** of computation, so this will be run soon.

There is also more to gain from parallelization: the two big loops in the program are essentially doing simple computation and are both CPU-bound. So far only the candidate-generation loop has been parallelized — the second loop (computing $\pi(\text{reverse}(a))$ for each candidate) is still sequential and is a good target next.

## 🤖 AI Acknowledgment

The optimization pipeline, analytical bounds derivation, C implementation, OpenMP parallelization, and performance-scaling analysis were developed collaboratively with AI. Every filter, math bound, and algorithm stage was thoroughly tested, verified, and benchmarked against all known OEIS terms ($a(1)$ through $a(9)$).
