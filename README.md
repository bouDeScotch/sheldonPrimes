# Sheldon Primes Search (OEIS A069469)

High-performance C implementation for finding terms in [OEIS A069469](https://oeis.org/A069469): numbers $k$ such that $\pi(\text{reverse}(k)) = \text{reverse}(\pi(k))$.

These numbers are related to the famous "Sheldon Conjecture" (named after Sheldon Cooper's character from *The Big Bang Theory*, who highlighted the unique properties of $73 = \text{prime}(21)$).

## 🚀 Key Result & Breakthrough

* **New Lower Bound:** **$a(10) > 10^{11}$**
* **Previous Known Bound:** $a(10) > 10^9$ (Giovanni Resta, 2017)

By applying tight asymptotic prime bounds and in-memory filtering, this repository exhaustively searched all **100,000,000,000 primes** up to $2.76 \times 10^{12}$ and confirmed that **no new Sheldon primes exist under $10^{11}$**.

## 💡 Algorithmic Improvements (v4 vs Older Versions)

Earlier brute-force approaches generated massive **35 GB disk sieves** and were heavily bottlenecked by SSD I/O. The current architecture (`prime_reverse_v4.c`) introduces a multi-stage filtering pipeline that operates entirely in RAM:

1. **Lazy Digit Filter ($O(1)$):** Instantly eliminates indices ending in `0` and candidates whose reversed prime is divisible by 2, 3, or 5 before heavy calculations.
2. **Axler (2017) Asymptotic Bounds:** Uses ultra-tight prime bounds based on $p_n$ expansions ($\ln n + \ln \ln n - 1 + \dots$) to reject **>99.99%** of non-candidate primes instantly without executing expensive primality tests.
3. **Deterministic Miller-Rabin Pre-Filter:** Reduces candidates down to a microscopic pool using 4 proven bases for numbers under $3.4 \times 10^{14}$.
4. **Zero-Disk In-Memory Verification:** The candidate pool is so small (~273,000 items out of $10^{11}$ primes) that it consumes **< 9 MB of RAM**, allowing a fast, single-pass verification loop without any file I/O.

### Performance Milestone
- **100 Billion Primes Processed** in **~61 minutes** on a single thread (~27.2 million primes/sec).
- **RAM Usage:** ~9 MB total (no disk writing required).

---

## 🛠️ Dependencies

This project relies on `primesieve` for fast prime generation and `primecount` for exact prime counting.

On **MSYS2 (UCRT64)** or **Arch Linux**:
```bash
pacman -S mingw-w64-ucrt-x86_64-primesieve mingw-w64-ucrt-x86_64-primecount
```
On **Ubuntu/Debian** (not tested)
```bash
sudo apt update
sudo apt install libprimesieve-dev libprimecount-dev
```
## 💻 Building and Running

Compile with maximum gcc optimizations (-O3):
```bash
gcc -Wall -Wextra -O3 prime_reverse.c -o output/prime_reverse.exe -lprimesieve
./output/prime_reverse.exe
```

## 🤖 AI Acknowledgment
The optimization pipeline, analytical bounds derivation, and C implementation were developed collaboratively with AI. Every filter, math bound, and algorithm stage was thoroughly tested, verified, and benchmarked against all known OEIS terms ($a(1)$ through $a(9)$).
