/*
 * prime_reverse_v2.c
 *
 * Find all a in [1, A_MAX] such that:
 *     n  = p(a)              (n is the a-th prime)
 *     b  = reverse_10(a)     (b is a written backwards in base 10)
 *     reverse_10(n) = p(b)   (reverse of n is the b-th prime)
 *
 * ---------------------------------------------------------------------
 * WHY THIS VERSION IS DIFFERENT FROM prime_reverse.c
 * ---------------------------------------------------------------------
 * v1 stored every prime as a raw 8-byte uint64_t in RAM. That's fine for
 * A_MAX=1e9 (8 GB) but breaks down for A_MAX=1e10 (80 GB, way past 32GB).
 *
 * v1 ALSO had a correctness gap: the sieve only extended slightly past
 * p(A_MAX), but reverse(n) for an n with D digits can be as large as
 * 10^D - 1, which can be ~4-10x bigger than p(A_MAX) itself. So some
 * legitimate hits near the end of a v1 run could have been silently
 * missed (reverse(n) simply wasn't in the sieved range, so it looked
 * "not found" instead of "found, but not the right index").
 *
 * v2 fixes both:
 *   1. Sieves all the way out to 10^D - 1, where D = digit count of the
 *      (margin-padded) estimate of p(A_MAX). This guarantees reverse(n)
 *      is always in range, whatever it turns out to be.
 *   2. Stores primes as GAP-ENCODED bytes on DISK instead of raw values
 *      in RAM: gap/2 fits in one byte for the overwhelming majority of
 *      primes in this range (average gap near 10^12 is ~28, and known
 *      maximal gaps below 10^12 are only in the low hundreds), with an
 *      escape byte (0x00 + 8-byte literal gap) for the rare larger ones.
 *      This is ~1 byte/prime on disk instead of 8 bytes/prime in RAM.
 *   3. Keeps small "checkpoints" (value + file offset) every 8192 primes
 *      fully in RAM (tens of MB, trivial) so any lookup can binary-search
 *      the checkpoints then decode forward a short distance on disk.
 *   4. Adds symmetry pruning: if (a,b) is a hit and a doesn't end in 0,
 *      then (b,a) is automatically also a hit - skip processing a when
 *      a > reverse(a), and record both directions when a hit is found.
 *   5. Parallelizes the search loop with OpenMP (each thread opens its
 *      own read handle onto the gap file).
 *
 * ---------------------------------------------------------------------
 * DISK / RAM REQUIREMENTS (approximate, printed exactly at runtime)
 * ---------------------------------------------------------------------
 *   A_MAX = 2e9   -> sieve to ~1e11, ~4.1e9 primes  -> ~4-5 GB disk
 *   A_MAX = 4e9   -> sieve to ~1e11, ~4.1e9 primes  -> ~4-5 GB disk
 *   A_MAX = 1e10  -> sieve to ~1e12, ~3.8e10 primes -> ~35-40 GB disk
 *   RAM usage stays small throughout (checkpoints + small buffers only,
 *   well under 1 GB even at A_MAX=1e10) - the big data lives on disk.
 *
 * Compile (MinGW/MSYS2 or Linux):
 *   gcc -O3 -march=native -fopenmp -o prime_reverse_v2 prime_reverse_v2.c -lm
 * Run (optional arg = A_MAX, default 10 billion):
 *   ./prime_reverse_v2 10000000000
 *
 * Needs ~40 GB free disk space next to the executable for A_MAX=1e10
 * (creates gaps.bin). Runtime for the sieve phase at that scale will be
 * substantially longer than the v1 run (order of an hour or more,
 * depending on CPU) since it sieves ~40x further out. The search phase
 * itself should be fast and scales with your core count via OpenMP.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _WIN32
  #define FSEEK _fseeki64
  #define FTELL _ftelli64
#else
  #define _FILE_OFFSET_BITS 64
  #define FSEEK fseeko
  #define FTELL ftello
#endif

/* ---------------- configuration ---------------- */
static uint64_t A_MAX = 10000000000ULL;
static const uint64_t CHECKPOINT_INTERVAL = 8192;
static const char *GAP_FILE_PATH = "gaps.bin";

#define SEG_SIZE   (1ULL << 27)   /* ~134M numbers per sieve segment */
#define WBUF_SIZE  (1 << 20)      /* 1MB write buffer */
#define RBUF_SIZE  (1 << 16)      /* 64KB read buffer per lookup */

static uint64_t total_primes_available = 0;

/* ---------------- utility ---------------- */

static inline uint64_t reverse_u64(uint64_t x) {
    uint64_t r = 0;
    while (x > 0) { r = r * 10 + x % 10; x /= 10; }
    return r;
}
static inline int digits_u64(uint64_t x) {
    int d = 1; while (x >= 10) { x /= 10; d++; } return d;
}
static uint64_t pow10_u64(int d) {
    uint64_t r = 1; for (int i = 0; i < d; i++) r *= 10; return r;
}
static uint64_t nth_prime_upper_bound(uint64_t n) {
    if (n < 6) return 15;
    double dn = (double)n;
    double bound = dn * (log(dn) + log(log(dn)));
    bound *= 1.06;
    return (uint64_t)bound + 1000;
}
static int is_palindrome_u64(uint64_t x) { return x == reverse_u64(x); }

/* ---------------- checkpoints (RAM, small) ---------------- */
typedef struct { uint64_t count; uint64_t value; int64_t offset; } Checkpoint;
static Checkpoint *checkpoints = NULL;
static uint64_t checkpoints_count = 0, checkpoints_cap = 0;

static void checkpoint_push(uint64_t count, uint64_t value, int64_t offset) {
    if (checkpoints_count == checkpoints_cap) {
        checkpoints_cap = checkpoints_cap ? checkpoints_cap * 2 : 1024;
        checkpoints = realloc(checkpoints, checkpoints_cap * sizeof(Checkpoint));
        if (!checkpoints) { fprintf(stderr, "OOM checkpoints\n"); exit(1); }
    }
    checkpoints[checkpoints_count].count = count;
    checkpoints[checkpoints_count].value = value;
    checkpoints[checkpoints_count].offset = offset;
    checkpoints_count++;
}
static uint64_t checkpoint_find_by_count(uint64_t target_count) {
    uint64_t lo = 0, hi = checkpoints_count - 1;
    while (lo < hi) {
        uint64_t mid = lo + (hi - lo + 1) / 2;
        if (checkpoints[mid].count <= target_count) lo = mid; else hi = mid - 1;
    }
    return lo;
}
static uint64_t checkpoint_find_by_value(uint64_t target_value) {
    uint64_t lo = 0, hi = checkpoints_count - 1;
    while (lo < hi) {
        uint64_t mid = lo + (hi - lo + 1) / 2;
        if (checkpoints[mid].value <= target_value) lo = mid; else hi = mid - 1;
    }
    return lo;
}

/* ---------------- gap record encode/decode ---------------- */

typedef struct { FILE *f; uint8_t buf[RBUF_SIZE]; int len, pos; } Reader;

static void reader_init(Reader *r, FILE *f, int64_t offset) {
    r->f = f; FSEEK(f, offset, SEEK_SET); r->len = 0; r->pos = 0;
}
static int reader_next_byte(Reader *r) {
    if (r->pos >= r->len) {
        r->len = (int)fread(r->buf, 1, RBUF_SIZE, r->f);
        r->pos = 0;
        if (r->len <= 0) return -1;
    }
    return r->buf[r->pos++];
}
static int64_t reader_next_gap(Reader *r) {
    int b = reader_next_byte(r);
    if (b < 0) return -1;
    if (b == 0) {
        uint64_t g = 0;
        for (int i = 0; i < 8; i++) {
            int c = reader_next_byte(r);
            if (c < 0) return -1;
            g |= ((uint64_t)c) << (8 * i);
        }
        return (int64_t)g;
    }
    return (int64_t)b * 2;
}

/* ---------------- writer (single-threaded, sieve phase) ---------------- */
static FILE *gap_write_f = NULL;
static uint8_t wbuf[WBUF_SIZE];
static int wbuf_pos = 0;
static int64_t total_bytes_written = 0;

static void writer_flush(void) {
    if (wbuf_pos > 0) {
        fwrite(wbuf, 1, wbuf_pos, gap_write_f);
        total_bytes_written += wbuf_pos;
        wbuf_pos = 0;
    }
}
static void writer_put_byte(uint8_t b) {
    if (wbuf_pos == WBUF_SIZE) writer_flush();
    wbuf[wbuf_pos++] = b;
}
static void writer_put_gap(uint64_t gap) {
    uint64_t half = gap / 2;
    if (half >= 1 && half <= 255) {
        writer_put_byte((uint8_t)half);
    } else {
        writer_put_byte(0);
        for (int i = 0; i < 8; i++) writer_put_byte((uint8_t)((gap >> (8 * i)) & 0xFF));
    }
}
static inline int64_t current_logical_offset(void) {
    return total_bytes_written + wbuf_pos;
}

/* ---------------- lookups (thread-safe: caller supplies its own FILE*) ---------------- */

static uint64_t get_nth_prime(FILE *f, uint64_t a) {
    if (a == 1) return 2;
    uint64_t ci = checkpoint_find_by_count(a);
    Reader r; reader_init(&r, f, checkpoints[ci].offset);
    uint64_t value = checkpoints[ci].value;
    uint64_t steps = a - checkpoints[ci].count;
    for (uint64_t s = 0; s < steps; s++) {
        int64_t g = reader_next_gap(&r);
        if (g < 0) { fprintf(stderr, "unexpected EOF in get_nth_prime(%llu)\n", (unsigned long long)a); exit(1); }
        value += (uint64_t)g;
    }
    return value;
}

/* returns 1-based prime index of target, or 0 if target is not prime / not in range */
static uint64_t find_prime_index(FILE *f, uint64_t target) {
    if (target == 2) return 1;
    if (target < 2) return 0;
    uint64_t ci = checkpoint_find_by_value(target);
    Reader r; reader_init(&r, f, checkpoints[ci].offset);
    uint64_t value = checkpoints[ci].value;
    uint64_t count = checkpoints[ci].count;
    if (value == target) return count;
    while (value < target) {
        int64_t g = reader_next_gap(&r);
        if (g < 0) return 0;
        value += (uint64_t)g;
        count++;
        if (value == target) return count;
        if (value > target) return 0;
    }
    return 0;
}

/* ---------------- segmented sieve + gap generation ---------------- */

static void generate_primes(uint64_t sieve_limit) {
    uint64_t sqrt_limit = (uint64_t)sqrt((double)sieve_limit) + 1;

    uint8_t *small_composite = calloc(sqrt_limit + 1, 1);
    uint64_t *small_primes = malloc(sizeof(uint64_t) * (sqrt_limit / 8 + 100));
    uint64_t small_count = 0;
    for (uint64_t i = 2; i <= sqrt_limit; i++) {
        if (!small_composite[i]) {
            small_primes[small_count++] = i;
            for (uint64_t j = i * i; j <= sqrt_limit; j += i) small_composite[j] = 1;
        }
    }
    free(small_composite);

    gap_write_f = fopen(GAP_FILE_PATH, "wb");
    if (!gap_write_f) { perror("fopen gaps.bin"); exit(1); }

    uint64_t primes_count = 2;
    uint64_t prev_value = 3;
    checkpoint_push(2, 3, 0);
    uint64_t next_checkpoint_at = 2 + CHECKPOINT_INTERVAL;

    time_t t0 = time(NULL);

    for (uint64_t low = 4; low <= sieve_limit; low += SEG_SIZE) {
        uint64_t high = low + SEG_SIZE - 1;
        if (high > sieve_limit) high = sieve_limit;
        uint64_t span = high - low + 1;
        uint64_t nbits = span / 2 + 2;
        uint8_t *seg = calloc(nbits / 8 + 1, 1);

        for (uint64_t k = 0; k < small_count; k++) {
            uint64_t p = small_primes[k];
            if (p < 3) continue;
            if (p * p > high) break;
            uint64_t start = (low % p == 0) ? low : (low + (p - low % p));
            if (start < p * p) start = p * p;
            if ((start & 1) == 0) start += p;
            for (uint64_t m = start; m <= high; m += 2 * p) {
                uint64_t idx = (m - low) / 2;
                seg[idx >> 3] |= (uint8_t)(1u << (idx & 7));
            }
        }

        uint64_t first_odd = (low % 2 == 0) ? low + 1 : low;
        for (uint64_t v = first_odd; v <= high; v += 2) {
            uint64_t idx = (v - low) / 2;
            if (seg[idx >> 3] & (1u << (idx & 7))) continue;
            uint64_t gap = v - prev_value;
            writer_put_gap(gap);
            prev_value = v;
            primes_count++;
            if (primes_count == next_checkpoint_at) {
                checkpoint_push(primes_count, v, current_logical_offset());
                next_checkpoint_at += CHECKPOINT_INTERVAL;
            }
        }
        free(seg);

        fprintf(stderr, "[sieve] %.1f%% (up to %llu / %llu), primes so far=%llu, elapsed=%lds\n",
                100.0 * high / sieve_limit, (unsigned long long)high,
                (unsigned long long)sieve_limit, (unsigned long long)primes_count,
                (long)(time(NULL) - t0));
    }

    writer_flush();
    fclose(gap_write_f);
    free(small_primes);

    fprintf(stderr, "[sieve] done. total primes=%llu, gap file bytes=%lld, elapsed=%lds\n",
            (unsigned long long)primes_count, (long long)total_bytes_written,
            (long)(time(NULL) - t0));

    if (primes_count < A_MAX) {
        fprintf(stderr, "WARNING: only found %llu primes, wanted %llu -- increase margin.\n",
                (unsigned long long)primes_count, (unsigned long long)A_MAX);
    }
    total_primes_available = primes_count;
}

/* ---------------- main search phase ---------------- */

static void search(void) {
    uint64_t limit_a = A_MAX;
    if (limit_a > total_primes_available) limit_a = total_primes_available;

    FILE *out = fopen("hits.txt", "w");
    if (!out) { perror("fopen hits.txt"); exit(1); }

    uint64_t hits = 0, nontrivial_hits = 0;
    uint64_t processed = 0;
    time_t t0 = time(NULL);

    #pragma omp parallel
    {
        FILE *f = fopen(GAP_FILE_PATH, "rb");
        if (!f) { perror("fopen gaps.bin (reader)"); exit(1); }
        uint64_t local_processed = 0;

        #pragma omp for schedule(dynamic, 100000)
        for (uint64_t a = 1; a <= limit_a; a++) {
            uint64_t b = reverse_u64(a);

            if (a % 10 != 0 && a > b) continue; /* mirror already handled */
            if (b == 0 || b > limit_a) continue;

            uint64_t n = get_nth_prime(f, a);
            uint64_t rn = reverse_u64(n);
            uint64_t idx = find_prime_index(f, rn);

            if (idx == b) {
                int trivial = is_palindrome_u64(a);
                #pragma omp critical
                {
                    fprintf(out, "a=%llu n=%llu b=%llu rev(n)=%llu trivial=%d\n",
                            (unsigned long long)a, (unsigned long long)n,
                            (unsigned long long)b, (unsigned long long)rn, trivial);
                    fflush(out);
                    hits++;
                    if (!trivial) {
                        nontrivial_hits++;
                        fprintf(stderr, ">>> NONTRIVIAL HIT: a=%llu n=%llu b=%llu rev(n)=%llu\n",
                                (unsigned long long)a, (unsigned long long)n,
                                (unsigned long long)b, (unsigned long long)rn);
                    }
                    /* mirror: (b, a) is automatically also a hit, since a % 10 != 0
                       guarantees reverse(b) == a exactly. Record it directly without
                       another lookup - we already have all four values. */
                    if (a != b && a % 10 != 0) {
                        fprintf(out, "a=%llu n=%llu b=%llu rev(n)=%llu trivial=%d\n",
                                (unsigned long long)b, (unsigned long long)rn,
                                (unsigned long long)a, (unsigned long long)n, trivial);
                        hits++;
                        if (!trivial) nontrivial_hits++;
                    }
                }
            }

            local_processed++;
            if (local_processed % 2000000 == 0) {
                #pragma omp atomic
                processed += 2000000;
                #pragma omp critical
                fprintf(stderr, "[progress] ~%llu / %llu (%.1f%%) hits=%llu nontrivial=%llu elapsed=%lds\n",
                        (unsigned long long)processed, (unsigned long long)limit_a,
                        100.0 * processed / limit_a, (unsigned long long)hits,
                        (unsigned long long)nontrivial_hits, (long)(time(NULL) - t0));
            }
        }
        fclose(f);
    }

    fclose(out);
    fprintf(stderr, "\nTotal hits: %llu, non-trivial: %llu. Results in hits.txt. Total time: %lds\n",
            (unsigned long long)hits, (unsigned long long)nontrivial_hits, (long)(time(NULL) - t0));
}

/* ---------------- main ---------------- */

int main(int argc, char **argv) {
    if (argc > 1) A_MAX = strtoull(argv[1], NULL, 10);

    uint64_t p_est = nth_prime_upper_bound(A_MAX);
    int D = digits_u64(p_est);
    uint64_t sieve_limit = pow10_u64(D) - 1;

    double est_primes = sieve_limit / (log((double)sieve_limit) - 1.0);
    double est_disk_gb = est_primes * 1.02 / 1e9;

    fprintf(stderr, "A_MAX = %llu\n", (unsigned long long)A_MAX);
    fprintf(stderr, "Estimated p(A_MAX) ~ %llu (%d digits)\n", (unsigned long long)p_est, D);
    fprintf(stderr, "Sieve limit (10^%d - 1) = %llu\n", D, (unsigned long long)sieve_limit);
    fprintf(stderr, "Estimated prime count ~ %.0f, estimated disk usage ~ %.1f GB\n",
            est_primes, est_disk_gb);
    fprintf(stderr, "Checkpoint RAM usage ~ %.1f MB\n",
            (est_primes / CHECKPOINT_INTERVAL) * sizeof(Checkpoint) / 1e6);
#ifdef _OPENMP
    fprintf(stderr, "OpenMP enabled, using up to %d threads\n", omp_get_max_threads());
#else
    fprintf(stderr, "OpenMP NOT enabled (compile with -fopenmp for parallel search)\n");
#endif
    fprintf(stderr, "\n");

    generate_primes(sieve_limit);
    search();

    return 0;
}