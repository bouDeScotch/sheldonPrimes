#include "primesieve.h"
#include "primecount.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <omp.h>
/*
This is a new version of the prime_reverse_v3.c file. We will now use a completly new approach to try to find a(10)
We will iterate through all primes like before, filtering those who can't be solutions based on basic filters like in the previous version.
Every index that passes the filters will be a candidate.
We will then sort the candidates by their reverse value and compute their reverse prime value.
Because we will already have computed a, pi(a) and reverse(a), computing pi(reverse(a)) will be easy and fast considering there will be
only "a few" candidates, and that because it is sorted we won't have to do a random access, in fact there probably won't even be
a need for a file because computing on the fly might be fast enough, or even could be done on the GPU if we want to go that far.

NOTE : After some testing, the number of candidates stays around the thousands, maybe millions at worst, but it fits in RAM easily.

I will reuse a lot of code from the previous version because I want the file to be independent.
*/

// Fast modular multiplication using 128-bit integers
inline uint64_t mul_mod(uint64_t a, uint64_t b, uint64_t m) {
    return (uint64_t)(((__uint128_t)a * b) % m);
}

// Fast modular exponentiation
uint64_t power_mod(uint64_t base, uint64_t exp, uint64_t mod) {
    uint64_t res = 1;
    base %= mod;
    while (exp > 0) {
        if (exp % 2 == 1) res = mul_mod(res, base, mod);
        base = mul_mod(base, base, mod);
        exp /= 2;
    }
    return res;
}

// Deterministic Miller-Rabin up to 3.3 x 10^24 (using 7 bases)
bool is_prime_mr(uint64_t n) {
    if (n < 2) return false;
    if (n == 2 || n == 3 || n == 5) return true;
    if (n % 2 == 0 || n % 3 == 0 || n % 5 == 0) return false;

    uint64_t d = n - 1;
    int s = 0;
    while (d % 2 == 0) {
        d /= 2;
        s++;
    }

    static const uint64_t bases[] = {2, 3, 5, 7, 11, 13, 17};
    for (int i = 0; i < 7; i++) {
        uint64_t a = bases[i];
        if (n <= a) break;
        
        uint64_t x = power_mod(a, d, n);
        if (x == 1 || x == n - 1) continue;
        
        bool composite = true;
        for (int r = 1; r < s; r++) {
            x = mul_mod(x, x, n);
            if (x == n - 1) {
                composite = false;
                break;
            }
        }
        if (composite) return false;
    }
    return true;
}

bool is_in_prime_bounds(uint64_t n, uint64_t candidate_prime) {
    if (n < 6) return true; // Bounds math is slightly off for tiny primes, allow them

    double dn = (double)n;
    double ln_n = log(dn);
    double ln_ln_n = log(ln_n);

    // Dusart 2010 Lower Bound (valid for n >= 3)
    double lower_factor = ln_n + ln_ln_n - 1.0 + (ln_ln_n - 2.1) / ln_n;
    double upper_factor;

    // Dusart 2010 Upper Bound (valid for n >= 688383)
    if (n >= 688383) {
        upper_factor = ln_n + ln_ln_n - 1.0 + (ln_ln_n - 2.0) / ln_n;
    } else {
        // Safe fallback upper bound for smaller 'n'
        upper_factor = ln_n + ln_ln_n; 
    }

    // Apply a 0.001 margin to protect against IEEE 754 float inaccuracies
    uint64_t min_possible = (uint64_t)(dn * (lower_factor - 0.001));
    uint64_t max_possible = (uint64_t)(dn * (upper_factor + 0.001)) + 1;

    return (candidate_prime >= min_possible && candidate_prime <= max_possible);
}

uint64_t reverse(uint64_t n) {
    uint64_t reversed = 0;
    while (n > 0) {
        reversed = reversed * 10 + n % 10;
        n /= 10;
    }
    return reversed;
}

double time_taken_between(struct timespec start, struct timespec end) {
    double time_taken = (end.tv_sec - start.tv_sec) * 1e9;
    time_taken = (time_taken + (end.tv_nsec - start.tv_nsec)) * 1e-9;
    return time_taken;
}

typedef struct {
    uint64_t a;
    uint64_t prime_a;
    uint64_t reverse_a;
    uint64_t reverse_prime_a;
} Candidate;

void merge_sort(Candidate* arr, size_t left, size_t right) {
    if (right <= left) return;
    size_t mid = left + (right - left) / 2;
    merge_sort(arr, left, mid);
    merge_sort(arr, mid + 1, right);
    size_t n1 = mid - left + 1;
    size_t n2 = right - mid;

    Candidate* L = malloc(n1 * sizeof(Candidate));
    Candidate* R = malloc(n2 * sizeof(Candidate));

    for (size_t i = 0; i < n1; i++) L[i] = arr[left + i];
    for (size_t j = 0; j < n2; j++) R[j] = arr[mid + 1 + j];
    arr[left + n1 + n2 - 1] = arr[right]; // Ensure the last element is copied

    // Merge the temporary arrays back into arr[left..right]
    size_t i = 0, j = 0, k = left;
    while (i < n1 && j < n2) {
        if (L[i].reverse_a <= R[j].reverse_a) {
            arr[k++] = L[i++];
        } else {
            arr[k++] = R[j++];
        }
    }
    while (i < n1) arr[k++] = L[i++];
    while (j < n2) arr[k++] = R[j++];
    free(L);
    free(R);
}


int main() {
    const int k = 11; // A_MAX is 10^k
    const uint64_t A_MAX = (uint64_t)pow(10, k);

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    // This is still needed to make sure the primesieve functions are optimized for our use case
    uint64_t LARGEST_PRIME_POSSIBLE;
    if (A_MAX == 100000000000) {
        // We already computed it, it is :
        LARGEST_PRIME_POSSIBLE = 2760727302517;
        // I am hard coding it because it takes 2minutes to compute it and I don't want to wait that long every time I run the program
    } else {
        LARGEST_PRIME_POSSIBLE = primesieve_nth_prime(A_MAX, 0);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);

    double time_taken = time_taken_between(start, end);
    printf("Largest prime possible is %llu\n", LARGEST_PRIME_POSSIBLE);
    printf("Time taken: %.9f seconds\n", time_taken);

    // Problem with this approach is that we do not know how many candidates we will have, we will first write them to a file and either
    // read them back to memory or sort them in the file, depending on how many candidates we have.

    // --- Parallel candidate search ---
    // primesieve_iterator can only generate primes in sequential order on a single thread,
    // so to parallelize we split the value range [3, LARGEST_PRIME_POSSIBLE] into one chunk
    // per thread, and give each thread its own iterator (this is the pattern recommended by
    // the primesieve docs, see the "Multi-threading" section of the C API).
    // The only extra piece of information each thread needs that a single sequential loop
    // gets "for free" is `a`, i.e. the rank/index of the first prime in its chunk. We get
    // that with primecount_pi(x), which counts primes <= x and is itself multi-threaded.
    clock_gettime(CLOCK_MONOTONIC, &start);

    size_t num_candidates = 0;
    size_t had_to_compute_bounds = 0;
    size_t had_to_compute_primality = 0;
    uint64_t total_primes_iterated = 0;

    int num_threads = omp_get_max_threads();
    printf("Searching for candidates using %d threads\n", num_threads);

    Candidate **thread_candidates = malloc(sizeof(Candidate*) * num_threads);
    size_t *thread_counts = malloc(sizeof(size_t) * num_threads);
    if (thread_candidates == NULL || thread_counts == NULL) {
        fprintf(stderr, "Error allocating memory for thread candidate buffers\n");
        return 1;
    }

    uint64_t dist = LARGEST_PRIME_POSSIBLE - 3 + 1;
    uint64_t chunk_dist = dist / (uint64_t)num_threads + 1;

    #pragma omp parallel for schedule(dynamic) \
        reduction(+:num_candidates, had_to_compute_bounds, had_to_compute_primality, total_primes_iterated)
    for (int t = 0; t < num_threads; t++) {
        uint64_t chunk_start = 3 + (uint64_t)t * chunk_dist;
        uint64_t chunk_stop = chunk_start + chunk_dist; // exclusive upper bound
        if (chunk_stop > LARGEST_PRIME_POSSIBLE + 1) chunk_stop = LARGEST_PRIME_POSSIBLE + 1;

        if (chunk_start > LARGEST_PRIME_POSSIBLE) {
            thread_candidates[t] = NULL;
            thread_counts[t] = 0;
            continue;
        }

        // Rank (1-based, prime(1)=2, prime(2)=3, ...) of the first prime >= chunk_start.
        uint64_t a = (uint64_t)primecount_pi((int64_t)(chunk_start - 1)) + 1;

        primesieve_iterator it_local;
        primesieve_init(&it_local);
        primesieve_jump_to(&it_local, chunk_start, chunk_stop);

        size_t local_cap = 4096;
        size_t local_count = 0;
        Candidate *local_candidates = malloc(sizeof(Candidate) * local_cap);
        if (local_candidates == NULL) {
            fprintf(stderr, "Error allocating memory for thread-local candidates\n");
            exit(1);
        }

        size_t local_bounds = 0, local_primality = 0;
        uint64_t local_iterated = 0;
        uint64_t prime_a;

        while ((prime_a = primesieve_next_prime(&it_local)) < chunk_stop) {
            local_iterated++;

            // Run the basic filters
            if (a % 10 == 0) {
                // If a ends with 0, then reverse(a) will start with 0 so will be an order of magnitude smaller
                // Because primes grow sequentially, this means that reverse(a) will be smaller than prime(a) and thus cannot be a solution
                a++;
                continue;
            }

            uint64_t reverse_of_prime_a = reverse(prime_a);

            if (reverse_of_prime_a % 2 == 0 || reverse_of_prime_a % 3 == 0 || reverse_of_prime_a % 5 == 0) {
                a++;
                continue;
            }

            uint64_t reverse_a = reverse(a);

            local_bounds++;
            if (!is_in_prime_bounds(reverse_a, reverse_of_prime_a)) {
                a++;
                continue;
            }

            local_primality++;

            if (!is_prime_mr(reverse_of_prime_a)) {
                a++;
                continue;
            }

            // If we reach here, we have a candidate
            if (local_count == local_cap) {
                local_cap *= 2;
                Candidate *tmp = realloc(local_candidates, sizeof(Candidate) * local_cap);
                if (tmp == NULL) {
                    fprintf(stderr, "Error reallocating memory for thread-local candidates\n");
                    exit(1);
                }
                local_candidates = tmp;
            }
            local_candidates[local_count++] = (Candidate){a, prime_a, reverse_a, reverse_of_prime_a};
            //printf("Found candidate: a = %llu, prime(a) = %llu, reverse(a) = %llu, reverse(prime(a)) = %llu\n", a, prime_a, reverse_a, reverse_of_prime_a);
            a++;
        }

        primesieve_free_iterator(&it_local);

        thread_candidates[t] = local_candidates;
        thread_counts[t] = local_count;

        num_candidates += local_count;
        had_to_compute_bounds += local_bounds;
        had_to_compute_primality += local_primality;
        total_primes_iterated += local_iterated;
    }

    // Merge every thread's local buffer into one contiguous array. The order in which
    // candidates end up in `candidates` doesn't matter: they get sorted by reverse_a right after this.
    Candidate *candidates = malloc(sizeof(Candidate) * (num_candidates > 0 ? num_candidates : 1));
    if (candidates == NULL) {
        fprintf(stderr, "Error allocating memory for candidates\n");
        return 1;
    }
    size_t offset = 0;
    for (int t = 0; t < num_threads; t++) {
        if (thread_counts[t] > 0) {
            memcpy(candidates + offset, thread_candidates[t], sizeof(Candidate) * thread_counts[t]);
            offset += thread_counts[t];
        }
        free(thread_candidates[t]);
    }
    free(thread_candidates);
    free(thread_counts);

    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("Finished iterating through primes up to %llu\n", LARGEST_PRIME_POSSIBLE);
    printf("Iterated over %llu primes\n", total_primes_iterated);
    time_taken = time_taken_between(start, end);
    printf("Time taken to check primes: %.9f seconds\n", time_taken);
    printf("This means %.9f primes per second\n", total_primes_iterated / time_taken);
    printf("Total candidates found: %zu\n", num_candidates);
    printf("This means %.9f primes per candidate, or that %.9f%% of the primes are candidates\n", (double)total_primes_iterated / (double)num_candidates, (double)num_candidates / (double)total_primes_iterated * 100.0);
    printf("We had to compute bounds for %zu candidates, leaving with %zu candidates that passed the bounds filter, removing %.9f%% of candidates tested\n", had_to_compute_bounds, had_to_compute_primality, (double)(had_to_compute_bounds - had_to_compute_primality) / (double)had_to_compute_bounds * 100.0);
    printf("We had to compute primality for %zu candidates, leaving with %zu candidates that passed the primality filter, removing %.9f%% of candidates tested\n", had_to_compute_primality, num_candidates, (double)(had_to_compute_primality - num_candidates) / (double)had_to_compute_primality * 100.0);
    
    // Now sort the candidates by their reverse value
    // I'll implement a simple merge sort for this, which I did above
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    merge_sort(candidates, 0, num_candidates - 1);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken = time_taken_between(start, end);
    printf("Time taken to sort candidates: %.9f seconds\n", time_taken);

    clock_gettime(CLOCK_MONOTONIC, &start);

    FILE* hits_file = fopen("hits.txt", "w");
    if (hits_file == NULL) {
        fprintf(stderr, "Error opening hits.txt for writing\n");
        free(candidates);
        return 1;
    } else {
        printf("Successfully opened hits.txt for writing\n");
        fprintf(hits_file, "------Search run A_MAX = 10^%d------\n", k);
    }

    // Now compute the reverse prime value for each candidate
    // For this we use a new iterator
    primesieve_iterator it2;
    primesieve_init(&it2);
    primesieve_jump_to(&it2, 2, LARGEST_PRIME_POSSIBLE);
    // We will now iterate through the candidates and compute their reverse prime value
    uint64_t current_prime;
    uint64_t a = 1; // We start with the first prime, which is 2, and we already have it in current_prime
    uint64_t candidate_index = 0; // Index for candidates array
    while ((current_prime = primesieve_next_prime(&it2)) <= LARGEST_PRIME_POSSIBLE) {
        // When a matches the reverse_a of candidate, we check if current_prime matches the reverse_prime_a of candidate
        if (candidate_index >= num_candidates) {
            break; // No more candidates to check
        }
        if (candidates[candidate_index].reverse_a == a) {
            if (candidates[candidate_index].reverse_prime_a == current_prime) {
                fprintf(hits_file, "Found a = %llu, pi(a) = %llu, reverse(a) = %llu, pi(reverse(a)) = %llu\n", candidates[candidate_index].a, candidates[candidate_index].prime_a, candidates[candidate_index].reverse_a, current_prime);
                printf("Found a = %llu, pi(a) = %llu, reverse(a) = %llu, pi(reverse(a)) = %llu\n", candidates[candidate_index].a, candidates[candidate_index].prime_a, candidates[candidate_index].reverse_a, current_prime);   
            }
            candidate_index++;
        }
        a++;
    }

    free(candidates);
    fclose(hits_file);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken = time_taken_between(start, end);
    printf("Time taken to compute reverse prime values: %.9f seconds\n", time_taken);

    return 0;
}#include "primesieve.h"
#include "primecount.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
/*
This is a new version of the prime_reverse_v3.c file. We will now use a completly new approach to try to find a(10)
We will iterate through all primes like before, filtering those who can't be solutions based on basic filters like in the previous version.
Every index that passes the filters will be a candidate.
We will then sort the candidates by their reverse value and compute their reverse prime value.
Because we will already have computed a, pi(a) and reverse(a), computing pi(reverse(a)) will be easy and fast considering there will be
only "a few" candidates, and that because it is sorted we won't have to do a random access, in fact there probably won't even be
a need for a file because computing on the fly might be fast enough, or even could be done on the GPU if we want to go that far.

NOTE : After some testing, the number of candidates stays around the thousands, maybe millions at worst, but it fits in RAM easily.

I will reuse a lot of code from the previous version because I want the file to be independent.
*/

// Fast modular multiplication using 128-bit integers
inline uint64_t mul_mod(uint64_t a, uint64_t b, uint64_t m) {
    return (uint64_t)(((__uint128_t)a * b) % m);
}

// Fast modular exponentiation
uint64_t power_mod(uint64_t base, uint64_t exp, uint64_t mod) {
    uint64_t res = 1;
    base %= mod;
    while (exp > 0) {
        if (exp % 2 == 1) res = mul_mod(res, base, mod);
        base = mul_mod(base, base, mod);
        exp /= 2;
    }
    return res;
}

// Deterministic Miller-Rabin up to 3.3 x 10^24 (using 7 bases)
bool is_prime_mr(uint64_t n) {
    if (n < 2) return false;
    if (n == 2 || n == 3 || n == 5) return true;
    if (n % 2 == 0 || n % 3 == 0 || n % 5 == 0) return false;

    uint64_t d = n - 1;
    int s = 0;
    while (d % 2 == 0) {
        d /= 2;
        s++;
    }

    static const uint64_t bases[] = {2, 3, 5, 7, 11, 13, 17};
    for (int i = 0; i < 7; i++) {
        uint64_t a = bases[i];
        if (n <= a) break;
        
        uint64_t x = power_mod(a, d, n);
        if (x == 1 || x == n - 1) continue;
        
        bool composite = true;
        for (int r = 1; r < s; r++) {
            x = mul_mod(x, x, n);
            if (x == n - 1) {
                composite = false;
                break;
            }
        }
        if (composite) return false;
    }
    return true;
}

bool is_in_prime_bounds(uint64_t n, uint64_t candidate_prime) {
    if (n < 6) return true; // Bounds math is slightly off for tiny primes, allow them

    double dn = (double)n;
    double ln_n = log(dn);
    double ln_ln_n = log(ln_n);

    // Dusart 2010 Lower Bound (valid for n >= 3)
    double lower_factor = ln_n + ln_ln_n - 1.0 + (ln_ln_n - 2.1) / ln_n;
    double upper_factor;

    // Dusart 2010 Upper Bound (valid for n >= 688383)
    if (n >= 688383) {
        upper_factor = ln_n + ln_ln_n - 1.0 + (ln_ln_n - 2.0) / ln_n;
    } else {
        // Safe fallback upper bound for smaller 'n'
        upper_factor = ln_n + ln_ln_n; 
    }

    // Apply a 0.001 margin to protect against IEEE 754 float inaccuracies
    uint64_t min_possible = (uint64_t)(dn * (lower_factor - 0.001));
    uint64_t max_possible = (uint64_t)(dn * (upper_factor + 0.001)) + 1;

    return (candidate_prime >= min_possible && candidate_prime <= max_possible);
}

uint64_t reverse(uint64_t n) {
    uint64_t reversed = 0;
    while (n > 0) {
        reversed = reversed * 10 + n % 10;
        n /= 10;
    }
    return reversed;
}

double time_taken_between(struct timespec start, struct timespec end) {
    double time_taken = (end.tv_sec - start.tv_sec) * 1e9;
    time_taken = (time_taken + (end.tv_nsec - start.tv_nsec)) * 1e-9;
    return time_taken;
}

typedef struct {
    uint64_t a;
    uint64_t prime_a;
    uint64_t reverse_a;
    uint64_t reverse_prime_a;
} Candidate;

void merge_sort(Candidate* arr, size_t left, size_t right) {
    if (right <= left) return;
    size_t mid = left + (right - left) / 2;
    merge_sort(arr, left, mid);
    merge_sort(arr, mid + 1, right);
    size_t n1 = mid - left + 1;
    size_t n2 = right - mid;

    Candidate* L = malloc(n1 * sizeof(Candidate));
    Candidate* R = malloc(n2 * sizeof(Candidate));

    for (size_t i = 0; i < n1; i++) L[i] = arr[left + i];
    for (size_t j = 0; j < n2; j++) R[j] = arr[mid + 1 + j];
    arr[left + n1 + n2 - 1] = arr[right]; // Ensure the last element is copied

    // Merge the temporary arrays back into arr[left..right]
    size_t i = 0, j = 0, k = left;
    while (i < n1 && j < n2) {
        if (L[i].reverse_a <= R[j].reverse_a) {
            arr[k++] = L[i++];
        } else {
            arr[k++] = R[j++];
        }
    }
    while (i < n1) arr[k++] = L[i++];
    while (j < n2) arr[k++] = R[j++];
    free(L);
    free(R);
}


int main() {
    const int k = 11; // A_MAX is 10^k
    const uint64_t A_MAX = (uint64_t)pow(10, k);

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    // This is still needed to make sure the primesieve functions are optimized for our use case
    uint64_t LARGEST_PRIME_POSSIBLE;
    if (A_MAX == 100000000000) {
        // We already computed it, it is :
        LARGEST_PRIME_POSSIBLE = 2760727302517;
        // I am hard coding it because it takes 2minutes to compute it and I don't want to wait that long every time I run the program
    } else {
        LARGEST_PRIME_POSSIBLE = primesieve_nth_prime(A_MAX, 0);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);

    double time_taken = time_taken_between(start, end);
    printf("Largest prime possible is %llu\n", LARGEST_PRIME_POSSIBLE);
    printf("Time taken: %.9f seconds\n", time_taken);

    clock_gettime(CLOCK_MONOTONIC, &start);
    primesieve_iterator it;

    primesieve_init(&it);
    primesieve_jump_to(&it, 3, LARGEST_PRIME_POSSIBLE);
    clock_gettime(CLOCK_MONOTONIC, &end);

    time_taken = time_taken_between(start, end);
    printf("Time taken to initialize iterator: %.9f seconds\n", time_taken);

    // Problem with this approach is that we do not know how many candidates we will have, we will first write them to a file and either
    // read them back to memory or sort them in the file, depending on how many candidates we have.

    clock_gettime(CLOCK_MONOTONIC, &start);
    uint64_t prime_a;
    uint64_t a = 2;
    size_t num_candidates = 0; // For now we'll count them
    size_t had_to_compute_bounds = 0; // For now we'll count them
    size_t had_to_compute_primality = 0; // For now we'll count them


    Candidate *candidates = malloc(sizeof(Candidate) * 1000000);
    if (candidates == NULL) {
        fprintf(stderr, "Error allocating memory for candidates\n");
        return 1;
    }
    while ((prime_a = primesieve_next_prime(&it)) <= LARGEST_PRIME_POSSIBLE) {
        // Run the basic filters
        if (a % 10 == 0) { 
            // If a ends with 0, then reverse(a) will start with 0 so will be an order of magnitude smaller 
            // Because primes grow sequentially, this means that reverse(a) will be smaller than prime(a) and thus cannot be a solution
            a++;
            continue;
        }

        uint64_t reverse_of_prime_a = reverse(prime_a);

        if (reverse_of_prime_a % 2 == 0 || reverse_of_prime_a % 3 == 0 || reverse_of_prime_a % 5 == 0) {
            a++;
            continue;
        }

        uint64_t reverse_a = reverse(a);

        had_to_compute_bounds++;
        if (!is_in_prime_bounds(reverse_a, reverse_of_prime_a)) {
            a++;
            continue;
        }


        had_to_compute_primality++;

        if (!is_prime_mr(reverse_of_prime_a)) {
            a++;
            continue;
        }

        // If we reach here, we have a candidate
        num_candidates++;
        Candidate candidate = {a, prime_a, reverse_a, reverse_of_prime_a};
        candidates[num_candidates - 1] = candidate;
        //printf("Found candidate: a = %llu, prime(a) = %llu, reverse(a) = %llu, reverse(prime(a)) = %llu\n", a, prime_a, reverse_a, reverse_of_prime_a);
        a++;
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("Finished iterating through primes up to %llu\n", LARGEST_PRIME_POSSIBLE);
    printf("Iterated over %llu primes\n", a - 1);
    time_taken = time_taken_between(start, end);
    printf("Time taken to check primes: %.9f seconds\n", time_taken);
    printf("This means %.9f primes per second\n", (a - 1) / time_taken);
    printf("Total candidates found: %zu\n", num_candidates);
    printf("This means %.9f primes per candidate, or that %.9f%% of the primes are candidates\n", (double)(a - 1) / (double)num_candidates, (double)num_candidates / (double)(a - 1) * 100.0);
    printf("We had to compute bounds for %zu candidates, leaving with %zu candidates that passed the bounds filter, removing %.9f%% of candidates tested\n", had_to_compute_bounds, had_to_compute_primality, (double)(had_to_compute_bounds - had_to_compute_primality) / (double)had_to_compute_bounds * 100.0);
    printf("We had to compute primality for %zu candidates, leaving with %zu candidates that passed the primality filter, removing %.9f%% of candidates tested\n", had_to_compute_primality, num_candidates, (double)(had_to_compute_primality - num_candidates) / (double)had_to_compute_primality * 100.0);
    
    // Now sort the candidates by their reverse value
    // I'll implement a simple merge sort for this, which I did above
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    merge_sort(candidates, 0, num_candidates - 1);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken = time_taken_between(start, end);
    printf("Time taken to sort candidates: %.9f seconds\n", time_taken);

    clock_gettime(CLOCK_MONOTONIC, &start);

    FILE* hits_file = fopen("hits.txt", "w");
    if (hits_file == NULL) {
        fprintf(stderr, "Error opening hits.txt for writing\n");
        free(candidates);
        return 1;
    } else {
        printf("Successfully opened hits.txt for writing\n");
        fprintf(hits_file, "------Search run A_MAX = 10^%d------\n", k);
    }

    // Now compute the reverse prime value for each candidate
    // For this we use a new iterator
    primesieve_iterator it2;
    primesieve_init(&it2);
    primesieve_jump_to(&it2, 2, LARGEST_PRIME_POSSIBLE);
    // We will now iterate through the candidates and compute their reverse prime value
    uint64_t current_prime;
    a = 1; // We start with the first prime, which is 2, and we already have it in current_prime
    uint64_t candidate_index = 0; // Index for candidates array
    while ((current_prime = primesieve_next_prime(&it2)) <= LARGEST_PRIME_POSSIBLE) {
        // When a matches the reverse_a of candidate, we check if current_prime matches the reverse_prime_a of candidate
        if (candidate_index >= num_candidates) {
            break; // No more candidates to check
        }
        if (candidates[candidate_index].reverse_a == a) {
            if (candidates[candidate_index].reverse_prime_a == current_prime) {
                fprintf(hits_file, "Found a = %llu, pi(a) = %llu, reverse(a) = %llu, pi(reverse(a)) = %llu\n", candidates[candidate_index].a, candidates[candidate_index].prime_a, candidates[candidate_index].reverse_a, current_prime);
                printf("Found a = %llu, pi(a) = %llu, reverse(a) = %llu, pi(reverse(a)) = %llu\n", candidates[candidate_index].a, candidates[candidate_index].prime_a, candidates[candidate_index].reverse_a, current_prime);   
            }
            candidate_index++;
        }
        a++;
    }

    free(candidates);
    fclose(hits_file);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken = time_taken_between(start, end);
    printf("Time taken to compute reverse prime values: %.9f seconds\n", time_taken);

    return 0;
}
