/*
 * hyperquicksort_omp.c
 * Parallel Hyperquicksort using OpenMP (shared-memory version).
 * Each OpenMP thread plays the role of one "process" of the hypercube and owns a
 * private sorted list; "message passing" is replaced by reading the partner's
 * published list between barriers.
 *
 * Compile : gcc -O2 -fopenmp -o hyperquicksort_omp hyperquicksort_omp.c
 * Run     : ./hyperquicksort_omp [N] [P]     (N default = 10,000,000, P default = 4)
 * Note    : P must be a power of 2
 */
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned long long rng_state = 88172645463325252ULL;
static inline int next_rand(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (int)(rng_state >> 33);
}
static int cmp_int(const void *a, const void *b) {
    int x = *(const int *)a, y = *(const int *)b;
    return (x > y) - (x < y);
}
static long upper_bound(const int *a, long n, int v) {
    long lo = 0, hi = n;
    while (lo < hi) { long mid = lo + (hi - lo) / 2; if (a[mid] <= v) lo = mid + 1; else hi = mid; }
    return lo;
}
static void merge(const int *a, long na, const int *b, long nb, int *out) {
    long i = 0, j = 0, k = 0;
    while (i < na && j < nb) out[k++] = (a[i] <= b[j]) ? a[i++] : b[j++];
    while (i < na) out[k++] = a[i++];
    while (j < nb) out[k++] = b[j++];
}

int main(int argc, char *argv[]) {
    long N = (argc > 1) ? atol(argv[1]) : 10000000L;
    int  P = (argc > 2) ? atoi(argv[2]) : 4;
    if (P < 1 || (P & (P - 1))) { fprintf(stderr, "P must be a power of 2.\n"); return 1; }
    int logp = 0; while ((1 << logp) < P) logp++;

    /* Generate data and a copy for the sequential baseline */
    int *data = (int *)malloc(N * sizeof(int));
    int *seq  = (int *)malloc(N * sizeof(int));
    long long sum_before = 0;
    for (long i = 0; i < N; i++) { data[i] = next_rand(); sum_before += data[i]; }
    memcpy(seq, data, N * sizeof(int));

    double s = omp_get_wtime();
    qsort(seq, N, sizeof(int), cmp_int);
    double t_seq = omp_get_wtime() - s;
    free(seq);

    /* Shared "mailboxes": each thread publishes its list, length, split index, pivot */
    int  **buf   = (int **)calloc(P, sizeof(int *));
    long  *len   = (long *)calloc(P, sizeof(long));
    long  *split = (long *)calloc(P, sizeof(long));
    int   *pivot = (int *)calloc(P, sizeof(int));
    long base = N / P, rem = N % P;

    double t0 = omp_get_wtime();
    #pragma omp parallel num_threads(P)
    {
        int r = omp_get_thread_num();
        long n = base + (r < rem ? 1 : 0);
        long off = r * base + (r < rem ? r : rem);

        /* Step 1: take own portion of the list and sort it with quicksort */
        int *a = (int *)malloc((n > 0 ? n : 1) * sizeof(int));
        memcpy(a, data + off, n * sizeof(int));
        qsort(a, n, sizeof(int), cmp_int);
        buf[r] = a; len[r] = n;
        #pragma omp barrier

        for (int d = logp - 1; d >= 0; d--) {
            int root = r & ~((1 << (d + 1)) - 1);
            /* Step 2: group root publishes its median as the pivot */
            if (r == root) pivot[r] = (len[r] > 0) ? buf[r][len[r] / 2] : 0;
            #pragma omp barrier
            /* Step 3: split own list around the pivot */
            split[r] = upper_bound(buf[r], len[r], pivot[root]);
            #pragma omp barrier
            /* Steps 4+5: read the partner's relevant part and merge with the kept part */
            int partner = r ^ (1 << d);
            int lower_half = ((r >> d) & 1) == 0;
            int *keep; long keep_n; int *recv; long recv_n;
            if (lower_half) {                       /* keep low, take partner's low */
                keep = buf[r];               keep_n = split[r];
                recv = buf[partner];         recv_n = split[partner];
            } else {                                /* keep high, take partner's high */
                keep = buf[r] + split[r];    keep_n = len[r] - split[r];
                recv = buf[partner] + split[partner]; recv_n = len[partner] - split[partner];
            }
            long new_n = keep_n + recv_n;
            int *merged = (int *)malloc((new_n > 0 ? new_n : 1) * sizeof(int));
            merge(keep, keep_n, recv, recv_n, merged);
            #pragma omp barrier                     /* everyone finished reading old lists */
            free(buf[r]);
            buf[r] = merged; len[r] = new_n;
            #pragma omp barrier
        }
    }
    double t_par = omp_get_wtime() - t0;

    /* Verification */
    int ok = 1; long total = 0; long long sum_after = 0; long long prev = -1;
    printf("Hyperquicksort with OpenMP: N = %ld, threads = %d\n", N, P);
    for (int r = 0; r < P; r++) {
        printf("  T%d holds %ld elements", r, len[r]);
        if (len[r] > 0) printf("  [min=%d, max=%d]", buf[r][0], buf[r][len[r] - 1]);
        printf("\n");
        for (long i = 0; i < len[r]; i++) {
            if (buf[r][i] < prev) ok = 0;
            prev = buf[r][i]; sum_after += buf[r][i];
        }
        total += len[r];
    }
    if (total != N || sum_after != sum_before) ok = 0;
    printf("Sorted correctly        : %s\n", ok ? "YES" : "NO");
    printf("Sequential qsort time   : %.4f s\n", t_seq);
    printf("Parallel hyperquicksort : %.4f s\n", t_par);
    printf("Speedup                 : %.2f\n", t_seq / t_par);
    printf("Efficiency              : %.2f\n", t_seq / t_par / P);

    for (int r = 0; r < P; r++) free(buf[r]);
    free(buf); free(len); free(split); free(pivot); free(data);
    return 0;
}
