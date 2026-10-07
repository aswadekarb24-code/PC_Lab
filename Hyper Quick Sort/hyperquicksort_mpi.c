/*
 * hyperquicksort_mpi.c
 * Parallel Hyperquicksort using MPI (Quinn, Parallel Programming in C
 * with MPI and OpenMP, Sec. 14.4).
 *
 * Compile : mpicc -O2 -o hyperquicksort_mpi hyperquicksort_mpi.c
 * Run     : mpirun -np 4 ./hyperquicksort_mpi [N]      (N default = 10,000,000)
 * Note    : number of processes must be a power of 2 (2, 4, 8, ...)
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---------- helpers ---------- */
static unsigned long long rng_state = 88172645463325252ULL;
static inline int next_rand(void) {               /* xorshift64, value in [0, 2^31-1] */
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (int)(rng_state >> 33);
}

static int cmp_int(const void *a, const void *b) {
    int x = *(const int *)a, y = *(const int *)b;
    return (x > y) - (x < y);
}

/* index of the first element strictly greater than v in sorted array a[0..n) */
static long upper_bound(const int *a, long n, int v) {
    long lo = 0, hi = n;
    while (lo < hi) {
        long mid = lo + (hi - lo) / 2;
        if (a[mid] <= v) lo = mid + 1; else hi = mid;
    }
    return lo;
}

/* merge two sorted arrays into out (size na+nb) */
static void merge(const int *a, long na, const int *b, long nb, int *out) {
    long i = 0, j = 0, k = 0;
    while (i < na && j < nb) out[k++] = (a[i] <= b[j]) ? a[i++] : b[j++];
    while (i < na) out[k++] = a[i++];
    while (j < nb) out[k++] = b[j++];
}

int main(int argc, char *argv[]) {
    int rank, p;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    long N = (argc > 1) ? atol(argv[1]) : 10000000L;

    if (p & (p - 1)) {
        if (rank == 0) fprintf(stderr, "Number of processes must be a power of 2.\n");
        MPI_Finalize();
        return 1;
    }
    int logp = 0;
    while ((1 << logp) < p) logp++;

    /* ---------- Step 0: generate data on P0 and divide among processes ---------- */
    long base = N / p, rem = N % p;
    long n = base + (rank < rem ? 1 : 0);               /* local element count */
    int *a = (int *)malloc((n > 0 ? n : 1) * sizeof(int));

    int *seq = NULL;                                     /* copy for sequential baseline */
    long long sum_before = 0;
    if (rank == 0) {
        int *data = (int *)malloc(N * sizeof(int));
        for (long i = 0; i < N; i++) { data[i] = next_rand(); sum_before += data[i]; }
        seq = (int *)malloc(N * sizeof(int));
        memcpy(seq, data, N * sizeof(int));
        memcpy(a, data, n * sizeof(int));
        long off = n;
        for (int r = 1; r < p; r++) {
            long cnt = base + (r < rem ? 1 : 0);
            MPI_Send(data + off, (int)cnt, MPI_INT, r, 0, MPI_COMM_WORLD);
            off += cnt;
        }
        free(data);
    } else {
        MPI_Recv(a, (int)n, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    /* ---------- Sequential baseline (rank 0 only) ---------- */
    double t_seq = 0.0;
    if (rank == 0) {
        double s = MPI_Wtime();
        qsort(seq, N, sizeof(int), cmp_int);
        t_seq = MPI_Wtime() - s;
        free(seq);
    }

    /* ---------- Parallel hyperquicksort ---------- */
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    /* Step 1: each process sorts its own list with quicksort */
    qsort(a, n, sizeof(int), cmp_int);

    /* Steps 2-5: log p split-and-merge phases, highest hypercube dimension first */
    for (int d = logp - 1; d >= 0; d--) {
        int group_root = rank & ~((1 << (d + 1)) - 1);   /* lowest rank of this sub-hypercube */
        int pivot = 0;

        /* Step 2: group root picks the median of its (sorted) list and broadcasts it */
        if (rank == group_root) {
            pivot = (n > 0) ? a[n / 2] : 0;
            for (int r = group_root + 1; r < group_root + (1 << (d + 1)); r++)
                MPI_Send(&pivot, 1, MPI_INT, r, 1, MPI_COMM_WORLD);
        } else {
            MPI_Recv(&pivot, 1, MPI_INT, group_root, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        /* Step 3: split local list into low (<= pivot) and high (> pivot) parts */
        long idx = upper_bound(a, n, pivot);
        int partner = rank ^ (1 << d);
        int lower_half = ((rank >> d) & 1) == 0;

        int *send_ptr; long send_cnt; int *keep_ptr; long keep_cnt;
        if (lower_half) { send_ptr = a + idx; send_cnt = n - idx; keep_ptr = a;       keep_cnt = idx; }
        else            { send_ptr = a;       send_cnt = idx;     keep_ptr = a + idx; keep_cnt = n - idx; }

        /* Step 4: swap lists with partner (lower half sends high list, upper half sends low list) */
        long recv_cnt;
        MPI_Sendrecv(&send_cnt, 1, MPI_LONG_LONG, partner, 2,
                     &recv_cnt, 1, MPI_LONG_LONG, partner, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        int *recv_buf = (int *)malloc((recv_cnt > 0 ? recv_cnt : 1) * sizeof(int));
        MPI_Sendrecv(send_ptr, (int)send_cnt, MPI_INT, partner, 3,
                     recv_buf, (int)recv_cnt, MPI_INT, partner, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* Step 5: merge kept sorted list with received sorted list */
        long new_n = keep_cnt + recv_cnt;
        int *merged = (int *)malloc((new_n > 0 ? new_n : 1) * sizeof(int));
        merge(keep_ptr, keep_cnt, recv_buf, recv_cnt, merged);
        free(recv_buf);
        free(a);
        a = merged;
        n = new_n;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_local = MPI_Wtime() - t0, t_par = 0.0;
    MPI_Reduce(&t_local, &t_par, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    /* ---------- Verification ---------- */
    long long info[4];                       /* count, min, max, locally-sorted flag */
    info[0] = n; info[1] = n ? a[0] : 0; info[2] = n ? a[n - 1] : 0; info[3] = 1;
    for (long i = 1; i < n; i++) if (a[i - 1] > a[i]) { info[3] = 0; break; }

    long long sum_local = 0, sum_after = 0;
    for (long i = 0; i < n; i++) sum_local += a[i];
    MPI_Reduce(&sum_local, &sum_after, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        MPI_Send(info, 4, MPI_LONG_LONG, 0, 4, MPI_COMM_WORLD);
    } else {
        int ok = 1; long total = 0; long long prev_max = -1;
        printf("Hyperquicksort with MPI: N = %ld, processes = %d\n", N, p);
        for (int r = 0; r < p; r++) {
            long long x[4];
            if (r == 0) memcpy(x, info, sizeof(x));
            else MPI_Recv(x, 4, MPI_LONG_LONG, r, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("  P%d holds %lld elements", r, x[0]);
            if (x[0] > 0) printf("  [min=%lld, max=%lld]", x[1], x[2]);
            printf("\n");
            total += x[0];
            if (!x[3]) ok = 0;
            if (x[0] > 0) { if (x[1] < prev_max) ok = 0; prev_max = x[2]; }
        }
        if (total != N || sum_after != sum_before) ok = 0;
        printf("Sorted correctly        : %s\n", ok ? "YES" : "NO");
        printf("Sequential qsort time   : %.4f s\n", t_seq);
        printf("Parallel hyperquicksort : %.4f s\n", t_par);
        printf("Speedup                 : %.2f\n", t_seq / t_par);
        printf("Efficiency              : %.2f\n", t_seq / t_par / p);
    }

    free(a);
    MPI_Finalize();
    return 0;
}
