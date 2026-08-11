#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
double get_wall_time()
{
    struct timespec ts;
    // CLOCK_MONOTONIC is immune to system time changes (e.g., NTP adjustments)
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + (ts.tv_nsec * 1e-9);
}
// Define the number of threads (OpenMP handles this dynamically by default)
#define NUM_THREADS 1000

// The function that each thread will execute
void *hello_world_thread(void *arg)
{
    // Cast the void pointer back to an integer to get our thread ID
    int tid = *((int *)arg);

    printf("Hello World from thread = %d\n", tid);

    // Only "master" thread (ID 0) does this
    if (tid == 0)
    {
        // Since pthreads doesn't have an omp_get_num_threads() equivalent,
        // we reference the total we manually created.
        printf("Number of threads = %d\n", NUM_THREADS);
    }

    pthread_exit(NULL);
}

int main()
{
    double start = get_wall_time();
    pthread_t threads[NUM_THREADS];
    int thread_ids[NUM_THREADS];

    /* Fork a team of threads giving them their own copies of variables */
    // OpenMP does this in one line; here we loop and spawn manually.
    for (int i = 0; i < NUM_THREADS; i++)
    {
        thread_ids[i] = i; // Assign ID

        // Create the thread and pass the address of its ID
        if (pthread_create(&threads[i], NULL, hello_world_thread, (void *)&thread_ids[i]) != 0)
        {
            printf("Error creating thread %d\n", i);
            exit(-1);
        }
    }

    /* All threads join master thread and disband */
    // OpenMP does this implicitly at the closing brace of the #pragma block.
    // With pthreads, we must explicitly wait for each to finish.
    for (int i = 0; i < NUM_THREADS; i++)
    {
        pthread_join(threads[i], NULL);
    }

    double end = get_wall_time();
    printf("OpenMP Elapsed Time: %.6f seconds\n", end - start);
    return 0;
}