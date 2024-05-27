#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

#define BLOCK_SIZE 8
#define COUNTER_MAX -1
#define N 4096
#define TEST_MODE 0

int find_local_col_index(int i, int block_size, int rank, int size) {
    int global_block_index = i / block_size;
    int relative_col_index_in_block = i % block_size;
    int local_block_index = global_block_index / size;
    return local_block_index * block_size + relative_col_index_in_block;
}

int find_global_col_index(int local_col_index, int block_size, int rank, int size) {
    int local_block_index = local_col_index / block_size;
    int relative_col_index_in_block = local_col_index % block_size;
    int global_block_index = local_block_index * size + rank;
    return global_block_index * block_size + relative_col_index_in_block;
}

void gaussian_elimination(int n, double A[][n]) {
    int i, j;
    double temp;
    double epsilon = 1e-10; // Define epsilon for near-singularity checks
    int counter = 0;

    for (i = 0; i < n - 1; i++) {
        int k = i;

        // Find the row with maximum absolute value in the column i
        for (j = i + 1; j < n; j++) {
            if (fabs(A[j][i]) > fabs(A[k][i])) {
                k = j;
            }
        }

        // Check if A is singular or nearly so
        if (fabs(A[k][i]) < epsilon) {
            printf("Matrix is singular or nearly singular\n");
            return;
        }

        // Swap rows i and k
        if (k != i) {
            for (j = 0; j < n; j++) {
                temp = A[i][j];
                A[i][j] = A[k][j];
                A[k][j] = temp;
            }
        }

        // Store multiplier in place of A(j,i)
        for (j = i + 1; j < n; j++) {
            A[j][i] /= A[i][i];
        }

        // Subtract multiple of row A(i,:) to zero out A(j,i)
        for (j = i + 1; j < n; j++) {
            for (k = i + 1; k < n; k++) {
                A[j][k] -= A[j][i] * A[i][k];
                /*printf("In %d: A[%d][%d] = %0.2f\n", i, j, k, A[j][k]);*/
            }

            A[j][i] = 0;
        }

        counter ++;

        if (counter == COUNTER_MAX) {
            return;
        }
    }
}

void gaussian_elimination_mpi2(int n, double A[][n], int rank, int size, int block_size, int b, double P[][b]) {
    int i, j;
    double temp;
    double epsilon = 1e-10; // Define epsilon for near-singularity checks
    int counter = 0;

    for (i = 0; i < n - 1; i++) {
        int k = i;
        int owning_rank = (i / block_size) % size;
        int local_col_index = find_local_col_index(i, block_size, rank, size);

        // Find the row with maximum absolute value in the column i
        if (rank == owning_rank) {

            for (j = i + 1; j < n; j++) {
                if (fabs(P[j][local_col_index]) > fabs(P[k][local_col_index])) {
                    k = j;
                }
            }

            // Check if A is singular or nearly so
            if (fabs(P[k][local_col_index]) < epsilon) {
                printf("Matrix is singular or nearly singular\n");
                return;
            }
        }


        MPI_Bcast(&k, 1, MPI_INT, owning_rank, MPI_COMM_WORLD);

        // Swap rows i and k
        if (k != i) {
            for (j = 0; j < b; j++) {
                temp = P[i][j];
                P[i][j] = P[k][j];
                P[k][j] = temp;
            }
        }

        /*printf("Process %d P (b = %d):\n", rank, b);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < b; j++) {
                printf("%0.2f ", P[i][j]);
            }
            printf("\n");
        }

        printf("\n");*/

        double *ratio_list = malloc((n - 1 - i) * sizeof(double));
        int index = 0;
        for (j = i + 1; j < n; j++) {
            if (rank == owning_rank) {
                double ratio = P[j][local_col_index] / P[i][local_col_index];
                ratio_list[index++] = ratio;
                P[j][local_col_index] = 0;
            }
        }

        MPI_Bcast(ratio_list, n - 1 - i, MPI_DOUBLE, owning_rank, MPI_COMM_WORLD);

        /*printf("Process %d P (b = %d):\n", rank, b);
        for (int k = 0; k < (n - 1 - i); k++) {
            printf("%0.2f ", ratio_list[k]);
        }
        printf("\n");*/

        // Subtract multiple of row A(i,:) to zero out A(j,i)
        for (j = i + 1; j < n; j += block_size) {
            for (int jj = j; jj < n && jj < j + block_size; jj++) {
                double ratio = ratio_list[jj - i - 1];
                for (k = 0; k < b; k++) {
                    int global_col_index = find_global_col_index(k, block_size, rank, size);
                    if (global_col_index > i) {
                        P[jj][k] -= ratio * P[i][k];
                    }

                }
            }
        }


        if (counter == COUNTER_MAX) {
            printf("Process %d P (b = %d):\n", rank, b);
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < b; j++) {
                    printf("%0.2f ", P[i][j]);
                }
                printf("\n");
            }

            printf("\n");
        }

        free(ratio_list);

        counter++;

        if (counter == COUNTER_MAX) {
            return;
        }

    }


    for (int i = 0; i < n; i++) {
        for (int j = 0; j < b; j++) {
            int col = find_global_col_index(j, block_size, rank, size);
            if (col < n) {
                A[i][col] = P[i][j];
                /*printf("col: %d in rank %d\n", col, rank);*/
            }
        }
    }

    MPI_Datatype column_type;
    MPI_Type_vector(n, block_size, n, MPI_DOUBLE, &column_type);
    MPI_Type_commit(&column_type);

    int num_bcasts = n / block_size;

    MPI_Request *request = (MPI_Request *)malloc(num_bcasts * sizeof(MPI_Request));

    for (int c = 0; c < num_bcasts; c ++){
        int col_owner = c % size;
        MPI_Ibcast(&A[0][c * block_size], 1, column_type, col_owner, MPI_COMM_WORLD, &request[c]);
    }

    MPI_Waitall(num_bcasts, request, MPI_STATUS_IGNORE);

    MPI_Type_free(&column_type);
    free(request);

}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n;
    int block_size;

    if (rank == 0) {
        printf("Enter the size of the matrix A: ");
        fflush(stdout);
        int state = scanf("%d", &n);
        if (state != 1) {
            MPI_Abort(MPI_COMM_WORLD, -1);
        }

        printf("Enter the block size: ");
        fflush(stdout);
        state = scanf("%d", &block_size);
        if (state != 1) {
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&block_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int num_blocks = (n + block_size - 1) / block_size;
    int blocks_per_proc = (num_blocks + size - 1) / size;
    int b = blocks_per_proc * block_size;

    double (*matrix)[n] = malloc(n * sizeof((*matrix)));
    double (*matrix2)[n] = malloc(n * sizeof((*matrix2)));
    double (*partition)[b] = malloc(n * sizeof(*partition));

    if (matrix == NULL) {
        printf("Memory allocation failed\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    srand(time(NULL));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int number = rand() % 100;
            matrix[i][j] = number;
            matrix2[i][j] = number;

            /*if (rank == 0){
                printf("%4d ", number);
            }*/

        }

        /*if (rank == 0){
            printf("\n");
        }*/
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < b; j++) {
            partition[i][j] = 0.0;
        }
    }

    int block_count = 0;
    for (int block_idx = rank; block_idx < num_blocks; block_idx += size) {
        int col_start = block_idx * block_size;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < block_size; j++) {
                int col = col_start + j;
                if (col < n) {
                    partition[i][block_count * block_size + j] = matrix[i][col];
                }
            }
        }
        block_count++;
    }

    if (partition == NULL) {
        printf("Memory allocation failed\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    gaussian_elimination_mpi2(n, matrix, rank, size, block_size, b, partition);
    clock_gettime(CLOCK_MONOTONIC, &end);

    MPI_Finalize();

    if (TEST_MODE && rank == 0) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                printf("%6.2f ", matrix[i][j]);
            }
            printf("\n");
        }

        printf("\n");
    }


    double elapsedTime = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    if (rank == 0) {
        printf("MPI function execution time: %f seconds\n", elapsedTime);
    }

    if (rank == 0) {
        struct timespec start2, end2;
        clock_gettime(CLOCK_MONOTONIC, &start2);
        gaussian_elimination(n, matrix2);
        clock_gettime(CLOCK_MONOTONIC, &end2);

        if (TEST_MODE) {
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    printf("%6.2f ", matrix2[i][j]);
                }
                printf("\n");
            }
        }

        double elapsedTime2 = (end2.tv_sec - start2.tv_sec) + (end2.tv_nsec - start2.tv_nsec) / 1e9;

        printf("Function execution time: %f seconds\n", elapsedTime2);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (matrix2[i][j] != matrix[i][j]) {
                    printf("Results are different!\n");
                    free(matrix);
                    free(matrix2);

                    return 0;
                }

            }
        }


        printf("Percentage of improvement: %0.2f\n", (elapsedTime2 - elapsedTime) * 100 / elapsedTime2);
    }


    free(matrix);
    free(matrix2);
    free(partition);

    return 0;
}
