#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include <math.h>
#include "matrix.h"

//Number of each matrix struct parameter
int B[] = {
    1,
    1,
    1
};

//Size of each matrix struct parameter
MPI_Aint D[] = {
    offsetof(struct matrix, rows),
    offsetof(struct matrix, cols),
    offsetof(struct matrix, val)
};

//Type of each matrix struct parameter
MPI_Datatype U[] = {
    MPI_INT,
    MPI_INT,
    MPI_DOUBLE
};

//Name of new MPI datatype based on matrix struct
MPI_Datatype mpi_dt_matrix;

/**
 * Basics of this code and matrix code is from ComS424 lecture notes.
 * Anything documented is instead from me. 
*/
int main(int argc, char *argv[])
{
    //Setup for MPI
    int comm_sz, my_rank;
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    //Create new MPI datatype based on matrix struct.
    MPI_Type_create_struct(3, B, D, U, &mpi_dt_matrix);
    MPI_Type_commit(&mpi_dt_matrix);

    //Getting parameters
    int K, N, M, W;
    void get_input(int argc, char *argv[],
                   const int my_rank,
                   const int comm_sz,
                   int *W,
                   int *M,
                   int *K,
                   int *N);
    get_input(argc, argv, my_rank, comm_sz, &W, &M, &K, &N);

    matrix features[M];
    matrix kernels[N];
    matrix output[N];

    int f, o;

    for(f = 0; f < M; f++)
    {
        features[f] = new_matrix(W+2, W+2);
    }

    for(o = 0; o < N; o++)
    {
        kernels[o] = new_matrix(K, K);
        output[o] = new_matrix(W, W);
    }

    const int s = N / comm_sz;

    // Get time
    double time_start;
    if (my_rank == 0)
    {
        time_start = MPI_Wtime();
    }

    //Getting variables set up per node. 
    const int local_N = s * my_rank;
    const int local_E = local_N + (s-1);
    matrix local_kerns[s];
    matrix local_outs[s];
    int p = 0;
    for(int l = local_N; l <= local_E; l++)
    {
        local_kerns[p] = kernels[l];
        local_outs[p] = output[l];
        p++;
    }

    //Calls the function to actual perform convolution layers.
    void convolute(const int S, int M, matrix* feats, matrix* kerns, matrix* outs);
    convolute(s, M, features, local_kerns, local_outs);

    // Add local results to the global result on Processor 0
    if (my_rank != 0)
    {
        MPI_Send(&local_outs, 1, mpi_dt_matrix, 0, 0,
                 MPI_COMM_WORLD);
    }
    else
    {
        for(int y = 0; y < s; y++)
        {
            output[y] = local_outs[y];
        }
 
        for (int i = 1; i < comm_sz; i++)
        {
            MPI_Recv(&local_outs, 1, mpi_dt_matrix, i, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            int j = 0;
            //Because arrays are used we need to go through each output a node is in charge of.
            for(int x = (s * i); x <= ((s * i) + (s-1)); x++)
            {
                output[x] = local_outs[j];
                j++;
            }
        }
    }
    // Print answer to screen
    if (my_rank == 0)
    {
        double time_end = MPI_Wtime();
        double time_elapsed = time_end - time_start;
        printf(" NP = %2i, N = %i, \n",
               comm_sz, N);
        printf(" Elapsed time = %20.13e\n",
               time_elapsed);
        //print_matrix(&output[3]);
    }

    // End program
    MPI_Finalize();
    return 0;
}

void get_input(int argc, char *argv[],
               const int my_rank,
               const int comm_sz,
               int *W,
               int *M,
               int *K,
               int *N)
{
    void usage(const char *prog_name);
    if (my_rank == 0)
    {
        if (argc != 5)
        {
            usage(argv[0]);
        }

        *W = strtol(argv[1], NULL, 10);
        *M = strtol(argv[2], NULL, 10);
        *K = strtol(argv[3], NULL, 10);
        *N = strtol(argv[4], NULL, 10);

        if(*N <= 0)
        {
            usage(argv[0]);
        }
        if (*N % comm_sz != 0)
        {
            usage(argv[0]);
        }
        for (int i = 1; i < comm_sz; i++)
        {
            MPI_Send(W, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(M, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(K, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(N, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
    }
    else
    {
        MPI_Recv(W, 1, MPI_INT, 0, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        MPI_Recv(M, 1, MPI_INT, 0, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        MPI_Recv(K, 1, MPI_INT, 0, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        MPI_Recv(N, 1, MPI_INT, 0, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
    }
}

void usage(const char *prog_name)
{
    fprintf(stderr, " usage : %s <W/D> <M> <K> <N>\n", prog_name);
    fprintf(stderr, " N should be positive \n");
    fprintf(stderr, " N should be exactly divisible by "
                    "the number of processors \n");
    exit(1);
}

void convolute(const int S, int M, matrix* feats, matrix* kerns, matrix* outs)
{
    /**
     * Applies a convolutional layer to the features and places results in outs.
     * S : size of the split(how many kernels each processor gets)
     * M : number of features
     * feats : array of features
     * kerns : array of kernels specific for this processor
     * outs : the output array to place results in, specific for this processor.
    */
    int x = feats[0].cols;
    int y = feats[0].rows;
    double weight = (1.0 / M);
    double aggregate(int W, int D, matrix feat, matrix kern);

    for(int i = 0; i < S; i++)
    {
        for(int f = 0; f < M; f++)
        {
        for(int w = 1; w < x-1; w++)
        {
            for(int d = 1; d < y-1; d++)
            {
                mget(outs[i], w, d) = (aggregate(w, d, feats[f], kerns[i]) * weight) + mget(outs[i], w, d);
            }
        }
        }
    }
}

double aggregate(int W, int D, matrix feat, matrix kern)
{
    /**
    * Applies an incredibly simple kernel filter onto an element.
    * W : X dimension of the element
    * D : Y dimension of the element
    * feat : Relevant feature matrix
    * kern : Relevant kernel matrix
    * returns : The aggregation result
    */

    double weight = 1.0 / 9.0;
    double total = 0;

    total += weight * (mget(feat, W, D) * mget(kern, 1, 1));
    total += weight * (mget(feat, W+1, D) * mget(kern, 2, 1));
    total += weight * (mget(feat, W+2, D) * mget(kern, 3, 1));
    total += weight * (mget(feat, W, D+1) * mget(kern, 1, 2));
    total += weight * (mget(feat, W+1, D+1) * mget(kern, 2, 2));
    total += weight * (mget(feat, W+2, D+1) * mget(kern, 3, 2));
    total += weight * (mget(feat, W, D+2) * mget(kern, 1, 3));
    total += weight * (mget(feat, W+1, D+2) * mget(kern, 2, 3));
    total += weight * (mget(feat, W+2, D+2) * mget(kern, 3, 3));

    return total;
}
