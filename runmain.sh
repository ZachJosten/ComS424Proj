#!/bin/bash

# Copy/paste this job script into a text file and submit with the command:
#    sbatch thefilename
# job standard output will go to the file slurm-%j.out (where %j is the job ID)

#SBATCH --time=00:15:00   # walltime limit (HH:MM:SS)
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=36   # 36 processor core(s) per node 
#SBATCH --partition=class-short    # class node(s)

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
mpirun -np 1 ./main.exe 32 128 3 64 > onesmall.txt
mpirun -np 2 ./main.exe 32 128 3 64 > twosmall.txt
mpirun -np 4 ./main.exe 32 128 3 64 > foursmall.txt
mpirun -np 16 ./main.exe 32 128 3 64 > sixteensmall.txt
mpirun -np 32 ./main.exe 32 128 3 64 > thirtytwosmall.txt
mpirun -np 1 ./main.exe 64 512 3 256 > onebig.txt
mpirun -np 32 ./main.exe 64 512 3 256 > thirtytwobig.txt
