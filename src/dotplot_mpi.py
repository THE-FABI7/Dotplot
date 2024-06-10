from mpi4py import MPI
import numpy as np
import argparse
from sequence_functions import merge_sequences_from_fasta

def dotplot_mpi(sec1, sec2):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    len1, len2 = len(sec1), len(sec2)
    chunk_size = len1 // size
    start = rank * chunk_size
    end = len1 if rank == size - 1 else (rank + 1) * chunk_size
    local_matrix = np.zeros((end - start, len2), dtype=int)  # Cambio a bool

    for i in range(start, end):
        for j in range(len2):
            local_matrix[i - start, j] = sec1[i] == sec2[j]

    # Gather all partial results to the root process
    matrix = None
    if rank == 0:
        matrix = np.zeros((len1, len2), dtype=int)  # Cambio a bool
    comm.Gather(local_matrix, matrix, root=0)
    if rank == 0:
        return matrix

def main():
    parser = argparse.ArgumentParser(description="Generate a dotplot using MPI.")
    parser.add_argument("--file1", required=True, help="Path to the first FASTA file")
    parser.add_argument("--file2", required=True, help="Path to the second FASTA file")
    args = parser.parse_args()

    if MPI.COMM_WORLD.Get_rank() == 0:
        sec1 = merge_sequences_from_fasta(args.file1)
        sec2 = merge_sequences_from_fasta(args.file2)
        print(len(sec1), len(sec1))
    else:
        sec1 = None
        sec2 = None

    # Broadcast sequences to all processes
    sec1 = MPI.COMM_WORLD.bcast(sec1, root=0)
    sec2 = MPI.COMM_WORLD.bcast(sec2, root=0)

    # Compute the dotplot matrix
    matrix = dotplot_mpi(sec1, sec2)

    # If this is the root process, handle the output
    if MPI.COMM_WORLD.Get_rank() == 0 and matrix is not None:
        print(matrix)

if __name__ == "__main__":
    main()
