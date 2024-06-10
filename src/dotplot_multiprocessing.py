import numpy as np
import multiprocessing as mp
import argparse
from sequence_functions import merge_sequences_from_fasta

# Función worker para el dotplot utilizando multiprocessing
def worker(args):
    i, sec1, sec2 = args
    return [1 if sec1[i] == sec2[j] else 0 for j in range(len(sec2))]

# Función para generar el dotplot utilizando multiprocessing
def dotplot_multiprocessing(sec1, sec2, threads=mp.cpu_count()):
    with mp.Pool(processes=threads) as pool:
        results = pool.map(worker, [(i, sec1, sec2) for i in range(len(sec1))])
    return np.array(results)

def main():
    parser = argparse.ArgumentParser(description="Generate a dotplot using multiprocessing.")
    parser.add_argument("--file1", required=True, help="Path to the first FASTA file")
    parser.add_argument("--file2", required=True, help="Path to the second FASTA file")
    parser.add_argument("--threads", type=int, default=mp.cpu_count(), help="Number of threads to use")
    args = parser.parse_args()

    sec1 = merge_sequences_from_fasta(args.file1)
    sec2 = merge_sequences_from_fasta(args.file2)

    matrix = dotplot_multiprocessing(sec1, sec2, args.threads)
    print("Dotplot matrix:")
    print(matrix)

if __name__ == "__main__":
    main()
