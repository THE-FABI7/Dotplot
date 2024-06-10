import threading
import numpy as np
import argparse
from sequence_functions import merge_sequences_from_fasta

# Función worker para el dotplot utilizando hilos
def dotplot_thread(sec1, sec2, start, end, matrix):
    len2 = len(sec2)
    for i in range(start, end):
        for j in range(len2):
            matrix[i, j] = 1 if sec1[i] == sec2[j] else 0

# Función para generar el dotplot utilizando hilos
def dotplot_threads(sec1, sec2, num_threads=4):
    len1, len2 = len(sec1), len(sec2)
    matrix = np.zeros((len1, len2), dtype=int)
    threads = []
    chunk_size = len1 // num_threads
    for i in range(num_threads):
        start = i * chunk_size
        end = len1 if i == num_threads - 1 else (i + 1) * chunk_size
        thread = threading.Thread(target=dotplot_thread, args=(sec1, sec2, start, end, matrix))
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()
    return matrix

def main():
    parser = argparse.ArgumentParser(description="Generate a dotplot using threading for two FASTA files.")
    parser.add_argument("--file1", required=True, help="Path to the first FASTA file")
    parser.add_argument("--file2", required=True, help="Path to the second FASTA file")
    parser.add_argument("--threads", type=int, default=4, help="Number of threads to use")
    args = parser.parse_args()

    # Load sequences from FASTA files
    sec1 = merge_sequences_from_fasta(args.file1)
    sec2 = merge_sequences_from_fasta(args.file2)

    # Generate dotplot using threading
    matrix = dotplot_threads(sec1, sec2, args.threads)
    print(matrix)

if __name__ == "__main__":
    main()
