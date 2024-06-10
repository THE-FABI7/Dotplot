import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from Bio import SeqIO
import time
import threading
import argparse
from fasta_reader import fasta_reader

# Función worker para el dotplot utilizando hilos
def dotplot_thread(seq1, seq2, start, end, matrix):
    for j in range(len(seq2)):
        if seq1[start:end] == seq2[j]:
            matrix[start:end, j] = 1
            
#Función para dotplot secuencial
def dotplot_secuencial(sec1, sec2):
    dotplot = np.empty([len(sec1), len(sec2)])
    for i in range(dotplot.shape[0]):
        for j in range(dotplot.shape[1]):
            dotplot[i, j] = 1 if sec1[i] == sec2[j] else 0
    return dotplot

# Función para generar el dotplot utilizando hilos
def dotplot(seq1, seq2, num_threads=4):
    len1, len2 = len(seq1), len(seq2)
    matrix = np.zeros((len1, len2), dtype=int)
    threads = []
    chunk_size = len1 // num_threads
    for i in range(num_threads):
        start = i * chunk_size
        end = start + chunk_size if i < num_threads - 1 else len1
        thread = threading.Thread(target=dotplot_thread, args=(seq1, seq2, start, end, matrix))
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()
    return matrix

# Función para visualizar el dotplot
def draw_dotplot(matrix, fig_name):
    plt.figure(figsize=(10, 10))
    plt.imshow(matrix, cmap='Greys', aspect='auto')
    plt.ylabel("Secuencia 1")
    plt.xlabel("Secuencia 2")
    plt.title("Dotplot")
    plt.savefig(fig_name)
    plt.show()

# Función worker para multiprocessing
def worker(args):
    i, sec1, sec2 = args
    return [1 if sec1[i] == sec2[j] else 0 for j in range(len(sec2))]

# Función para generar el dotplot utilizando multiprocessing
def dotplot_multiprocessing(sec1, sec2, threads=mp.cpu_count()):
    with mp.Pool(processes=threads) as pool:
        results = pool.map(worker, [(i, sec1, sec2) for i in range(len(sec1))])
    return np.array(results)

def main():
    parser = argparse.ArgumentParser(description="Generate a dotplot from two FASTA files.")
    parser.add_argument("--file1", required=True, help="Path to the first FASTA file")
    parser.add_argument("--file2", required=True, help="Path to the second FASTA file")
    parser.add_argument("--output", default="dotplot.svg", help="Output file name for the dotplot image")
    parser.add_argument("--threads", type=int, default=mp.cpu_count(), help="Number of threads to use")
    args = parser.parse_args()

    seq1 = fasta_reader.merge_sequences_from_fasta(args.file1)
    seq2 = fasta_reader.merge_sequences_from_fasta(args.file2)

    # Choose the function based on the number of threads
    if args.threads > 1:
        matrix = dotplot_multiprocessing(seq1, seq2, args.threads)
    else:
        matrix = dotplot(seq1, seq2)

    draw_dotplot(matrix, args.output)

if __name__ == "__main__":
    main()
