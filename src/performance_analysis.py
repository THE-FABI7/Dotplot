import time
import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI
import multiprocessing as mp
from dotplot_hilos import dotplot_threads
from dotplot_cuda import dotplot_cuda
from dotplot_multiprocessing import dotplot_multiprocessing
from sequence_functions import merge_sequences_from_fasta, draw_dotplot
from mtprocessing import dotplot_multiprocessing, dotplot_secuencial
from dotplot_mpi import dotplot_mpi


def main():
    # Cargar las secuencias desde los archivos fasta
    Secuencia1 = "E_coli.fna"
    Secuencia2 = "Salmonella.fna"

    # Fusionar las secuencias desde los archivos fasta
    seq1 = merge_sequences_from_fasta(Secuencia1)
    seq2 = merge_sequences_from_fasta(Secuencia2)

    # Reducir el tamaño de las secuencias para una mejor visualización
    seq1 = seq1[:1000]  # Tomar los primeros 1000 nucleótidos
    seq2 = seq2[:1000]  # Tomar los primeros 1000 nucleótidos

    # Realizar el análisis de rendimiento
    begin_secuencial = time.time()
    dotplot_secuencial(seq1, seq2)
    end_secuencial = time.time()
    print(f"El tiempo secuencial es: {end_secuencial - begin_secuencial} segundos")

    begin_paralelo = time.time()
    dotplot_multiprocessing(seq1, seq2, 2)  # Cambia el número de procesadores según necesites
    end_paralelo = time.time()
    print(f"El tiempo paralelo (multiprocesamiento) es: {end_paralelo - begin_paralelo} segundos")

    begin_hilos = time.time()
    dotplot_threads(seq1, seq2, 4)
    end_hilos = time.time()
    print(f"El tiempo paralelo (hilos) es: {end_hilos - begin_hilos} segundos")

    begin_cuda = time.time()
    dotplot_cuda(np.array(list(seq1)), np.array(list(seq2)))
    end_cuda = time.time()
    print(f"El tiempo paralelo (CUDA) es: {end_cuda - begin_cuda} segundos")

    begin_mpi = MPI.Wtime()
    dotplot_mpi(seq1, seq2)
    end_mpi = MPI.Wtime()
    if MPI.COMM_WORLD.Get_rank() == 0:
        print(f"El tiempo paralelo (MPI) es: {end_mpi - begin_mpi} segundos")

    n_proc = [1, 2, 4, 8]
    times = []
    for i in n_proc:
        begin_paralelo = time.time()
        dotplot_multiprocessing(seq1, seq2, i)
        end_paralelo = time.time()
        times.append(end_paralelo - begin_paralelo)
        print(f"Dotplot con {i} procesadores (multiprocesamiento), tiempo: {end_paralelo - begin_paralelo} segundos")

    # Visualizar el tiempo de ejecución
    plt.figure(figsize=(5, 5))
    plt.plot(n_proc, times)
    plt.xlabel("Número de procesadores")
    plt.ylabel("Tiempo de ejecución")
    plt.show()

    # Calcular aceleración y eficiencia
    acel = [times[0] / i for i in times]
    efic = [acel[i] / n_proc[i] for i in range(len(n_proc))]
    print("Aceleración:", acel)
    print("Eficiencia:", efic)

    # Visualizar aceleración y eficiencia
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(n_proc, times)
    plt.xlabel("Número de procesadores")
    plt.ylabel("Tiempo de ejecución")
    plt.subplot(1, 2, 2)
    plt.plot(n_proc, acel)
    plt.plot(n_proc, efic)
    plt.xlabel("Número de procesadores")
    plt.ylabel("Aceleración y eficiencia")
    plt.legend(["Aceleración", "Eficiencia"])
    plt.show()

    # Generar el dotplot final y visualizarlo
    matrix = dotplot_multiprocessing(seq1, seq2, threads=4)
    draw_dotplot(matrix, fig_name='dotplot_final.svg')

if __name__ == '__main__':
    main()
