import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from mpi4py import MPI
from pycuda.compiler import SourceModule


# Código CUDA para calcular el dotplot en la GPU
mod = SourceModule("""
_global_ void dotplot_kernel(unsigned char *seq1, unsigned char *seq2, unsigned char *dot_matrix, int len1, int len2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < len1 && j < len2) {
        dot_matrix[i * len2 + j] = (seq1[i] == seq2[j]) ? 1 : 0;
    }
}
""")

class DotPlot:
    
    def dotplot_sequential(self, sequence1, sequence2):
        dotplot = np.empty((len(sequence1), len(sequence2)))
        for i in tqdm(range(len(sequence1))):
            for j in range(len(sequence2)):
                if sequence1[i] == sequence2[j]:
                    if i == j:
                        dotplot[i, j] = 1
                    else:
                        dotplot[i, j] = 0.7
                else:
                    dotplot[i, j] = 0
        return dotplot
    
    #Función para convertir secuencia de ADN a una representación numérica
    def convertir_a_numerico(self,secuencia):
        mapa = {'A': 1, 'T': 2, 'C': 3, 'G': 4}
        return np.array([mapa[base] for base in secuencia], dtype=np.uint8)
    
    def dotplot_CUDA(self,secuencia1, secuencia2, block_size=500):
        len1, len2 = len(secuencia1), len(secuencia2)
        matriz_puntos = np.zeros((len1, len2), dtype=np.uint8)
        
        seq1_array = self.convertir_a_numerico(secuencia1)
        seq2_array = self.convertir_a_numerico(secuencia2)
        
        for i in range(0, len1, block_size):
            end_i = min(i + block_size, len1)
            for j in range(0, len2, block_size):
                end_j = min(j + block_size, len2)
                
                sub_seq1 = seq1_array[i:end_i]
                sub_seq2 = seq2_array[j:end_j]
                
                sub_matriz_puntos = np.zeros((sub_seq1.size, sub_seq2.size), dtype=np.uint8)
                
                # Copiar datos a la GPU
                seq1_gpu = cuda.mem_alloc(sub_seq1.nbytes)
                seq2_gpu = cuda.mem_alloc(sub_seq2.nbytes)
                dot_matrix_gpu = cuda.mem_alloc(sub_matriz_puntos.nbytes)
                
                cuda.memcpy_htod(seq1_gpu, sub_seq1)
                cuda.memcpy_htod(seq2_gpu, sub_seq2)
                cuda.memcpy_htod(dot_matrix_gpu, sub_matriz_puntos)
                
                # Configurar el tamaño del bloque y la cuadrícula
                threadsperblock = (16, 16, 1)
                blockspergrid_x = int(np.ceil(sub_seq1.size / threadsperblock[0]))
                blockspergrid_y = int(np.ceil(sub_seq2.size / threadsperblock[1]))
                blockspergrid = (blockspergrid_x, blockspergrid_y, 1)
                
                # Ejecutar el kernel en la GPU
                func = mod.get_function("dotplot_kernel")
                func(seq1_gpu, seq2_gpu, dot_matrix_gpu, np.int32(sub_seq1.size), np.int32(sub_seq2.size), block=threadsperblock, grid=blockspergrid)
                
                # Copiar el resultado de vuelta a la CPU
                cuda.memcpy_dtoh(sub_matriz_puntos, dot_matrix_gpu)
                
                matriz_puntos[i:end_i, j:end_j] = sub_matriz_puntos
    
        return matriz_puntos

    def worker_multiprocessing(self, args):
        i, sequence1, sequence2 = args
        dotplot = []
        for j in range(len(sequence2)):
            if sequence1[i] == sequence2[j]:
                if i == j:
                    dotplot.append(1)
                else:
                    dotplot.append(0.7)
            else:
                dotplot.append(0)
        return dotplot

    def parallel_multiprocessing_dotplot(self, sequence1, sequence2, threads=mp.cpu_count()):
        with mp.Pool(processes=threads) as pool:
            dotplot = pool.map(self.worker_multiprocessing, [(i, sequence1, sequence2) for i in range(len(sequence1))])
        return np.array(dotplot)

    def parallel_mpi_dotplot(self, sequence1, sequence2):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        chunks = np.array_split(range(len(sequence1)), size)
        dotplot = np.empty([len(chunks[rank]), len(sequence2)], dtype=np.float16)

        for i in tqdm(range(len(chunks[rank]))):
            for j in range(len(sequence2)):
                if sequence1[chunks[rank][i]] == sequence2[j]:
                    if (i == j):    
                        dotplot[i, j] = np.float16(1.0)
                    else:
                        dotplot[i, j] = np.float16(0.6)
                else:
                    dotplot[i, j] = np.float16(0.0)

        dotplot = comm.gather(dotplot, root=0)
        if rank == 0:
            return np.vstack(dotplot)
