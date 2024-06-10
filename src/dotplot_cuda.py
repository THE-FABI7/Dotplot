# dotplot_cuda.py
import numpy as np
from numba import cuda

# Función CUDA para el dotplot
@cuda.jit
def dotplot_cuda_kernel(sec1, sec2, matrix):
    i, j = cuda.grid(2)
    if i < len(sec1) and j < len(sec2):
        matrix[i, j] = 1 if sec1[i] == sec2[j] else 0

def dotplot_cuda(sec1, sec2):
    len1, len2 = len(sec1), len(sec2)
    matrix = np.zeros((len1, len2), dtype=np.int32)
    d_matrix = cuda.to_device(matrix)
    threadsperblock = (16, 16)
    blockspergrid_x = int(np.ceil(len1 / threadsperblock[0]))
    blockspergrid_y = int(np.ceil(len2 / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    dotplot_cuda_kernel[blockspergrid, threadsperblock](sec1, sec2, d_matrix)
    return d_matrix.copy_to_host()
