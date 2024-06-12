import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from mpi4py import MPI

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
