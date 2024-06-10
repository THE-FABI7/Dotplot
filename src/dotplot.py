from asyncio import threads
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from mtprocessing import worker

def generate_dotplot(seq1, seq2, threads=mp.cpu_count()):
    len_seq1 = len(seq1)
    len_seq2 = len(seq2)
    
    print(len_seq1)
    print(len_seq2)
    
    with mp.Pool(processes=threads) as pool:
        results = pool.map(worker, [(i, seq1, seq2) for i in range(len(seq1))])
    return np.array(results)