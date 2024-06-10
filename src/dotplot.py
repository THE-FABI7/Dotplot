import numpy as np
import matplotlib.pyplot as plt

def generate_dotplot(seq1, seq2):
    len_seq1 = len(seq1)
    len_seq2 = len(seq2)
    dotplot_matrix = np.zeros((len_seq1, len_seq2))

    for i in range(len_seq1):
        for j in range(len_seq2):
            if seq1[i] == seq2[j]:
                dotplot_matrix[i][j] = 1

    plt.imshow(dotplot_matrix, cmap='Greys', interpolation='nearest')
    plt.xlabel('Sequence 1')
    plt.ylabel('Sequence 2')
    plt.title('Dotplot')
    plt.show()
