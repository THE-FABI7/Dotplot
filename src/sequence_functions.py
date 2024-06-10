# sequence_functions.py
from Bio import SeqIO

# Función para fusionar secuencias desde un archivo fasta
def merge_sequences_from_fasta(file_path):
    sequences = []
    for record in SeqIO.parse(file_path, "fasta"):
        sequences.append(str(record.seq))
    return "".join(sequences)
