from Bio import SeqIO
# Es probable que esta clase esté diseñada para leer y procesar archivos FASTA, comúnmente utilizados
# en bioinformática para almacenar secuencias de nucleótidos o proteínas.
class fasta_reader:

    @staticmethod
    def merge_sequences_from_fasta(file_path):
        sequences = []  # List to store all sequences
        for record in SeqIO.parse(file_path, "fasta"):
            # `record.seq` gives the sequence
            sequences.append(str(record.seq))
        return "".join(sequences)
