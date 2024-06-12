from Bio import SeqIO

class SequenceProcessor:
    
    def read_fasta(self, file_name):
        sequences = []
        for record in SeqIO.parse(file_name, "fasta"):
            sequences.append(str(record.seq))
        return "".join(sequences)

    def save_results_to_file(self, results, file_name="utils/results.txt"):
        with open(file_name, "w") as file:
            for result in results:
                file.write(str(result) + "\n")
