import argparse
from fasta_reader import fasta_reader
from dotplot import generate_dotplot


class main:

    def main():
        # parser = argparse.ArgumentParser(
        # description="Generate dotplot for two sequences.")
        # parser.add_argument("file1", help="First FASTA file")
        # parser.add_argument("file2", help="Second FASTA file")
        # args = parser.parse_args()

       # Verifique a implementação desta função
        sequences1 = fasta_reader.merge_sequences_from_fasta(
            '../Data/E_coli.fna')
        sequences2 = fasta_reader.merge_sequences_from_fasta(
            '../Data/Salmonella.fna')

        print("longitud Archivo 1:", len(sequences1))
        print("longitud Archivo 2:", len(sequences2))
        

    if __name__ == "__main__":
        main()
