import argparse
import time
from fasta_reader import fasta_reader
from dotplot import generate_dotplot
from mtprocessing import dotplot_multiprocessing, dotplot_secuencial
    
def main():
    parser = argparse.ArgumentParser(
        description="Generate dotplot for two sequences.")
    parser.add_argument("--file1", help="First FASTA file")
    parser.add_argument("--file2", help="Second FASTA file")
    parser.add_argument("--thres", help="Threshold for dotplot", type=float)
    parser.add_argument("--output", help="Output file for dotplot")
    args = parser.parse_args()

    sequences1 = fasta_reader.merge_sequences_from_fasta(args.file1)
    sequences2 = fasta_reader.merge_sequences_from_fasta(args.file2)
    threshold = args.thres
    output_file = args.output

    generate_dotplot(sequences1, sequences2)

    
if __name__ == "__main__":
    main()
