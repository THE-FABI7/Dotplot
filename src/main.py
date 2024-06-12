import argparse
import time
import numpy as np
from DotPlot import DotPlot
from GraphicalOutput import GraphicalOutput
from ImageFilter import ImageFilter
from performance_analysis import PerformanceAnalysis
from sequence_processor import SequenceProcessor

    
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--file1', type=str, help='Query sequence in FASTA format')
    parser.add_argument('--file2', type=str, help='Subject sequence in FASTA format')
    parser.add_argument('--sequential', action='store_true', help='Run in sequential mode')
    parser.add_argument('--multiprocessing', action='store_true', help='Run using multiprocessing')
    parser.add_argument('--mpi', action='store_true', help='Run using MPI')
    parser.add_argument('--num_processes', dest='num_processes', type=int, nargs='+',
                        default=[4], help='Número de procesos para la opción MPI')
    args = parser.parse_args()
    
    sequence_processor = SequenceProcessor()
    dot_plot = DotPlot()
    graphical_output = GraphicalOutput()
    image_filter = ImageFilter()
    performance_analysis = PerformanceAnalysis()
    
    
    if args.file1 and args.file2:
        # Measure time to load files
        start_load_time = time.time()
        sequence1 = sequence_processor.read_fasta(args.file1)
        sequence2 = sequence_processor.read_fasta(args.file2)
        end_load_time = time.time()
        load_time = end_load_time - start_load_time
        sequence_processor.save_results_to_file([f"File loading time: {load_time}"], "utils/load_time.txt")

        # Truncate sequences for demonstration
        sequence1 = sequence1[:1000]
        sequence2 = sequence2[:1000]
        
    
    if args.sequential:
        # Perform sequential dotplot calculation
        start_time = time.time()
        sequential_dotplot = dot_plot.dotplot_sequential(sequence1, sequence2)
        elapsed_time = time.time() - start_time
        graphical_output.draw_dotplot(sequential_dotplot[:600, :600], "imagenes/secuencial/dotplot_sequential.png")
        image_filter.apply_filter(sequential_dotplot[:600, :600], "imagenes/secuencial/filtered_dotplot_sequential.png")
        sequence_processor.save_results_to_file([f"Sequential run time: {elapsed_time}"], "utils/secuencial_times.txt")
    
    if args.multiprocessing:
        # Perform multiprocessing dotplot calculation
        num_threads = [1, 2, 4, 8]  # Example thread counts
        times = []
        for threads in num_threads:
            start_time = time.time()
            multiprocessing_dotplot = dot_plot.parallel_multiprocessing_dotplot(sequence1, sequence2, threads)
            elapsed_time = time.time() - start_time
            times.append(elapsed_time)
        accelerations = performance_analysis.acceleration(times)
        efficiencies = performance_analysis.efficiency(accelerations, num_threads)
        graphical_output.draw_graphic_multiprocessing(times, accelerations, efficiencies, num_threads)
        graphical_output.draw_dotplot(multiprocessing_dotplot[:600, :600], "imagenes/multiprocessing/dotplot_multiprocessing.png")
        image_filter.apply_filter(multiprocessing_dotplot[:600, :600], "imagenes/multiprocessing/filtered_dotplot_multiprocessing.png")
        print(times)
        sequence_processor.save_results_to_file(f"Multiprocessing times: {times}", "utils/multiprovessing_results.txt")

    if args.mpi:
        # Perform MPI dotplot calculation
        mpi_dotplot = dot_plot.parallel_mpi_dotplot(sequence1, sequence2)
        graphical_output.draw_dotplot(mpi_dotplot[:600, :600], "imagenes/mpi/dotplot_mpi.png")
        # graphical_output.draw_graphic_mpi(times, accelerations, efficiencies, num_threads)
        print("Tipo de datos antes de aplicar el filtro:", mpi_dotplot.dtype)
        mpi_dotplot = mpi_dotplot.astype(np.float32) 
        image_filter.apply_filter(mpi_dotplot[:600, :600], "imagenes/mpi/filtered_dotplot_mpi.png")
        sequence_processor.save_results_to_file(["MPI results placeholder"], "utils/mpi_results.txt")

    
if __name__ == "__main__":
    main()
