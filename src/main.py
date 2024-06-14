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
    parser.add_argument('--pycuda', action='store_true', help='Run using CUDA')
    parser.add_argument('--num_processes', dest='num_processes', type=int, nargs='+',
                        default=[4], help='Número de procesos para la opción MPI')
    args = parser.parse_args()

    sequence_processor = SequenceProcessor()
    dot_plot = DotPlot()
    graphical_output = GraphicalOutput()
    image_filter = ImageFilter()
    performance_analysis = PerformanceAnalysis()

    try:
        if args.file1 and args.file2:
            start_load_time = time.time()
            try:
                sequence1 = sequence_processor.read_fasta(args.file1)
                sequence2 = sequence_processor.read_fasta(args.file2)
            except Exception as e:
                print(f"Error loading files: {e}")
                return
            end_load_time = time.time()
            load_time = end_load_time - start_load_time
            sequence_processor.save_results_to_file([f"File loading time: {load_time}"], "utils/load_time.txt")

            # Truncate sequences for demonstration
            sequence1 = sequence1[:5000]
            sequence2 = sequence2[:5000]
            
            if args.sequential:
                try:
                    start_time = time.time()
                    sequential_dotplot = dot_plot.dotplot_sequential(sequence1, sequence2)
                    elapsed_time = time.time() - start_time
                    graphical_output.draw_dotplot(sequential_dotplot[:5000, :5000], "imagenes/secuencial/dotplot_sequential.png")
                    image_filter.apply_filter(sequential_dotplot[:5000, :5000], "imagenes/secuencial/filtered_dotplot_sequential.png")
                    sequence_processor.save_results_to_file([f"Sequential run time: {elapsed_time}"], "utils/secuencial_times.txt")
                except Exception as e:
                    print(f"Error in sequential processing: {e}")

            if args.multiprocessing:
                try:
                    num_threads = [1, 2, 4, 8]
                    times = []
                    for threads in num_threads:
                        start_time = time.time()
                        multiprocessing_dotplot = dot_plot.parallel_multiprocessing_dotplot(sequence1, sequence2, threads)
                        elapsed_time = time.time() - start_time
                        times.append(elapsed_time)
                    accelerations = performance_analysis.acceleration(times)
                    efficiencies = performance_analysis.efficiency(accelerations, num_threads)
                    graphical_output.draw_graphic_multiprocessing(times, accelerations, efficiencies, num_threads)
                    graphical_output.draw_dotplot(multiprocessing_dotplot[:7000, :7000], "imagenes/multiprocessing/dotplot_multiprocessing.png")
                    image_filter.apply_filter(multiprocessing_dotplot[:7000, :7000], "imagenes/multiprocessing/filtered_dotplot_multiprocessing.png")
                    sequence_processor.save_results_to_file([f"Multiprocessing times: {times}"], "utils/multiprocessing_results.txt")
                except Exception as e:
                    print(f"Error in multiprocessing: {e}")

            if args.mpi:
                try:
                    num_threads = [1, 2, 4, 8]
                    times = []
                    for threads in num_threads:
                        start_time = time.time()
                        mpi_dotplot = dot_plot.parallel_mpi_dotplot(sequence1, sequence2)
                        elapsed_time = time.time() - start_time
                        times.append(elapsed_time)
                    accelerations = performance_analysis.acceleration(times)
                    efficiencies = performance_analysis.efficiency(accelerations, num_threads)
                    graphical_output.draw_graphic_multiprocessing(times, accelerations, efficiencies, num_threads)
                    graphical_output.draw_dotplot(mpi_dotplot[:25000, :25000], "imagenes/mpi/dotplot_mpi.png")
                    image_filter.apply_filter(mpi_dotplot[:25000, :25000], "imagenes/mpi/filtered_dotplot_mpi.png")
                    sequence_processor.save_results_to_file(["MPI results placeholder"], "utils/mpi_results.txt")
                except Exception as e:
                    print(f"Error in MPI processing: {e}")

            if args.cuda:
                try:
                    CUDA_dotplot = dot_plot.dotplot_CUDA(sequence1, sequence2)
                    graphical_output.draw_dotplot(CUDA_dotplot[:5000, :5000], "imagenes/CUDA/dotplot_CUDA.png")
                    image_filter.apply_filter(CUDA_dotplot[:5000, :5000], "imagenes/CUDA/filtered_dotplot_CUDA.png")
                    sequence_processor.save_results_to_file(["CUDA results placeholder"], "utils/CUDA_results.txt")
                except Exception as e:
                    print(f"Error in CUDA processing: {e}")
    except Exception as e:
        print(f"Unhandled exception: {e}")

if __name__ == "__main__":
    main()
