class PerformanceAnalysis:
    
    def acceleration(self, times):
        return [times[0] / i for i in times]

    def efficiency(self, accelerations, num_threads):
        return [accelerations[i] / num_threads[i] for i in range(len(num_threads))]
