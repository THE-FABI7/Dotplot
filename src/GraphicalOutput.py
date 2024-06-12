import matplotlib.pyplot as plt

class GraphicalOutput:
    
    def draw_dotplot(self, dotplot, fig_name='dotplot.svg'):
        plt.figure(figsize=(10, 10))
        plt.imshow(dotplot, cmap="Greys", aspect="auto")
        plt.xlabel("Secuencia 1")
        plt.ylabel("Secuencia 2")
        plt.savefig(fig_name)
        # plt.show()

    def draw_graphic_multiprocessing(self, times, accelerations, efficiencies, num_threads):
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 2, 1)
        plt.plot(num_threads, times)
        plt.xlabel("Número de procesadores")
        plt.ylabel("Tiempo")
        plt.subplot(1, 2, 2)
        plt.plot(num_threads, accelerations)
        plt.plot(num_threads, efficiencies)
        plt.xlabel("Número de procesadores")
        plt.ylabel("Aceleración y Eficiencia")
        plt.legend(["Aceleración", "Eficiencia"])
        plt.savefig("graficas/graficasMultiprocessing.png")

    def draw_graphic_mpi(self, times, accelerations, efficiencies, num_threads):
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 2, 1)
        plt.plot(num_threads, times)
        plt.xlabel("Número de procesadores")
        plt.ylabel("Tiempo")
        plt.subplot(1, 2, 2)
        plt.plot(num_threads, accelerations)
        plt.plot(num_threads, efficiencies)
        plt.xlabel("Número de procesadores")
        plt.ylabel("Aceleración y Eficiencia")
        plt.legend(["Aceleración", "Eficiencia"])
        plt.savefig("graficas/graficasMPI.png")
