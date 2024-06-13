import cv2
import numpy as np

class ImageFilter:
    
    def apply_filter(self, matrix, path_image):
        
        # Define un kernel de convolución para detectar características diagonales
        kernel_diagonales = np.array([[1, -1, -1], [-1, 1, -1], [-1, -1, 1]])
        # Aplica el filtro definido por el kernel a la matriz de imagen original
        filtered_matrix = cv2.filter2D(matrix, -1, kernel_diagonales)
        # Normaliza la matriz filtrada para escalar los valores de los píxeles a un rango de 0 a 127
        normalized_matrix = cv2.normalize(filtered_matrix, None, 0, 127, cv2.NORM_MINMAX)
        # Establece un valor de umbral para convertir la imagen a una versión binaria
        threshold_value = 50
        # Aplica el umbral: píxeles por encima de 50 se convierten a blanco (255), los demás a negro (0)
        _, thresholded_matrix = cv2.threshold(normalized_matrix, threshold_value, 255, cv2.THRESH_BINARY)
        # Guarda la imagen umbralizada en el sistema de archivos en la ruta especificada
        cv2.imwrite(path_image, thresholded_matrix)
        cv2.imshow('Diagonales', thresholded_matrix)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
