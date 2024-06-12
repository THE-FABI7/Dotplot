import cv2
import numpy as np

class ImageFilter:
    
    def apply_filter(self, matrix, path_image):
        
        kernel_diagonales = np.array([[1, -1, -1], [-1, 1, -1], [-1, -1, 1]])
        filtered_matrix = cv2.filter2D(matrix, -1, kernel_diagonales)
        normalized_matrix = cv2.normalize(filtered_matrix, None, 0, 127, cv2.NORM_MINMAX)
        threshold_value = 50
        _, thresholded_matrix = cv2.threshold(normalized_matrix, threshold_value, 255, cv2.THRESH_BINARY)
        cv2.imwrite(path_image, thresholded_matrix)
        cv2.imshow('Diagonales', thresholded_matrix)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
