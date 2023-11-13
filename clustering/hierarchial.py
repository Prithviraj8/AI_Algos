import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt


class Hierarchial(object):
    # Provided distance data
    def __init__(self):
        self.distances = {
            'A': {'x': 359, 'y': 761},
            'B': {'x': 937, 'y': 1093},
            'C': {'x': 529, 'y': 192},
            'D': {'x': 283, 'y': 422},
            'E': {'x': 532, 'y': 858},
            'F': {'x': 27, 'y': 444},
            'G': {'x': 14, 'y': 19},
            'H': {'x': 1277, 'y': 350},
            'I': {'x': 1143, 'y': 925},
            'J': {'x': 711, 'y': 659}
        }
        self.labels = None

    # Assuming the provided x and y values are distances from some reference point (not coordinates),
    # we'll create a distance matrix. Normally, a distance matrix is symmetric, but based on the
    # provided values, we'll construct a matrix for the clustering process.

    def initialize_empty_matrix(self):
        # Initialize an empty matrix
        self.labels = list(self.distances.keys())
        matrix_size = len(self.labels)
        distance_matrix = np.zeros((matrix_size, matrix_size))

        # Populate the matrix with the provided distances
        for i, label_i in enumerate(self.labels):
            for j, label_j in enumerate(self.labels):
                if i != j:  # No need to calculate distance from a point to itself
                    # Use Euclidean distance formula (assuming 'x' and 'y' are coordinate-like distances)
                    distance_matrix[i, j] = np.sqrt((self.distances[label_i]['x'] - self.distances[label_j]['x']) ** 2 +
                                                    (self.distances[label_i]['y'] - self.distances[label_j]['y']) ** 2)
        print('distance_matrix:')
        print(distance_matrix)
        return distance_matrix

    def perform_single_linkage(self):
        # Perform single-linkage hierarchical clustering
        distance_matrix = self.initialize_empty_matrix()
        single_linkage = linkage(distance_matrix, method='single')
        self.plot_dendograms('single', single_linkage)

    def perform_complete_linkage(self):
        # Perform complete-linkage hierarchical clustering
        distance_matrix = self.initialize_empty_matrix()
        complete_linkage = linkage(distance_matrix, method='complete')
        self.plot_dendograms('complete', complete_linkage)

    def plot_dendograms(self, linkage_type, linkage):
        # Plot the dendrograms
        fig, ax = plt.subplots(1, 2, figsize=(16, 8))

        if linkage_type == 'single':
            # Single-linkage dendrogram
            dendrogram(linkage, ax=ax[0], labels=self.labels, above_threshold_color='blue')
            ax[0].set_title('Single-linkage Clustering')
        else:
            # Complete-linkage dendrogram
            dendrogram(linkage, ax=ax[1], labels=self.labels, above_threshold_color='red')
            ax[1].set_title('Complete-linkage Clustering')

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    h = Hierarchial()
    # h.perform_single_linkage()
    h.perform_complete_linkage()