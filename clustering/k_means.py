from sklearn.cluster import KMeans
import numpy as np


class K_Means(object):
    def __init__(self):
        # Assuming x and y values are coordinates
        self.points = np.array([
            [359, 761],  # A
            [937, 1093],  # B
            [529, 192],  # C
            [283, 422],  # D
            [532, 858],  # E
            [27, 444],  # F
            [14, 19],  # G
            [1277, 350],  # H
            [1143, 925],  # I
            [711, 659]  # J
        ])

        # Initial centroids as provided
        self.initial_centroids1 = np.array([[400, 10], [300, 700], [800, 300]])
        self.initial_centroids2 = np.array([[937, 1093], [283, 422], [532, 858]])  # Coordinates of B, D, E

    # Function to run k-means and return centroids and labels
    def perform_kmeans(self, coordinates, initial_centroids):
        # Create a KMeans instance with k=3 and provided initial centroids
        kmeans = KMeans(n_clusters=3, init=initial_centroids, n_init=1, max_iter=300, tol=1e-04, random_state=0)
        # Compute k-means clustering
        kmeans.fit(coordinates)
        # Return the centroids and the labels of the clusters
        return kmeans.cluster_centers_, kmeans.labels_

    def perform_kmeans_with_initial_centroids(self):
        # Perform k-means with the first set of initial centroids
        centroids1, labels1 = self.perform_kmeans(self.points, self.initial_centroids1)
        return (centroids1, labels1)

    def perform_kmeans_with_initial_centroid_points(self):
        # Perform k-means with the second set of initial centroids (B, D, E)
        centroids2, labels2 = self.perform_kmeans(self.points, self.initial_centroids2)

        return (centroids2, labels2)


if __name__ == '__main__':
    k_means = K_Means()
    print(k_means.perform_kmeans_with_initial_centroids())
    print(k_means.perform_kmeans_with_initial_centroid_points())