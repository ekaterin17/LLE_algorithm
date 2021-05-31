import numpy as np

from scipy.spatial import KDTree


class LLE:
    def __init__(self, n_neighbors: int = 5, n_components: int = 2):
        self.n_neighbors = n_neighbors
        self.n_components = n_components

    def fit_transform(self, X: np.ndarray = [], remove_zero: bool = True):
        n_samples = X.shape[0]
        W = np.zeros([n_samples, n_samples])
        adjacency_index_matrix = self._construct_graph(X)
        for idx in range(n_samples):
            Z = X[adjacency_index_matrix[idx, :], :]
            Zi = Z - X[idx, :]
            covariance_matrix = Zi @ Zi.T
            wi = np.linalg.pinv(covariance_matrix) @ np.ones([self.n_neighbors, 1]) / \
                 np.sum(np.linalg.pinv(covariance_matrix))
            for jdx in range(self.n_neighbors):
                W[adjacency_index_matrix[idx, jdx], idx] = wi[jdx]

        M = (np.eye(n_samples) - W) @ (np.eye(n_samples) - W).T
        eigen_vector, eigen_value, vT = np.linalg.svd(M)
        X_embedded = eigen_vector[:, -self.n_components - 1:-1]

        return X_embedded

    def _construct_graph(self, X: np.ndarray):
        n_samples = X.shape[0]
        kd_tree = KDTree(X.copy())
        adjacency_index_matrix = np.ones([n_samples, self.n_neighbors], dtype=int)
        for idx in range(n_samples):
            _, neighbors = kd_tree.query(X[idx, :], k=self.n_neighbors + 1, p=2)
            neighbors = neighbors.tolist()
            neighbors.remove(idx)
            adjacency_index_matrix[idx, :] = np.array([neighbors])

        return adjacency_index_matrix
