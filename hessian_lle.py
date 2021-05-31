import numpy as np

from scipy.spatial import KDTree


class HessianLLE:
    def __init__(self, n_neighbors: int = 5, n_components: int = 2):
        self.n_neighbors = n_neighbors
        self.n_components = n_components

    def fit_transform(self, X: np.ndarray = []):
        n_samples = X.shape[0]
        dp = self.n_components * (self.n_components + 1) // 2

        if self.n_neighbors <= self.n_components + dp:
            raise ValueError("for hessian lle n_neighbors must be "
                             "greater than "
                             "[n_components * (n_components + 3) / 2]")

        adjacency_index_matrix = self._construct_graph(X)
        W = np.zeros((dp * n_samples, n_samples))
        Yi = np.empty((self.n_neighbors, self.n_components + dp + 1))
        Yi[:, 0] = 1
        for i in range(n_samples):
            neighbors_i = adjacency_index_matrix[i, :]
            Gi = X[neighbors_i]
            Gi -= Gi.mean(0)

            U = np.linalg.svd(Gi, full_matrices=0)[0]

            Yi[:, 1:1 + self.n_components] = U[:, :self.n_components]
            j = 1 + self.n_components
            for k in range(self.n_components):
                Yi[:, j:j + self.n_components - k] =\
                    (U[:, k:k + 1] * U[:, k:self.n_components])
                j += self.n_components - k

            Q, R = np.linalg.qr(Yi)
            w = np.array(Q[:, self.n_components + 1:])
            S = w.sum(0)

            S[np.where(np.abs(S) < 1e-5)] = 1.
            w /= S
            neighbors_x, neighbors_y = np.meshgrid(neighbors_i, neighbors_i)
            W[neighbors_x, neighbors_y] += w @ w.T

        _, sig, VT = np.linalg.svd(W, full_matrices=0)
        idx = np.argsort(sig)[1:self.n_components + 1]
        Y = VT[idx, :] * np.sqrt(n_samples)

        _, sig, VT = np.linalg.svd(Y, full_matrices=0)
        S = np.matrix(np.diag(sig ** (-1)))
        R = VT.T * S * VT

        return np.array(Y * R).T

    def _construct_graph(self, X: np.ndarray):
        n_samples = X.shape[0]
        kd_tree = KDTree(X.copy())
        adjacency_index_matrix = np.ones([n_samples, self.n_neighbors],
                                         dtype=int)
        for idx in range(n_samples):
            _, neighbors = kd_tree.query(X[idx, :],
                                         k=self.n_neighbors + 1,
                                         p=2)
            neighbors = neighbors.tolist()
            neighbors.remove(idx)
            adjacency_index_matrix[idx, :] = np.array([neighbors])

        return adjacency_index_matrix
