import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np

from sklearn.datasets import make_swiss_roll
from sklearn.cluster import AgglomerativeClustering

from lle import LLE
from hessian_lle import HessianLLE


def plot_3d(X, label, title):
    fig = plt.figure(figsize=(7, 7))
    ax = p3.Axes3D(fig)
    ax.view_init(7, -80)
    for l in np.unique(label):
        ax.scatter(X[label == l, 0], X[label == l, 1], X[label == l, 2],
                   color=plt.cm.jet(float(l) / np.max(label + 1)),
                   s=20, edgecolor='k')
    fig.savefig(title, dpi=50)


def plot_2d(X, label, title):
    plt.figure(figsize=(7, 7))
    plt.title("LLE")
    for l in np.unique(label):
        plt.scatter(X[label == l, 0], X[label == l, 1],
                    color=plt.cm.jet(float(l) / np.max(label + 1)),
                    s=20, edgecolor='k')
    plt.savefig(title, dpi=50)


X, _ = make_swiss_roll(1000, noise=0, random_state=0)
ward = AgglomerativeClustering(n_clusters=8, linkage='ward').fit(X)
label = ward.labels_

lle = LLE(n_neighbors=12, n_components=2)
x_lle = lle.fit_transform(X)

hlle = HessianLLE(n_neighbors=12, n_components=2)
x_hlle = hlle.fit_transform(X)

plot_3d(X, label, 'data')
plot_2d(x_lle, label, 'lle')
plot_2d(x_hlle, label, 'hessian_lle')
