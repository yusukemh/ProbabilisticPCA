import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

class PPCA():
    def __init__(self, n_components, sigma=None):
        self.n_components_ = n_components
        self.mean_ = None
        self.components_ = None
        self.sigma_ = sigma # noise_variance
        self.b_fitted = False
    
    def fit(self, X):
        import numpy as np
        import scipy
        import matplotlib.pyplot as plt
        # eigen decomposition
        self.mean_ = X.mean(axis=0)
        X_ = X - self.mean_
        S = X_.T.dot(X_) * (1.0/X_.shape[0])
        eigenvalues, eigenvectors = np.linalg.eig(S)
        eigenvalues = np.real(eigenvalues)
        a = eigenvalues.argsort()[::-1]
        
        eigenvalues = eigenvalues[a]
        eigenvectors = eigenvectors[a]
        eigenvectors = np.real(eigenvectors)

        # ugly math
        if self.sigma_ is None:
            self.sigma_ = np.mean(eigenvalues[self.n_components_:])
        
        # self.components_ = eigenvectors[:, :self.n_components_]
        L = np.diag(eigenvalues[:self.n_components_])# watch out for this for n_components_ == 1
        if self.n_components_ == 1:
            L = L[0] - 1
            L = np.sqrt(L)
            self.components_ = ((eigenvectors[:, 0]) * L).reshape(-1, 1)
        else:
            L -= self.sigma_ * np.identity(self.n_components_)
            L = scipy.linalg.sqrtm(L)
            self.components_ = eigenvectors[:, :self.n_components_].dot(L)
        
        self.b_fitted = True
    
    def transform(self, X):
        if not self.b_fitted:
            print("Not so fast")
            return
        M = np.matmul(self.components_.T, self.components_) + self.sigma_ * np.identity(self.n_components_)
        M = np.linalg.inv(M)
        Z = []
        for x in X:
            z = M.dot(self.components_.T).dot(x - self.mean_)
            Z.append(z)
        
        return np.array(Z)
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    def plot_gaussian_contour(
        self,
        resolution,    # play with this parameter and you'll figure out
        xlim,          # xlim for which probability density is calculated. do not confuse with xlim of plot.
        ylim,          # ylim for which probability density is calculated. do not confuse with xlim of plot.
        means,         # mean vector for the multivariate gaussian
        cov,           # covariant matrix for the multivariate gaussian
        levels=5,      # 'levels' parameter for ax.contour(). Check their documentation yourself for more info pls.
        linewidths=1.2,# the linewidth of contour lines
        ax=None        # which ax to plot on
    ):
        # compute cartesian product of coordinates
        x = np.linspace(xlim[0], xlim[1], resolution)
        y = np.linspace(ylim[0], ylim[1], resolution)
        xx, yy = np.meshgrid(x, y)
        coordinates = np.dstack((xx, yy))
        # compute gaussian pdf for points in coordinates
        rv = multivariate_normal(means, cov)
        density = rv.pdf(coordinates)

        if ax is None:
            # fig, ax = plt.subplots(1)
            ax = plt.gca()
        ax.contour(xx, yy, density, colors='k', levels=levels, linewidths=linewidths)
        ax.set_aspect('equal')

        return ax
    
    def plot_noise_gaussian_contour(self, ax, xlim, ylim, resolution=200, levels=5, linewidths=0.2):
        if self.n_components_ != 1:
            print(f"Value Error: n_components_ has to be 1, but got {self.n_components_} indetead.")
            return
        
        if not self.b_fitted:
            print("Not so fast")
            return
        self.plot_gaussian_contour(
            resolution=resolution,
            xlim=xlim,
            ylim=ylim,
            means=[0, 0],
            cov=self.sigma_ * np.identity(2),
            levels=levels,
            linewidths=linewidths,
            ax=ax
        )
        return ax
    
    def plot_data_gaussian_contour(self, ax, xlim, ylim, resolution=200, levels=5, linewidths=0.2):
        self.plot_gaussian_contour(
            resolution=resolution,
            xlim=xlim,
            ylim=ylim,
            means=self.mean_,
            cov=np.matmul(self.components_, self.components_.T) + self.sigma_ * np.identity(2),
            levels=levels,
            linewidths=linewidths,
            ax=ax
        )
    
    def plot_latent_gaussian_contour(self, ax, xlim, ylim, resolution=200, levels=5, linewidths=0.2):
        self.plot_gaussian_contour(
            resolution=resolution,
            xlim=xlim,
            ylim=ylim,
            means=self.mean_,
            cov=np.identity(2) * np.linalg.norm(self.components_),
            levels=levels,
            linewidths=linewidths,
            ax=ax
        )