import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV

class DistributionAdaptation:
    def __init__(self, method='density_ratio'):
        """
        Initialize the distribution adapter.
        
        Parameters:
        -----------
        method: str
            'density_ratio': density-ratio estimation
            'kl_divergence': KL-divergence weighting
            'covariate_shift': covariate-shift correction
        """
        self.method = method
        self.sample_weights = None
        self.feature_importances = None
        
    def estimate_density_ratio(self, X_train, X_oot):
        """
        Compute sample weights using density-ratio estimation.
        A simplified implementation of methods such as KLIEP/KMM.
        """
        from sklearn.neighbors import KernelDensity
        
        # Estimate the densities with KDE
        kde_train = KernelDensity(kernel='gaussian', bandwidth=0.5)
        kde_oot = KernelDensity(kernel='gaussian', bandwidth=0.5)
        
        kde_train.fit(X_train)
        kde_oot.fit(X_oot)
        
        # Compute the density ratio: p_oot(x) / p_train(x)
        log_density_train = kde_train.score_samples(X_train)
        log_density_oot = kde_oot.score_samples(X_train)
        
        # Guard against division by zero and numerical instability
        density_ratio = np.exp(log_density_oot - log_density_train)
        
        # Clip outliers
        density_ratio = np.clip(density_ratio, 0.1, 10)
        
        # Normalize
        density_ratio = density_ratio / density_ratio.mean()
        
        return density_ratio
    
    def covariate_shift_weighting(self, X_train, X_oot):
        """
        Estimate sample importance weights using a domain classifier.
        """
        from sklearn.linear_model import LogisticRegression
        
        # Create domain labels: 0 for the training set, 1 for OOT
        n_train = len(X_train)
        n_oot = len(X_oot)
        
        X_combined = np.vstack([X_train, X_oot])
        y_domain = np.hstack([np.zeros(n_train), np.ones(n_oot)])
        
        # Train the domain classifier
        domain_classifier = LogisticRegression(
            C=1.0, max_iter=1000, random_state=42
        )
        domain_classifier.fit(X_combined, y_domain)
        
        # Predict the probability that each training sample comes from OOT
        p_oot = domain_classifier.predict_proba(X_train)[:, 1]
        
        # Compute the weights: p(oot|x) / p(train|x)
        # Guard against numerical instability
        epsilon = 1e-10
        weights = p_oot / (1 - p_oot + epsilon)
        
        # Smooth the weights using a beta distribution
        weights = np.clip(weights, 0.1, 10)
        weights = weights / weights.mean()
        
        return weights
    
    def fit(self, X_train, X_oot, y_train=None):
        """
        Compute sample weights that adapt the training distribution to the OOT distribution.
        """
        if self.method == 'density_ratio':
            self.sample_weights = self.estimate_density_ratio(X_train, X_oot)
        elif self.method == 'covariate_shift':
            self.sample_weights = self.covariate_shift_weighting(X_train, X_oot)
        else:
            # By default, use the hybrid method (average of both weights)
            w1 = self.estimate_density_ratio(X_train, X_oot)
            w2 = self.covariate_shift_weighting(X_train, X_oot)
            self.sample_weights = (w1 + w2) / 2
        
        return self
    
    def get_weights(self):
        """Return the sample weights."""
        return self.sample_weights
    
    def visualize_distribution_comparison(self, X_train, X_oot, features=None, n_features=5):
        """
        Visualize the distribution differences between the training set and the OOT set.
        """
        if features is None:
            # Select the features with the largest variance
            variances = np.var(X_train, axis=0)
            features = np.argsort(variances)[-n_features:]
        
        fig, axes = plt.subplots(1, n_features, figsize=(5*n_features, 4))
        
        for idx, feature_idx in enumerate(features[:n_features]):
            ax = axes[idx] if n_features > 1 else axes
            
            # Plot the distributions
            sns.kdeplot(X_train[:, feature_idx], ax=ax, label='Train', fill=True, alpha=0.5)
            sns.kdeplot(X_oot[:, feature_idx], ax=ax, label='OOT', fill=True, alpha=0.5)
            
            ax.set_title(f'Feature {feature_idx}')
            ax.legend()
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
        
        plt.tight_layout()
        plt.show()