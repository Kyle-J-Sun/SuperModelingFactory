import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV

class DistributionAdaptation:
    """
    Sample weights that adapt a training set to an out-of-time (OOT) distribution.

    The class computes one importance weight per training row, so that the weighted
    training distribution moves towards the OOT distribution. The weights are
    produced either by `fit` (stored in `sample_weights`, read with `get_weights`) or
    by calling `estimate_density_ratio` / `covariate_shift_weighting` directly.

    Parameters
    ----------
    method : str, default 'density_ratio'
        Weighting method used by `fit`:

        - ``'density_ratio'``: density-ratio estimation
          (`estimate_density_ratio`).
        - ``'covariate_shift'``: covariate-shift correction
          (`covariate_shift_weighting`).
        - any other value, for example ``'kl_divergence'``: hybrid weighting,
          the average of the two weight vectors above. No separate KL-divergence
          weighting is implemented, and the value is not validated.

    Attributes
    ----------
    method : str
        Weighting method passed to the constructor; only `fit` reads it.
    sample_weights : numpy.ndarray or None
        Weights computed by the last call to `fit` (one per training row);
        ``None`` before `fit` is called.
    feature_importances : None
        Reserved attribute that is always ``None``; nothing in the class sets it.
    """
    def __init__(self, method='density_ratio'):
        """
        Initialize the distribution adapter.

        Parameters
        ----------
        method : str, default 'density_ratio'
            Weighting method used by `fit`:

            - ``'density_ratio'``: density-ratio estimation
              (`estimate_density_ratio`).
            - ``'covariate_shift'``: covariate-shift correction
              (`covariate_shift_weighting`).
            - any other value, for example ``'kl_divergence'``: hybrid weighting,
              the average of the two weight vectors above. No separate KL-divergence
              weighting is implemented, and the value is not validated.
        """
        self.method = method
        self.sample_weights = None
        self.feature_importances = None
        
    def estimate_density_ratio(self, X_train, X_oot):
        """
        Compute sample weights using density-ratio estimation.
        A simplified implementation of methods such as KLIEP/KMM.

        Two Gaussian kernel density estimates (bandwidth 0.5) are fitted, one on
        ``X_train`` and one on ``X_oot``. The weight of each training row is the ratio
        ``p_oot(x) / p_train(x)`` evaluated at that row. The ratios are clipped to
        ``[0.1, 10]`` and then divided by their mean.

        Parameters
        ----------
        X_train : array-like of shape (n_train, n_features)
            Training features (numpy array or DataFrame), numeric without missing values.
        X_oot : array-like of shape (n_oot, n_features)
            OOT features with the same columns, in the same order, as ``X_train``.

        Returns
        -------
        numpy.ndarray of shape (n_train,)
            Importance weight of each training row, in the row order of ``X_train``.
            The weights average exactly 1.

        Raises
        ------
        ValueError
            If the inputs contain NaN or have a different number of columns
            (raised by scikit-learn).

        Notes
        -----
        The kernel bandwidth is fixed at 0.5 and no feature scaling is applied, so the
        features should be on comparable scales. The method does not read ``method``
        and does not store its result in ``sample_weights`` (`fit` does).
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

        A logistic regression (``C=1.0``, ``max_iter=1000``, ``random_state=42``) is
        trained on the stacked rows of ``X_train`` (domain 0) and ``X_oot`` (domain 1).
        The weight of each training row is the odds ``p(oot|x) / (1 - p(oot|x) + 1e-10)``
        predicted for that row. The odds are clipped to ``[0.1, 10]`` and then divided
        by their mean.

        Parameters
        ----------
        X_train : array-like of shape (n_train, n_features)
            Training features (numpy array or DataFrame), numeric without missing values.
        X_oot : array-like of shape (n_oot, n_features)
            OOT features with the same columns, in the same order, as ``X_train``.

        Returns
        -------
        numpy.ndarray of shape (n_train,)
            Importance weight of each training row, in the row order of ``X_train``.
            The weights average exactly 1.

        Raises
        ------
        ValueError
            If the inputs contain NaN or have a different number of columns
            (raised by numpy or scikit-learn).

        Notes
        -----
        No feature scaling is applied and the odds are not corrected for the different
        sizes of ``X_train`` and ``X_oot`` before clipping. The method does not read
        ``method`` and does not store its result in ``sample_weights`` (`fit` does).
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

        The weights are computed according to ``method`` and stored in
        ``sample_weights``: ``'density_ratio'`` uses `estimate_density_ratio`,
        ``'covariate_shift'`` uses `covariate_shift_weighting`, and any other value
        uses the average of the two weight vectors.

        Parameters
        ----------
        X_train : array-like of shape (n_train, n_features)
            Training features (numpy array or DataFrame), numeric without missing values.
        X_oot : array-like of shape (n_oot, n_features)
            OOT features with the same columns, in the same order, as ``X_train``.
        y_train : array-like or None, default None
            Not used; accepted only for interface compatibility.

        Returns
        -------
        DistributionAdaptation
            The fitted instance itself (``self``), with ``sample_weights`` set to a
            numpy array of shape (n_train,).
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
        """
        Return the sample weights.

        Returns
        -------
        numpy.ndarray or None
            The weights stored by the last call to `fit` (one per training row), or
            ``None`` if `fit` has not been called.
        """
        return self.sample_weights
    
    def visualize_distribution_comparison(self, X_train, X_oot, features=None, n_features=5):
        """
        Visualize the distribution differences between the training set and the OOT set.

        Draws one filled kernel-density plot per feature (training set labelled
        ``Train``, OOT set labelled ``OOT``) on a single row of ``n_features``
        subplots, then calls ``plt.tight_layout()`` and ``plt.show()``.

        Parameters
        ----------
        X_train : numpy.ndarray of shape (n_train, n_total_features)
            Training features. Must be a 2-D numpy array because columns are
            selected by position; a DataFrame raises an error.
        X_oot : numpy.ndarray of shape (n_oot, n_total_features)
            OOT features with the same column order as ``X_train``.
        features : sequence of int or None, default None
            Column positions (not names) to plot; only the first ``n_features``
            entries are used. If ``None``, the ``n_features`` columns of ``X_train``
            with the largest variance are plotted, in ascending order of variance.
        n_features : int, default 5
            Number of subplots in the figure and maximum number of features plotted.

        Returns
        -------
        None
            The figure is only displayed.

        Notes
        -----
        The figure always has ``n_features`` subplots, so subplots are left empty when
        fewer features are available. Each subplot is titled ``Feature <column position>``.
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