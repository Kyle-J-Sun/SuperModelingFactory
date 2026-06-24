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
        初始化分布适配器
        
        Parameters:
        -----------
        method: str
            'density_ratio': 密度比估计
            'kl_divergence': KL散度加权
            'covariate_shift': 协变量偏移修正
        """
        self.method = method
        self.sample_weights = None
        self.feature_importances = None
        
    def estimate_density_ratio(self, X_train, X_oot):
        """
        使用密度比估计方法计算样本权重
        KLIEP/KMM等方法的简化实现
        """
        from sklearn.neighbors import KernelDensity
        
        # 使用KDE估计密度
        kde_train = KernelDensity(kernel='gaussian', bandwidth=0.5)
        kde_oot = KernelDensity(kernel='gaussian', bandwidth=0.5)
        
        kde_train.fit(X_train)
        kde_oot.fit(X_oot)
        
        # 计算密度比: p_oot(x) / p_train(x)
        log_density_train = kde_train.score_samples(X_train)
        log_density_oot = kde_oot.score_samples(X_train)
        
        # 防止除零和数值不稳定
        density_ratio = np.exp(log_density_oot - log_density_train)
        
        # 截断处理异常值
        density_ratio = np.clip(density_ratio, 0.1, 10)
        
        # 归一化
        density_ratio = density_ratio / density_ratio.mean()
        
        return density_ratio
    
    def covariate_shift_weighting(self, X_train, X_oot):
        """
        使用领域分类器估计样本重要性权重
        """
        from sklearn.linear_model import LogisticRegression
        
        # 创建领域标签: 训练集为0, OOT为1
        n_train = len(X_train)
        n_oot = len(X_oot)
        
        X_combined = np.vstack([X_train, X_oot])
        y_domain = np.hstack([np.zeros(n_train), np.ones(n_oot)])
        
        # 训练领域分类器
        domain_classifier = LogisticRegression(
            C=1.0, max_iter=1000, random_state=42
        )
        domain_classifier.fit(X_combined, y_domain)
        
        # 预测训练样本来自OOT的概率
        p_oot = domain_classifier.predict_proba(X_train)[:, 1]
        
        # 计算权重: p(oot|x) / p(train|x)
        # 防止数值不稳定
        epsilon = 1e-10
        weights = p_oot / (1 - p_oot + epsilon)
        
        # 使用beta分布平滑权重
        weights = np.clip(weights, 0.1, 10)
        weights = weights / weights.mean()
        
        return weights
    
    def fit(self, X_train, X_oot, y_train=None):
        """
        计算适应OOT分布的样本权重
        """
        if self.method == 'density_ratio':
            self.sample_weights = self.estimate_density_ratio(X_train, X_oot)
        elif self.method == 'covariate_shift':
            self.sample_weights = self.covariate_shift_weighting(X_train, X_oot)
        else:
            # 默认使用混合方法
            w1 = self.estimate_density_ratio(X_train, X_oot)
            w2 = self.covariate_shift_weighting(X_train, X_oot)
            self.sample_weights = (w1 + w2) / 2
        
        return self
    
    def get_weights(self):
        """返回样本权重"""
        return self.sample_weights
    
    def visualize_distribution_comparison(self, X_train, X_oot, features=None, n_features=5):
        """
        可视化训练集和OOT集的分布差异
        """
        if features is None:
            # 选择方差最大的特征
            variances = np.var(X_train, axis=0)
            features = np.argsort(variances)[-n_features:]
        
        fig, axes = plt.subplots(1, n_features, figsize=(5*n_features, 4))
        
        for idx, feature_idx in enumerate(features[:n_features]):
            ax = axes[idx] if n_features > 1 else axes
            
            # 绘制分布
            sns.kdeplot(X_train[:, feature_idx], ax=ax, label='Train', fill=True, alpha=0.5)
            sns.kdeplot(X_oot[:, feature_idx], ax=ax, label='OOT', fill=True, alpha=0.5)
            
            ax.set_title(f'Feature {feature_idx}')
            ax.legend()
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
        
        plt.tight_layout()
        plt.show()