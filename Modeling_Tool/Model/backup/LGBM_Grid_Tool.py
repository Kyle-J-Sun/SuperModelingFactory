import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, classification_report
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import seaborn as sns
import time
import joblib
import warnings
warnings.filterwarnings('ignore')

class LGBGridSearch:
    """
    LightGBM网格搜索优化类
    适用于信贷风控模型的超参数调优
    """
    
    def __init__(self, scoring='roc_auc', cv_folds=5, n_jobs=-1, random_state=42):
        """
        初始化网格搜索参数
        
        参数:
        scoring: 评估指标，默认为'roc_auc'
        cv_folds: 交叉验证折数，默认为5
        n_jobs: 并行作业数，默认为-1（使用所有CPU核心）
        random_state: 随机种子，默认为42
        """
        self.scoring = scoring
        self.cv_folds = cv_folds
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.best_model = None
        self.grid_search = None
        self.results_df = None
        
    def create_param_grid(self, complexity='medium'):
        """
        创建不同复杂度的参数网格
        
        参数:
        complexity: 网格复杂度，'simple'/'medium'/'complex'
        
        返回:
        参数字典
        """
        if complexity == 'simple':
            # 简单网格 - 快速测试
            param_grid = {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1],
                'num_leaves': [31, 63],
                'max_depth': [5, 10],
                'min_child_samples': [20, 50]
            }
            
        elif complexity == 'medium':
            # 中等网格 - 推荐用于生产环境
            param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'num_leaves': [31, 63, 127],
                'max_depth': [5, 10, -1],  # -1表示无限制
                'min_child_samples': [20, 50, 100],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0],
                'reg_alpha': [0, 0.1, 1],
                'reg_lambda': [0, 0.1, 1]
            }
            
        else:  # complex
            # 复杂网格 - 全面搜索但耗时较长
            param_grid = {
                'n_estimators': [50, 100, 200, 300, 500],
                'learning_rate': [0.005, 0.01, 0.05, 0.1],
                'num_leaves': [15, 31, 63, 127, 255],
                'max_depth': [3, 5, 7, 10, -1],
                'min_child_samples': [10, 20, 50, 100],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'reg_alpha': [0, 0.001, 0.01, 0.1, 1],
                'reg_lambda': [0, 0.001, 0.01, 0.1, 1]
            }
        
        print(f"创建 {complexity} 参数网格，包含 {len([x for x in ParameterGrid(param_grid)])} 种组合")
        return param_grid
    
    def fit(self, X, y, param_grid=None, complexity='medium', refit=True):
        """
        执行网格搜索
        
        参数:
        X: 特征数据
        y: 目标变量
        param_grid: 自定义参数网格，如果为None则使用默认网格
        complexity: 网格复杂度
        refit: 是否用最佳参数重新拟合模型
        """
        if param_grid is None:
            param_grid = self.create_param_grid(complexity)
        
        # 创建LightGBM分类器
        lgb_model = lgb.LGBMClassifier(
            random_state=self.random_state,
            n_jobs=1,  # 避免与GridSearchCV的n_jobs冲突
            verbose=-1
#             force_row_wise=True  # 确保在最新版本中兼容
        )
        
        # 创建分层K折交叉验证
        cv = StratifiedKFold(
            n_splits=self.cv_folds, 
            shuffle=True, 
            random_state=self.random_state
        )
        
        # 创建GridSearchCV对象
        self.grid_search = GridSearchCV(
            estimator=lgb_model,
            param_grid=param_grid,
            cv=cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            verbose=1,
            refit=refit,
            return_train_score=True
        )
        
        print("开始LightGBM网格搜索...")
        start_time = time.time()
        
        # 执行网格搜索
        self.grid_search.fit(X, y)
        
        end_time = time.time()
        self.search_time = end_time - start_time
        
        print(f"网格搜索完成! 耗时: {self.search_time:.2f}秒")
        
        # 保存最佳模型
        if refit:
            self.best_model = self.grid_search.best_estimator_
        
        # 创建结果DataFrame
        self._create_results_dataframe()
        
        return self
    
    def _create_results_dataframe(self):
        """创建网格搜索结果的DataFrame"""
        self.results_df = pd.DataFrame(self.grid_search.cv_results_)
        
        # 选择重要的列
        important_cols = [
            'mean_test_score', 'std_test_score', 'rank_test_score',
            'mean_train_score', 'std_train_score'
        ]
        
        # 添加参数列
        param_cols = [col for col in self.results_df.columns if col.startswith('param_')]
        important_cols.extend(param_cols)
        
        self.results_df = self.results_df[important_cols].sort_values('rank_test_score')
        
        return self.results_df
    
    def get_best_params(self):
        """获取最佳参数"""
        if self.grid_search is None:
            raise ValueError("请先执行fit方法")
        
        return self.grid_search.best_params_
    
    def get_best_score(self):
        """获取最佳分数"""
        if self.grid_search is None:
            raise ValueError("请先执行fit方法")
        
        return self.grid_search.best_score_
    
    def predict(self, X):
        """使用最佳模型进行预测"""
        if self.best_model is None:
            raise ValueError("没有可用的最佳模型")
        
        return self.best_model.predict(X)
    
    def predict_proba(self, X):
        """使用最佳模型进行概率预测"""
        if self.best_model is None:
            raise ValueError("没有可用的最佳模型")
        
        return self.best_model.predict_proba(X)
    
    def evaluate_model(self, X_test, y_test):
        """
        在测试集上评估最佳模型
        
        返回:
        包含各项指标的字典
        """
        if self.best_model is None:
            raise ValueError("没有可用的最佳模型")
        
        # 预测
        y_pred = self.best_model.predict(X_test)
        y_pred_proba = self.best_model.predict_proba(X_test)[:, 1]
        
        # 计算各项指标
        metrics = {
            'test_auc': roc_auc_score(y_test, y_pred_proba),
            'test_f1': f1_score(y_test, y_pred),
            'test_precision': precision_score(y_test, y_pred),
            'test_recall': recall_score(y_test, y_pred),
            'test_accuracy': np.mean(y_test == y_pred)
        }
        
        return metrics
    
    def plot_search_results(self, top_n=10):
        """
        可视化网格搜索结果
        
        参数:
        top_n: 显示前N个最佳结果
        """
        if self.results_df is None:
            raise ValueError("请先执行fit方法")
        
        # 获取前top_n个结果
        top_results = self.results_df.head(top_n).copy()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 前N个参数组合的测试分数
        axes[0, 0].errorbar(
            range(len(top_results)), 
            top_results['mean_test_score'],
            yerr=top_results['std_test_score'],
            fmt='o-', capsize=5
        )
        axes[0, 0].set_xlabel('参数组合排名')
        axes[0, 0].set_ylabel(f'{self.scoring} 分数')
        axes[0, 0].set_title(f'前{top_n}个参数组合的交叉验证性能')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 训练分数 vs 测试分数
        axes[0, 1].scatter(
            top_results['mean_train_score'],
            top_results['mean_test_score'],
            s=50, alpha=0.7
        )
        max_val = max(top_results[['mean_train_score', 'mean_test_score']].max())
        min_val = min(top_results[['mean_train_score', 'mean_test_score']].min())
        axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
        axes[0, 1].set_xlabel('训练分数')
        axes[0, 1].set_ylabel('测试分数')
        axes[0, 1].set_title('训练 vs 测试性能')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 参数重要性分析
        param_importance = self._calculate_param_importance()
        if param_importance is not None:
            y_pos = np.arange(len(param_importance))
            axes[1, 0].barh(y_pos, param_importance['importance'])
            axes[1, 0].set_yticks(y_pos)
            axes[1, 0].set_yticklabels(param_importance['parameter'])
            axes[1, 0].set_xlabel('参数重要性')
            axes[1, 0].set_title('超参数重要性分析')
        
        # 4. 学习率和树数量的关系
        if 'param_learning_rate' in top_results.columns and 'param_n_estimators' in top_results.columns:
            scatter = axes[1, 1].scatter(
                top_results['param_learning_rate'].astype(float),
                top_results['param_n_estimators'].astype(float),
                c=top_results['mean_test_score'],
                cmap='viridis', s=100
            )
            axes[1, 1].set_xlabel('学习率')
            axes[1, 1].set_ylabel('树数量')
            axes[1, 1].set_title('学习率 vs 树数量 (颜色表示性能)')
            plt.colorbar(scatter, ax=axes[1, 1])
        
        plt.tight_layout()
        plt.show()
    
    def _calculate_param_importance(self):
        """
        计算参数重要性（基于性能变化的范围）
        """
        if self.results_df is None:
            return None
        
        # 获取参数列
        param_cols = [col for col in self.results_df.columns if col.startswith('param_')]
        
        param_importance = []
        
        for param in param_cols:
            # 计算该参数不同取值的平均分数范围
            param_values = self.results_df.groupby(param)['mean_test_score'].agg(['mean', 'std'])
            
            if len(param_values) > 1:  # 只有多个取值时才计算重要性
                importance = param_values['mean'].max() - param_values['mean'].min()
                param_importance.append({
                    'parameter': param.replace('param_', ''),
                    'importance': importance
                })
        
        if param_importance:
            return pd.DataFrame(param_importance).sort_values('importance', ascending=False)
        return None
    
    def save_model(self, filepath):
        """保存最佳模型"""
        if self.best_model is None:
            raise ValueError("没有可用的最佳模型")
        
        joblib.dump(self.best_model, filepath)
        print(f"模型已保存到: {filepath}")
    
    def save_results(self, filepath):
        """保存网格搜索结果"""
        if self.results_df is None:
            raise ValueError("没有可用的结果")
        
        self.results_df.to_csv(filepath, index=False)
        print(f"结果已保存到: {filepath}")
    
    def print_summary(self):
        """打印搜索摘要"""
        if self.grid_search is None:
            raise ValueError("请先执行fit方法")
        
        print("=" * 60)
        print("LightGBM网格搜索摘要")
        print("=" * 60)
        print(f"搜索时间: {self.search_time:.2f}秒")
        print(f"最佳分数 ({self.scoring}): {self.grid_search.best_score_:.6f}")
        print(f"最佳参数:")
        for key, value in self.grid_search.best_params_.items():
            print(f"  {key}: {value}")
        print(f"总参数组合数: {len(self.results_df)}")
        print("=" * 60)

# 辅助函数：从sklearn.model_selection导入ParameterGrid
from sklearn.model_selection import ParameterGrid

# 示例使用
if __name__ == "__main__":
    # 创建示例数据 - 替换为您的实际数据
    print("生成示例数据...")
    X, y = make_classification(
        n_samples=10000, 
        n_features=20, 
        n_informative=8, 
        n_redundant=5,
        n_clusters_per_class=1,
        flip_y=0.05,  # 添加噪声
        random_state=42
    )
    
    # 划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
    print(f"训练集违约率: {y_train.mean():.3f}")
    
    # 创建网格搜索实例
    lgb_searcher = LGBGridSearch(
        scoring='roc_auc',
        cv_folds=5,
        n_jobs=-1,
        random_state=42
    )
    
    # 执行网格搜索（使用中等复杂度网格）
    print("\n开始网格搜索...")
    lgb_searcher.fit(
        X_train, 
        y_train, 
        complexity='medium',  # 使用中等复杂度网格
        refit=True
    )
    
    # 打印摘要
    lgb_searcher.print_summary()
    
    # 在测试集上评估最佳模型
    test_metrics = lgb_searcher.evaluate_model(X_test, y_test)
    print("\n测试集性能:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # 可视化结果
    print("\n生成可视化图表...")
    lgb_searcher.plot_search_results(top_n=15)
    
    # 保存模型和结果
    lgb_searcher.save_model('best_lgb_model.pkl')
    lgb_searcher.save_results('grid_search_results.csv')
    
    # 特征重要性分析（如果可用）
    if lgb_searcher.best_model is not None:
        feature_importance = pd.DataFrame({
            'feature': [f'feature_{i}' for i in range(X.shape[1])],
            'importance': lgb_searcher.best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n前10个最重要特征:")
        print(feature_importance.head(10))
        
        # 绘制特征重要性
        plt.figure(figsize=(10, 8))
        top_features = feature_importance.head(15)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('特征重要性')
        plt.title('LightGBM特征重要性')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

# 高级用法：自定义参数网格
def create_custom_param_grid():
    """创建自定义参数网格的示例"""
    custom_grid = {
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200, 300],
        'num_leaves': [31, 63, 127],
        'max_depth': [5, 10, -1],
        'min_child_samples': [20, 50],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'reg_alpha': [0, 0.1],
        'reg_lambda': [0, 0.1]
    }
    return custom_grid

# 如果需要使用自定义网格：
# custom_grid = create_custom_param_grid()
# lgb_searcher.fit(X_train, y_train, param_grid=custom_grid, refit=True)