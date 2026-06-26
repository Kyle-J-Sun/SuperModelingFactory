"""
梯度提升模型训练工具包
=============================

本模块提供LightGBM和XGBoost模型的快速训练和评估功能，
包括模型训练、特征重要性提取等常用操作。

Functions
---------
set_num_leaves
    根据最大深度计算叶子节点数，避免过拟合。
lgb_model
    快速训练LightGBM模型。
lgb_varimp
    获取LightGBM特征重要性。
lgbm_quick_train
    快速训练LightGBM模型（使用DataFrame接口）。
xgb_model
    训练XGBoost模型。
xgb_varimp
    获取XGBoost特征重要性。
xgbm_quick_train
    快速训练XGBoost模型（使用DataFrame接口）。

Classes
-------
LightGBMModel
    LightGBM模型封装类，提供统一的训练和评估接口。
XGBoostModel
    XGBoost模型封装类，提供统一的训练和评估接口。
GradientBoostingModel
    统一封装类，支持LightGBM和XGBoost切换。

Examples
--------
# 函数式调用
>>> model = lgb_model(x_train, y_train, x_val, y_val, params)
>>> varimp = lgb_varimp(model)

# 类封装调用
>>> lgb_model = LightGBMModel(params)
>>> lgb_model.fit(x_train, y_train, x_val, y_val)
>>> varimp = lgb_model.get_feature_importance()

# 统一接口调用
>>> model = GradientBoostingModel('lgb', params)
>>> model.fit(x_train, y_train, x_val, y_val)
"""

import pandas as pd
import numpy as np
from Modeling_Tool.Core.utils import load_model, save_model
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score

# lightgbm and xgboost are imported lazily inside each function/method to
# avoid triggering the lightgbm → dask → numpy_compat → np.float chain at
# module import time. Old dask versions (<2022) use np.float which was
# removed in NumPy 1.20+, causing AttributeError on import.
def _get_lgb():
    try:
        import lightgbm as lgb
        return lgb
    except ImportError:
        raise ImportError("lightgbm is required. Install with: pip install lightgbm>=3.3.0")

def _get_xgb():
    try:
        import xgboost as xgb
        return xgb
    except ImportError:
        raise ImportError("xgboost is required. Install with: pip install xgboost>=1.7.0")

# ============================================================================
# 工具函数（保持独立）
# ============================================================================

def set_num_leaves(max_depth=5, wgt=1):
    """根据最大深度设置叶子节点数，避免过拟合。

    根据给定的最大深度和权重系数，计算合适的叶子节点数量。
    计算公式：2^max_depth - 2^max_depth * wgt

    Parameters
    ----------
    max_depth : int, default 5
        树的最大深度
    wgt : float, default 1
        权重系数，取値范围[0, 1]

    Returns
    -------
    int
        建议的叶子节点数

    Examples
    --------
    >>> set_num_leaves(max_depth=5)
    32
    >>> set_num_leaves(max_depth=5, wgt=0.5)
    16
    """
    return int(2 ** max_depth - 2 ** max_depth * wgt)


def lgb_model(x, y, valx, valy, params_dict, wgt=None, init_score=None):
    """快速训练LightGBM模型。

    使用训练集和验证集训练LightGBM模型，支持早停机制。

    Parameters
    ----------
    x : array-like or pd.DataFrame
        训练集特征
    y : array-like
        训练集标签
    valx : array-like or pd.DataFrame
        验证集特征
    valy : array-like
        验证集标签
    params_dict : dict
        LightGBM参数字典
    wgt : array-like, optional
        样本权重
    init_score : array-like, optional
        初始化分数

    Returns
    -------
    lgb.LGBMClassifier
        训练好的LightGBM模型

    Examples
    --------
    >>> params = {
    ...     'n_estimators': 100,
    ...     'max_depth': 5,
    ...     'learning_rate': 0.1,
    ...     'early_stopping_rounds': 20,
    ...     'eval_metric': 'auc'
    ... }
    >>> model = lgb_model(x_train, y_train, x_val, y_val, params)
    """
    lgb = _get_lgb()

    # lgb_params = {k: v for k, v in params_dict.items() if k != 'eval_metric'}
    # model = lgb.LGBMClassifier(**lgb_params)
    # model.fit(
    #     x, y,
    #     eval_set=[(valx, valy)],
    #     eval_metric=params_dict['eval_metric'] if 'eval_metric' in params_dict else params_dict['metric'],
    #     callbacks=[
    #         lgb.early_stopping(stopping_rounds=params_dict['early_stopping_rounds'], verbose=False)
    #     ],
    #     verbose=False,
    #     sample_weight=wgt
    # )

    lgb_params = {k: v for k, v in params_dict.items() if k != 'eval_metric'}
    model = lgb.LGBMClassifier(**lgb_params)
    # NOTE: `verbose=False` was removed from .fit() in lightgbm>=4. Log control
    # is now done via callbacks only (log_evaluation).
    model.fit(
        x, y,
        eval_set=[(valx, valy)],
        eval_metric=params_dict['eval_metric'] if 'eval_metric' in params_dict else params_dict.get('metric', 'auc'),
        callbacks=[
            lgb.early_stopping(stopping_rounds=params_dict['early_stopping_rounds'], verbose=False),
            lgb.log_evaluation(period=0),
        ],
        sample_weight=wgt,
        init_score=init_score
    )
    return model


def lgb_varimp(model):
    """获取LightGBM模型特征重要性。

    返回按特征重要性排序的DataFrame。

    Parameters
    ----------
    model : lgb.LGBMClassifier
        训练好的LightGBM模型

    Returns
    -------
    pd.DataFrame
        包含 feature 和 importance 列的DataFrame，按重要性降序排列

    Examples
    --------
    >>> varimp = lgb_varimp(model)
    >>> varimp.head(10)
    """
    feature_names = model.booster_.feature_name()
    importance = model.booster_.feature_importance(importance_type='gain')
    varimp_df = pd.DataFrame({'feature': feature_names, 'importance': importance})
    varimp_df = varimp_df.sort_values('importance', ascending=False).reset_index(drop=True)
    return varimp_df


def lgbm_quick_train(train_data, validation_data, x, y, params, wgt_col = None, cat_x_train=None):
    """快速训练LightGBM模型（使用DataFrame接口）。

    接受DataFrame格式的训练集和验证集，自动提取特征和标签。

    Parameters
    ----------
    train_data : pd.DataFrame
        训练数据集
    validation_data : pd.DataFrame
        验证数据集
    x : list of str
        特征列名列表
    y : str
        目标变量列名
    params : dict
        LightGBM参数字典
    wgt_col : str, optional
        样本权重列名
    cat_x_train : list of str, optional
        类别型特征列名列表

    Returns
    -------
    lgb.LGBMClassifier
        训练好的LightGBM模型

    Examples
    --------
    >>> model = lgbm_quick_train(
    ...     train_data=train_df,
    ...     validation_data=val_df,
    ...     x=['feat1', 'feat2'],
    ...     y='target',
    ...     params=params_dict
    ... )
    """
    lgb = _get_lgb()

    wgt = train_data[wgt_col] if wgt_col is not None else None
    model = lgb_model(
        x=train_data[x],
        y=train_data[y],
        valx=validation_data[x],
        valy=validation_data[y],
        params_dict=params,
        wgt=wgt
    )
    return model


def xgb_model(x, y, valx, valy, params_dict, sample_weight=None, sample_weight_eval_set=None, base_margin=None):
    """训练XGBoost模型。

    使用训练集和验证集训练XGBoost模型，支持早停机制。

    Parameters
    ----------
    x : array-like or pd.DataFrame
        训练集特征
    y : array-like
        训练集标签
    valx : array-like or pd.DataFrame
        验证集特征
    valy : array-like
        验证集标签
    params_dict : dict
        XGBoost参数字典
    sample_weight : array-like, optional
        训练集样本权重
    sample_weight_eval_set : list, optional
        验证集样本权重列表
    base_margin : array-like, optional
        基础边际度

    Returns
    -------
    xgb.XGBClassifier
        训练好的XGBoost模型

    Examples
    --------
    >>> params = {
    ...     'n_estimators': 100,
    ...     'max_depth': 5,
    ...     'learning_rate': 0.1,
    ...     'early_stopping_rounds': 20,
    ...     'eval_metric': 'auc'
    ... }
    >>> model = xgb_model(x_train, y_train, x_val, y_val, params)
    """
    xgb = _get_xgb()

    xgb_params = {k: v for k, v in params_dict.items() if k not in ('eval_metric',)}
    model = xgb.XGBClassifier(**xgb_params)
    model.fit(
        x, y,
        eval_set=[(valx, valy)],
        verbose=False,
        sample_weight=sample_weight,
        sample_weight_eval_set=sample_weight_eval_set,
        base_margin=base_margin
    )
    return model


def xgb_varimp(model):
    """获取XGBoost模型特征重要性。

    返回按特征重要性排序的DataFrame。

    Parameters
    ----------
    model : xgb.XGBClassifier
        训练好的XGBoost模型

    Returns
    -------
    pd.DataFrame
        包含 feature 和 importance 列的DataFrame，按重要性降序排列

    Examples
    --------
    >>> varimp = xgb_varimp(model)
    >>> varimp.head(10)
    """
    importance = model.get_booster().get_fscore()
    varimp_df = pd.DataFrame(
        list(importance.items()),
        columns=['feature', 'importance']
    ).sort_values('importance', ascending=False).reset_index(drop=True)
    return varimp_df


def xgbm_quick_train(train_data, validation_data, x, y, wgt_col, params,
                     sample_weight_eval_set=None):
    """快速训练XGBoost模型（使用DataFrame接口）。

    接受DataFrame格式的训练集和验证集，自动提取特征和标签。

    Parameters
    ----------
    train_data : pd.DataFrame
        训练数据集
    validation_data : pd.DataFrame
        验证数据集
    x : list of str
        特征列名列表
    y : str
        目标变量列名
    wgt_col : str
        样本权重列名
    params : dict
        XGBoost参数字典
    sample_weight_eval_set : list, optional
        验证集样本权重列表

    Returns
    -------
    xgb.XGBClassifier
        训练好的XGBoost模型

    Examples
    --------
    >>> model = xgbm_quick_train(
    ...     train_data=train_df,
    ...     validation_data=val_df,
    ...     x=['feat1', 'feat2'],
    ...     y='target',
    ...     wgt_col='weight',
    ...     params=params_dict
    ... )
    """
    xgb = _get_xgb()

    wgt = train_data[wgt_col] if wgt_col is not None else None
    model = xgb_model(
        x=train_data[x],
        y=train_data[y],
        valx=validation_data[x],
        valy=validation_data[y],
        params_dict=params,
        sample_weight=wgt,
        sample_weight_eval_set=sample_weight_eval_set
    )
    return model


class LightGBMModel:
    """
    LightGBM模型封装类。

    提供统一的LightGBM模型训练、预测、保存和加载接口。
    支持模型校准、特征重要性获取等功能。

    Parameters
    ----------
    params : dict
        LightGBM模型参数字典
    model : lgb.LGBMClassifier, optional
        预加载的模型实例

    Examples
    --------
    >>> lgb_clf = LightGBMModel(params)
    >>> lgb_clf.fit(x_train, y_train, x_val, y_val)
    >>> preds = lgb_clf.predict(x_test)
    """

    def __init__(self, params, model=None):
        """
        初始化LightGBM模型封装类。

        Parameters
        ----------
        params : dict
            LightGBM模型参数字典
        model : lgb.LGBMClassifier, optional
            预加载的模型实例
        """
        lgb = _get_lgb()
        self.params = params
        self.model = model
        self.feature_names_ = None

    def fit(self, x, y, valx, valy, wgt=None, init_score=None):
        """训练LightGBM模型。

        使用训练集和验证集训练模型，支持早停机制。

        Parameters
        ----------
        x : array-like or pd.DataFrame
            训练集特征
        y : array-like
            训练集标签
        valx : array-like or pd.DataFrame
            验证集特征
        valy : array-like
            验证集标签
        wgt : array-like, optional
            样本权重
        init_score : array-like, optional
            初始化分数

        Returns
        -------
        self
        """
        self.model = lgb_model(
            x=x, y=y, valx=valx, valy=valy,
            params_dict=self.params, wgt=wgt, init_score=init_score
        )
        if hasattr(x, 'columns'):
            self.feature_names_ = list(x.columns)
        return self

    def predict(self, x):
        """预测样本的概率。

        Parameters
        ----------
        x : array-like or pd.DataFrame
            预测特征

        Returns
        -------
        np.ndarray
            预测概率
        """
        return self.model.predict_proba(x)[:, 1]

    def get_feature_importance(self, importance_type='gain'):
        """获取特征重要性。

        Parameters
        ----------
        importance_type : str, default 'gain'
            特征重要性类型，可选 'gain' 或 'split'

        Returns
        -------
        pd.DataFrame
            包含 feature 和 importance 列的DataFrame
        """
        return lgb_varimp(self.model)

    def save(self, path):
        """保存模型。

        Parameters
        ----------
        path : str
            模型保存路径
        """
        save_model(self.model, path)

    def load(self, path):
        """加载模型。

        Parameters
        ----------
        path : str
            模型文件路径

        Returns
        -------
        self
        """
        self.model = load_model(path)
        return self

    def calibrate(self, x, y, method='sigmoid', cv='prefit'):
        """模型概率校准。

        Parameters
        ----------
        x : array-like
            校准特征
        y : array-like
            校准标签
        method : str, default 'sigmoid'
            校准方法，'sigmoid' 或 'isotonic'
        cv : str or int, default 'prefit'
            交叉验证方式

        Returns
        -------
        self
        """
        self.model = CalibratedClassifierCV(self.model, method=method, cv=cv)
        self.model.fit(x, y)
        return self

    def calibration_curve(self, x, y, n_bins=10):
        """获取校准曲线数据。

        Parameters
        ----------
        x : array-like
            特征
        y : array-like
            标签
        n_bins : int, default 10
            分箱数

        Returns
        -------
        tuple
            (fraction_of_positives, mean_predicted_value)
        """
        y_prob = self.predict(x)
        return calibration_curve(y, y_prob, n_bins=n_bins)

    def brier_score(self, x, y):
        """计算Brier分数。

        Parameters
        ----------
        x : array-like
            特征
        y : array-like
            标签

        Returns
        -------
        float
            Brier分数
        """
        y_prob = self.predict(x)
        return brier_score_loss(y, y_prob)

    def roc_auc(self, x, y):
        """计算ROC AUC。

        Parameters
        ----------
        x : array-like
            特征
        y : array-like
            标签

        Returns
        -------
        float
            ROC AUC分数
        """
        y_prob = self.predict(x)
        return roc_auc_score(y, y_prob)


class XGBoostModel:
    """
    XGBoost模型封装类。

    提供统一的XGBoost模型训练、预测、保存和加载接口。
    支持模型校准、特征重要性获取等功能。

    Parameters
    ----------
    params : dict
        XGBoost模型参数字典
    model : xgb.XGBClassifier, optional
        预加载的模型实例

    Examples
    --------
    >>> xgb_clf = XGBoostModel(params)
    >>> xgb_clf.fit(x_train, y_train, x_val, y_val)
    >>> preds = xgb_clf.predict(x_test)
    """

    def __init__(self, params, model=None):
        """
        初始化XGBoost模型封装类。

        Parameters
        ----------
        params : dict
            XGBoost模型参数字典
        model : xgb.XGBClassifier, optional
            预加载的模型实例
        """
        xgb = _get_xgb()
        self.params = params
        self.model = model
        self.feature_names_ = None

    def fit(self, x, y, valx, valy, sample_weight=None, sample_weight_eval_set=None, base_margin=None):
        """训练XGBoost模型。

        Parameters
        ----------
        x : array-like or pd.DataFrame
            训练集特征
        y : array-like
            训练集标签
        valx : array-like or pd.DataFrame
            验证集特征
        valy : array-like
            验证集标签
        sample_weight : array-like, optional
            样本权重
        sample_weight_eval_set : list, optional
            验证集样本权重列表
        base_margin : array-like, optional
            基础边际（init_score / log-odds 偏移），用于增量训练（warm-start）。

        Returns
        -------
        self
        """
        self.model = xgb_model(
            x=x, y=y, valx=valx, valy=valy,
            params_dict=self.params,
            sample_weight=sample_weight,
            sample_weight_eval_set=sample_weight_eval_set,
            base_margin=base_margin
        )
        if hasattr(x, 'columns'):
            self.feature_names_ = list(x.columns)
        return self

    def predict(self, x):
        """预测样本的概率。

        Parameters
        ----------
        x : array-like or pd.DataFrame
            预测特征

        Returns
        -------
        np.ndarray
            预测概率
        """
        return self.model.predict_proba(x)[:, 1]

    def get_feature_importance(self, importance_type='gain'):
        """获取特征重要性。

        Parameters
        ----------
        importance_type : str, default 'gain'
            特征重要性类型

        Returns
        -------
        pd.DataFrame
            包含 feature 和 importance 列的DataFrame
        """
        return xgb_varimp(self.model)

    def save(self, path):
        """保存模型。

        Parameters
        ----------
        path : str
            模型保存路径
        """
        save_model(self.model, path)

    def load(self, path):
        """加载模型。

        Parameters
        ----------
        path : str
            模型文件路径

        Returns
        -------
        self
        """
        self.model = load_model(path)
        return self

    def calibrate(self, x, y, method='sigmoid', cv='prefit'):
        """模型概率校准。

        Parameters
        ----------
        x : array-like
            校准特征
        y : array-like
            校准标签
        method : str, default 'sigmoid'
            校准方法
        cv : str or int, default 'prefit'
            交叉验证方式

        Returns
        -------
        self
        """
        self.model = CalibratedClassifierCV(self.model, method=method, cv=cv)
        self.model.fit(x, y)
        return self

    def calibration_curve(self, x, y, n_bins=10):
        """获取校准曲线数据。

        Parameters
        ----------
        x : array-like
            特征
        y : array-like
            标签
        n_bins : int, default 10
            分箱数

        Returns
        -------
        tuple
        """
        y_prob = self.predict(x)
        return calibration_curve(y, y_prob, n_bins=n_bins)

    def brier_score(self, x, y):
        """计算Brier分数。

        Parameters
        ----------
        x : array-like
        y : array-like

        Returns
        -------
        float
        """
        y_prob = self.predict(x)
        return brier_score_loss(y, y_prob)

    def roc_auc(self, x, y):
        """计算ROC AUC。

        Parameters
        ----------
        x : array-like
        y : array-like

        Returns
        -------
        float
        """
        y_prob = self.predict(x)
        return roc_auc_score(y, y_prob)


class GradientBoostingModel:
    """
    统一梯度提升模型封装类。

    支持LightGBM和XGBoost两种框架的统一接口。
    通过model_type参数切换框架，其他接口保持一致。

    Parameters
    ----------
    model_type : str
        模型类型，'lgb' 或 'xgb'
    params : dict
        模型参数字典

    Examples
    --------
    >>> model = GradientBoostingModel('lgb', params)
    >>> model.fit(x_train, y_train, x_val, y_val)
    >>> preds = model.predict(x_test)

    # 增量学习（warm-start）
    >>> base_margin = init_model.get_base_margin(x_train)
    >>> new_model = GradientBoostingModel('xgb', params)
    >>> new_model.fit(x_train, y_train, x_val, y_val, init_score=base_margin)
    >>> proba = new_model.predict_with_base_margin(
    ...     x_score, init_model.get_base_margin(x_score))
    """

    def __init__(self, model_type, params):
        """
        初始化统一模型封装类。

        Parameters
        ----------
        model_type : str
            模型类型，'lgb' 或 'xgb'
        params : dict
            模型参数字典

        Raises
        ------
        ValueError
            当model_type不为'lgb'或'xgb'时
        """
        if model_type not in ('lgb', 'xgb'):
            raise ValueError(f"model_type must be 'lgb' or 'xgb', got: {model_type!r}")
        self.model_type = model_type
        self.params = params
        if model_type == 'lgb':
            self._model = LightGBMModel(params)
        else:
            self._model = XGBoostModel(params)

    @staticmethod
    def _sigmoid(z):
        """数值稳定的 Sigmoid（log-odds → 概率）。"""
        z = np.clip(np.asarray(z, dtype=float), -709, 709)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, x, y, valx, valy, init_score=None, **kwargs):
        """训练模型（支持增量学习 warm-start）。

        当传入 ``init_score`` 时，以其作为 log-odds 偏移在新数据上继续训练：
        LightGBM 走 ``init_score``，XGBoost 走 ``base_margin``（两者语义一致，
        本方法统一对外暴露为 ``init_score``）。

        Parameters
        ----------
        x : array-like or pd.DataFrame
            训练集特征
        y : array-like
            训练集标签
        valx : array-like or pd.DataFrame
            验证集特征
        valy : array-like
            验证集标签
        init_score : array-like, optional
            初始 log-odds 偏移（增量学习起点）。一般由基准模型的
            :meth:`get_base_margin` 产生。``None`` 时即普通从头训练。
        **kwargs
            其余参数透传给底层模型（如 lgb 的 ``wgt``、xgb 的
            ``sample_weight`` / ``sample_weight_eval_set``）。

        Returns
        -------
        self

        Notes
        -----
        与既有生产流程一致，偏移仅作用于训练集；验证集未注入偏移，因此早停
        的 eval 指标是在“未加偏移”的空间上评估的。如需严格一致，可后续透传
        lgb 的 ``eval_init_score`` / xgb 的 ``base_margin_eval_set``。
        """
        if self.model_type == 'lgb':
            self._model.fit(x, y, valx, valy, init_score=init_score, **kwargs)
        else:
            self._model.fit(x, y, valx, valy, base_margin=init_score, **kwargs)
        return self

    def get_base_margin(self, x):
        """返回本模型对 ``x`` 的原始 log-odds（base margin / init score）。

        统一兼容两种框架取“未经 sigmoid 的原始分数”：

        - XGBoost: ``predict(x, output_margin=True)``
        - LightGBM: ``predict(x, raw_score=True)``

        该结果可作为下一个增量模型 :meth:`fit` 的 ``init_score``，或喂给
        :meth:`predict_with_base_margin` 做融合预测。

        Parameters
        ----------
        x : array-like or pd.DataFrame
            待计算的特征

        Returns
        -------
        np.ndarray
            一维 log-odds 数组，形状 ``(n_samples,)``

        Raises
        ------
        RuntimeError
            当模型尚未训练（``fit`` 之前）时
        """
        est = self._model.model
        if est is None:
            raise RuntimeError("model is not fitted yet; call fit() first")
        if self.model_type == 'lgb':
            margin = est.predict(x, raw_score=True)
        else:
            margin = est.predict(x, output_margin=True)
        return np.asarray(margin).ravel()

    def predict_with_base_margin(self, x, base_margin, return_prob=True):
        """融合预测：``sigmoid(base_margin + 本模型 raw score)``。

        把一个基准模型的 log-odds（``base_margin``，通常来自
        ``init_model.get_base_margin(x)``）与本（增量）模型自身的 raw score
        在 log-odds 空间相加，再做 sigmoid。这种手动融合是唯一对 lgb 与 xgb
        行为一致的方式——LightGBM 在预测期并不支持注入 init_score。

        Parameters
        ----------
        x : array-like or pd.DataFrame
            待预测的特征
        base_margin : array-like
            基准模型的 log-odds 偏移，形状须与 ``x`` 的样本数一致
        return_prob : bool, default True
            ``True`` 返回概率（sigmoid 后）；``False`` 返回融合后的原始
            log-odds

        Returns
        -------
        np.ndarray
            一维数组，``return_prob=True`` 时取值于 ``[0, 1]``
        """
        combined = np.asarray(base_margin).ravel() + self.get_base_margin(x)
        return self._sigmoid(combined) if return_prob else combined

    def predict(self, x):
        """预测样本的概率。

        Parameters
        ----------
        x : array-like or pd.DataFrame

        Returns
        -------
        np.ndarray
        """
        return self._model.predict(x)

    def get_feature_importance(self, importance_type='gain'):
        """获取特征重要性。

        Returns
        -------
        pd.DataFrame
        """
        return self._model.get_feature_importance(importance_type=importance_type)

    def save(self, path):
        """保存模型。"""
        self._model.save(path)

    def load(self, path):
        """加载模型。"""
        self._model.load(path)
        return self

    def calibrate(self, x, y, method='sigmoid', cv='prefit'):
        """模型概率校准。"""
        self._model.calibrate(x, y, method=method, cv=cv)
        return self

    def brier_score(self, x, y):
        """计算Brier分数。"""
        return self._model.brier_score(x, y)

    def roc_auc(self, x, y):
        """计算ROC AUC。"""
        return self._model.roc_auc(x, y)
