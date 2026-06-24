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
import lightgbm as lgb
import xgboost as xgb
from Modeling_Tool.Core.utils import load_model, save_model
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score

# ============================================================================
# 工具函数（保持独立）
# ============================================================================

def set_num_leaves(max_depth=5, wgt=1):
    """根据最大深度设置叶子节点数，避免过拟合。

    根据给定的最大深度和权重系数，计算合适的叶子节点数量。
    计算公式：2^max_depth - 2^max_depth * wgt

    Parameters
    ----------
    max_depth : int, optional
        树的最大深度，默认为5
    wgt : float, optional
        权重系数，控制叶子节点数的缩减比例，取值范围0-1，
        默认为1表示完全使用2^max_depth

    Returns
    -------
    int
        计算得到的叶子节点数

    Examples
    --------
    >>> leaves = set_num_leaves(max_depth=5, wgt=0.5)
    >>> print(leaves)  # 16
    """
    return 2 ** max_depth - int(2 ** max_depth * wgt)


# ============================================================================
# LightGBM相关函数（保持独立）
# ============================================================================

def lgb_model(x, y, valx, valy, params_dict, wgt=None, init_score=None):
    """快速训练LightGBM模型。

    使用给定的参数和验证集训练LightGBM分类器，
    支持早停机制。

    Parameters
    ----------
    x : array-like
        训练集特征
    y : array-like
        训练集标签
    valx : array-like
        验证集特征
    valy : array-like
        验证集标签
    params_dict : dict
        LightGBM模型参数字典，必须包含：
        - early_stopping_rounds: 早停轮数
        - eval_metric: 评估指标（也支持'metric'键名）
        其他参数将传递给LGBMClassifier
    wgt : array-like, optional
        样本权重，默认为None

    Returns
    -------
    lgb.LGBMClassifier
        训练好的LightGBM模型

    Examples
    --------
    >>> params = {
    ...     'n_estimators': 100,
    ...     'learning_rate': 0.1,
    ...     'max_depth': 5,
    ...     'early_stopping_rounds': 10,
    ...     'eval_metric': 'auc'
    ... }
    >>> model = lgb_model(x_train, y_train, x_val, y_val, params)
    """

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
    model.fit(
        x, y,
        eval_set=[(valx, valy)],
        eval_metric=params_dict['eval_metric'] if 'eval_metric' in params_dict else params_dict['metric'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=params_dict['early_stopping_rounds'], verbose=False)
        ],
        verbose=False,
        sample_weight=wgt,
        init_score=init_score   # 新增
    )

    return model


def lgb_varimp(model):
    """获取LightGBM模型的特征重要性。

    从训练好的LightGBM模型中提取特征重要性信息，
    包括分裂次数重要性（split）和增益重要性（gain）。

    Parameters
    ----------
    model : lgb.LGBMClassifier
        训练好的LightGBM模型

    Returns
    -------
    pd.DataFrame
        包含以下列的DataFrame：
        - rank: 重要性排名
        - variable: 特征名称
        - imp_gain: 增益重要性
        - imp_split: 分裂次数重要性
        - percentage: 增益占比（百分比）

    Examples
    --------
    >>> model = lgb_model(x_train, y_train, x_val, y_val, params)
    >>> varimp = lgb_varimp(model)
    >>> print(varimp.head())
    """

    varimp = pd.DataFrame({
        "variable": model.booster_.feature_name() if hasattr(model, 'booster_') else model.feature_name(),
        "imp_split": model.booster_.feature_importance(importance_type="split") if hasattr(model, 'booster_') else model.feature_importance(importance_type="split"),
        "imp_gain": model.booster_.feature_importance(importance_type="gain") if hasattr(model, 'booster_') else model.feature_importance(importance_type="gain")
    })

    varimp.loc[:, "percentage"] = varimp["imp_gain"] / varimp["imp_gain"].sum()
    varimp = varimp.sort_values(by="imp_gain", ascending=False).reset_index(drop=True)
    varimp.loc[:, "rank"] = range(1, len(varimp) + 1)
    varimp = varimp[['rank', 'variable', 'imp_gain', 'imp_split', 'percentage']]

    return varimp


def lgbm_quick_train(train_data, validation_data, x, y, params, wgt_col = None, cat_x_train=None):
    """快速训练LightGBM模型（使用DataFrame接口）。

    简化版训练接口，通过列名直接指定特征和标签，
    适合快速迭代实验。

    Parameters
    ----------
    train_data : pd.DataFrame
        训练数据集
    validation_data : pd.DataFrame
        验证数据集
    x : str or list
        特征列名或列名列表
    y : str
        目标变量列名
    wgt_col : str, optional
        样本权重列名，默认为None
    params : dict
        LightGBM模型参数字典，必须包含：
        - early_stopping_rounds: 早停轮数
        - eval_metric: 评估指标
    cat_x_train : None
        保留参数，暂未使用

    Returns
    -------
    lgb.LGBMClassifier
        训练好的LightGBM模型

    Examples
    --------
    >>> model = lgbm_quick_train(
    ...     train_df, val_df,
    ...     x=['feat1', 'feat2', 'feat3'],
    ...     y='target',
    ...     wgt_col='weight',
    ...     params=params
    ... )
    """

    wgt = None
    if wgt_col is not None:
        wgt = train_data[wgt_col].values  # 提取权重列

    lgbm_quick = lgb_model(
        train_data[x], train_data[y],
        validation_data[x], validation_data[y],
        params,
        wgt=wgt  # 传递给底层函数
    )
    
    return lgbm_quick


# ============================================================================
# XGBoost相关函数（保持独立）
# ============================================================================

def xgb_model(x, y, valx, valy, params_dict, sample_weight=None, sample_weight_eval_set=None, base_margin=None):
    """训练XGBoost分类模型。

    使用给定的参数和验证集训练XGBoost分类器，
    支持样本权重和早停机制。

    Parameters
    ----------
    x : array-like
        训练集特征
    y : array-like
        训练集标签
    valx : array-like
        验证集特征
    valy : array-like
        验证集标签
    params_dict : dict
        XGBoost模型参数字典，将直接传递给XGBClassifier
    sample_weight : array-like, optional
        训练集样本权重，默认为None
    sample_weight_eval_set : list, optional
        验证集样本权重列表，默认为None

    Returns
    -------
    xgb.XGBClassifier
        训练好的XGBoost模型

    Examples
    --------
    >>> params = {
    ...     'n_estimators': 100,
    ...     'learning_rate': 0.1,
    ...     'max_depth': 5,
    ...     'early_stopping_rounds': 10
    ... }
    >>> model = xgb_model(x_train, y_train, x_val, y_val, params)
    """
    # model = xgb.XGBClassifier(**params_dict)
    # model.fit(
    #     x, y,
    #     eval_set=[(valx, valy)],
    #     verbose=False,
    #     sample_weight=sample_weight,
    #     sample_weight_eval_set=sample_weight_eval_set
    # )

    model = xgb.XGBClassifier(**params_dict)
    model.fit(
        x, y,
        eval_set=[(valx, valy)],
        verbose=False,
        sample_weight=sample_weight,
        sample_weight_eval_set=sample_weight_eval_set,
        base_margin=base_margin   # 新增
    )

    return model


def xgbm_quick_train(train_data, validation_data, x, y, wgt_col, params,
                     cat_x_train=None, weight_eval_set=False):
    """快速训练XGBoost模型（使用DataFrame接口）。

    简化版训练接口，通过列名直接指定特征和标签，
    支持训练集和验证集的样本权重设置。

    Parameters
    ----------
    train_data : pd.DataFrame
        训练数据集
    validation_data : pd.DataFrame
        验证数据集
    x : str or list
        特征列名或列名列表
    y : str
        目标变量列名
    wgt_col : str, optional
        样本权重列名，默认为None
    params : dict
        XGBoost模型参数字典
    cat_x_train : None
        保留参数，暂未使用
    weight_eval_set : bool, optional
        是否将验证集权重传递给早停评估，默认为False

    Returns
    -------
    xgb.XGBClassifier
        训练好的XGBoost模型

    Examples
    --------
    >>> model = xgbm_quick_train(
    ...     train_df, val_df,
    ...     x=['feat1', 'feat2', 'feat3'],
    ...     y='target',
    ...     wgt_col='weight',
    ...     params=params,
    ...     weight_eval_set=True
    ... )
    """
    sample_weight = None
    sample_weight_eval_set = None

    if wgt_col:
        sample_weight = train_data[wgt_col]

    if weight_eval_set:
        sample_weight_eval_set = [validation_data[wgt_col]]

    xgbm_quick = xgb_model(
        train_data[x], train_data[y],
        validation_data[x], validation_data[y],
        params,
        sample_weight=sample_weight,
        sample_weight_eval_set=sample_weight_eval_set
    )

    return xgbm_quick


def xgb_varimp(model):
    """获取XGBoost模型的特征重要性。

    从训练好的XGBoost模型中提取特征重要性信息，
    包括权重重要性（weight/split）和增益重要性（gain）。

    Parameters
    ----------
    model : xgb.XGBClassifier
        训练好的XGBoost模型

    Returns
    -------
    pd.DataFrame
        包含以下列的DataFrame：
        - rank: 重要性排名
        - variable: 特征名称
        - imp_gain: 增益重要性
        - imp_split: 分裂次数重要性
        - percentage: 增益占比（百分比）

    Examples
    --------
    >>> model = xgb_model(x_train, y_train, x_val, y_val, params)
    >>> varimp = xgb_varimp(model)
    >>> print(varimp.head())
    """
    imp_gain = pd.DataFrame(
        model.get_booster().get_score(importance_type='gain').items(),
        columns=["variable", "imp_gain"]
    )
    imp_weight = pd.DataFrame(
        model.get_booster().get_score(importance_type='weight').items(),
        columns=["variable", "imp_split"]
    )
    fnl_imp = imp_gain.merge(imp_weight, on=["variable"])
    fnl_imp["percentage"] = fnl_imp["imp_gain"] / fnl_imp["imp_gain"].sum()
    fnl_imp = fnl_imp.sort_values(["percentage"], ascending=False).reset_index(drop=True)
    fnl_imp["rank"] = pd.Series([x for x in range(1, fnl_imp.shape[0] + 1)]).astype(int)
    fnl_imp = fnl_imp[['rank', 'variable', 'imp_gain', 'imp_split', 'percentage']]
    return fnl_imp


# ============================================================================
# 类封装：LightGBMModel
# ============================================================================

class LightGBMModel:
    """LightGBM模型封装类。

    提供LightGBM模型的统一训练和评估接口，
    支持特征重要性提取和模型持久化。

    Parameters
    ----------
    params : dict
        LightGBM模型参数字典，必须包含：
        - early_stopping_rounds: 早停轮数
        - eval_metric: 评估指标（也支持'metric'键名）
    model : lgb.LGBMClassifier, optional
        预加载的模型实例，默认为None

    Attributes
    ----------
    model : lgb.LGBMClassifier
        训练好的模型实例
    feature_names_ : list
        特征名称列表

    Examples
    --------
    >>> params = {
    ...     'n_estimators': 100,
    ...     'learning_rate': 0.1,
    ...     'max_depth': 5,
    ...     'early_stopping_rounds': 10,
    ...     'eval_metric': 'auc'
    ... }
    >>> lgb_model = LightGBMModel(params)
    >>> lgb_model.fit(x_train, y_train, x_val, y_val)
    >>> varimp = lgb_model.get_feature_importance()
    >>> pred = lgb_model.predict(x_test)
    """

    def __init__(self, params, model=None):
        """初始化LightGBM模型封装类。

        Parameters
        ----------
        params : dict
            LightGBM模型参数字典
        model : lgb.LGBMClassifier, optional
            预加载的模型实例
        """
        self.params = params
        self.model = model
        self.feature_names_ = None

    def fit(self, x, y, valx, valy, wgt=None, init_score=None):
        """训练LightGBM模型。

        使用训练集和验证集训练模型，支持早停机制。

        Parameters
        ----------
        x : array-like
            训练集特征
        y : array-like
            训练集标签
        valx : array-like
            验证集特征
        valy : array-like
            验证集标签
        wgt : array-like, optional
            样本权重，默认为None

        Returns
        -------
        self
            返回自身实例，支持链式调用

        Examples
        --------
        >>> lgb_model = LightGBMModel(params)
        >>> lgb_model.fit(x_train, y_train, x_val, y_val, wgt=weights)
        """
        # self.model = lgb_model(x, y, valx, valy, self.params, wgt=wgt)
        # # 尝试多种方式获取特征名称（兼容不同版本LightGBM）
        # try:
        #     self.feature_names_ = self.model.booster_.feature_name()
        # except AttributeError:
        #     try:
        #         self.feature_names_ = self.model.feature_name()
        #     except AttributeError:
        #         self.feature_names_ = list(x.columns) if hasattr(x, 'columns') else None

        self.model = lgb_model(x, y, valx, valy, self.params, wgt=wgt, init_score=init_score)
        # 特征名称获取逻辑保持不变
        try:
            self.feature_names_ = self.model.booster_.feature_name()
        except AttributeError:
            try:
                self.feature_names_ = self.model.feature_name()
            except AttributeError:
                self.feature_names_ = list(x.columns) if hasattr(x, 'columns') else None
        return self

    def predict(self, x):
        """使用模型进行预测。

        Parameters
        ----------
        x : array-like
            待预测数据

        Returns
        -------
        array-like
            预测结果
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        return self.model.predict(x)

    def predict_proba(self, x):
        """使用模型进行概率预测。

        Parameters
        ----------
        x : array-like
            待预测数据

        Returns
        -------
        array-like
            预测概率，形状为(n_samples, n_classes)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        return self.model.predict_proba(x)

    def get_feature_importance(self):
        """获取特征重要性。

        Returns
        -------
        pd.DataFrame
            包含以下列的DataFrame：
            - rank: 重要性排名
            - variable: 特征名称
            - imp_gain: 增益重要性
            - imp_split: 分裂次数重要性
            - percentage: 增益占比
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        return lgb_varimp(self.model)

    def get_best_iteration(self):
        """获取最佳迭代次数。

        Returns
        -------
        int
            早停确定的最佳迭代次数
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        return self.model.best_iteration_

    def get_learning_curve(self):
        """获取学习曲线数据。

        返回训练过程中的评估指标变化。

        Returns
        -------
        dict
            包含验证集评估指标历史的字典
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        return self.model.evals_result_

    def save_model(self, filepath):
        """保存模型到文件。

        Parameters
        ----------
        filepath : str
            模型保存路径
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        save_model(self.model, filepath)

    def load_model(self, filepath):
        """从文件加载模型。

        支持pickle格式和LightGBM原生格式。
        """
        
        model = load_model(filepath)
        self.model = model
        self.model._Booster = model.booster_
        self.feature_names_ = model.booster_.feature_name() if hasattr(model, 'booster_') else model.feature_name()
        return self


# ============================================================================
# 类封装：XGBoostModel
# ============================================================================

class XGBoostModel:
    """XGBoost模型封装类。

    提供XGBoost模型的统一训练和评估接口，
    支持特征重要性提取和模型持久化。

    Parameters
    ----------
    params : dict
        XGBoost模型参数字典
    model : xgb.XGBClassifier, optional
        预加载的模型实例，默认为None

    Attributes
    ----------
    model : xgb.XGBClassifier
        训练好的模型实例

    Examples
    --------
    >>> params = {
    ...     'n_estimators': 100,
    ...     'learning_rate': 0.1,
    ...     'max_depth': 5,
    ...     'early_stopping_rounds': 10
    ... }
    >>> xgb_model = XGBoostModel(params)
    >>> xgb_model.fit(x_train, y_train, x_val, y_val)
    >>> varimp = xgb_model.get_feature_importance()
    >>> pred = xgb_model.predict(x_test)
    """

    def __init__(self, params, model=None):
        """初始化XGBoost模型封装类。

        Parameters
        ----------
        params : dict
            XGBoost模型参数字典
        model : xgb.XGBClassifier, optional
            预加载的模型实例
        """
        self.params = params
        self.model = model

    def fit(self, x, y, valx, valy, sample_weight=None, sample_weight_eval_set=None, base_margin=None):
        """训练XGBoost模型。

        使用训练集和验证集训练模型，支持样本权重。

        Parameters
        ----------
        x : array-like
            训练集特征
        y : array-like
            训练集标签
        valx : array-like
            验证集特征
        valy : array-like
            验证集标签
        sample_weight : array-like, optional
            训练集样本权重
        sample_weight_eval_set : list, optional
            验证集样本权重列表

        Returns
        -------
        self
            返回自身实例，支持链式调用

        Examples
        --------
        >>> xgb_model = XGBoostModel(params)
        >>> xgb_model.fit(x_train, y_train, x_val, y_val,
        ...               sample_weight=weights)
        """
        # self.model = xgb_model(
        #     x, y, valx, valy,
        #     self.params,
        #     sample_weight=sample_weight,
        #     sample_weight_eval_set=sample_weight_eval_set
        # )

        self.model = xgb_model(
            x, y, valx, valy,
            self.params,
            sample_weight=sample_weight,
            sample_weight_eval_set=sample_weight_eval_set,
            base_margin=base_margin
        )

        return self

    def predict(self, x):
        """使用模型进行预测。

        Parameters
        ----------
        x : array-like
            待预测数据

        Returns
        -------
        array-like
            预测结果
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        return self.model.predict(x)

    def predict_proba(self, x):
        """使用模型进行概率预测。

        Parameters
        ----------
        x : array-like
            待预测数据

        Returns
        -------
        array-like
            预测概率，形状为(n_samples, n_classes)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        return self.model.predict_proba(x)

    def get_feature_importance(self):
        """获取特征重要性。

        Returns
        -------
        pd.DataFrame
            包含以下列的DataFrame：
            - rank: 重要性排名
            - variable: 特征名称
            - imp_gain: 增益重要性
            - imp_split: 分裂次数重要性
            - percentage: 增益占比
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        return xgb_varimp(self.model)

    def get_best_iteration(self):
        """获取最佳迭代次数。

        Returns
        -------
        int
            早停确定的最佳迭代次数
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        return self.model.best_iteration

    def save_model(self, filepath):
        """保存模型到文件。

        Parameters
        ----------
        filepath : str
            模型保存路径
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        save_model(self.model, filepath)

    def load_model(self, filepath):
        """从文件加载模型。

        支持pickle格式和XGB原生格式。
        """
        
        model = load_model(filepath)
        self.model = model
        self.feature_names_ = model.feature_names_in_.tolist()
        return self
    
    @staticmethod
    def get_base_margin_from_model(model, X, objective='binary:logistic'):
        """从已训练的 XGBoost 模型获取原始输出（log-odds / margin）"""
        if hasattr(model, 'get_booster'):
            booster = model.get_booster()
        else:
            booster = model
        # 输出原始分数（未经过 sigmoid/softmax）
        base_margin = booster.predict(xgb.DMatrix(X), output_margin=True)
        if objective in ['binary:logistic', 'reg:logistic']:
            return base_margin.flatten()
        return base_margin

    @staticmethod
    def predict_with_base_margin(old_model, new_model, X, objective='binary:logistic', return_prob=True):
        """
        使用旧模型 + 新模型（增量模型）进行融合预测。
        旧模型提供 base_margin，新模型在 base_margin 上预测残差。
        """
        # 1. 获取旧模型的原始输出（作为 base_margin）
        if hasattr(old_model, 'get_booster'):
            old_booster = old_model.get_booster()
        else:
            old_booster = old_model
            
        base = old_booster.predict(xgb.DMatrix(X), output_margin=True)

        # 2. 创建 DMatrix 并设置 base_margin
        dmat = xgb.DMatrix(X, base_margin=base)
        
        #### 获取新模型的原始输出（作为 base_margin）####
        if hasattr(new_model, 'get_booster'):
            new_booster = new_model.get_booster()
        else:
            new_booster = new_model

        # 3. 新模型预测（会自动使用 base_margin + 树的输出）
        #   对于 binary:logistic，predict 返回的是概率
        pred = new_booster.predict(dmat)   # 这里不要传任何额外参数

        if return_prob:
            return pred
        else:
            # 如果需要返回原始分数（log-odds），可以设置输出原始分数
            return new_booster.predict(dmat, output_margin=True)

# ============================================================================
# 统一封装类：GradientBoostingModel
# ============================================================================

class GradientBoostingModel:
    """梯度提升模型统一封装类。

    支持在LightGBM和XGBoost之间切换，
    提供统一的训练和评估接口。
    支持概率校准（基于CalibratedClassifierCV）。

    Parameters
    ----------
    model_type : str
        模型类型，可选值：
        - 'lgb' 或 'lightgbm': 使用LightGBM
        - 'xgb' 或 'xgboost': 使用XGBoost
    params : dict
        模型参数字典
    model : estimator, optional
        预加载的模型实例，默认为None

    Attributes
    ----------
    model_type : str
        当前使用的模型类型
    params : dict
        模型参数字典
    model_instance : LightGBMModel or XGBoostModel
        底层的模型实例
    calibrated_model : CalibratedClassifierCV or None
        校准后的模型实例

    Examples
    --------
    >>> # 使用LightGBM
    >>> params = {'n_estimators': 100, 'learning_rate': 0.1}
    >>> model = GradientBoostingModel('lgb', params)
    >>> model.fit(x_train, y_train, x_val, y_val)
    >>> varimp = model.get_feature_importance()

    >>> # 切换为XGBoost
    >>> params = {'n_estimators': 100, 'learning_rate': 0.1}
    >>> model = GradientBoostingModel('xgb', params)
    >>> model.fit(x_train, y_train, x_val, y_val)

    >>> # 概率校准
    >>> model.fit_calibrated_model(X_cal, y_cal, method='sigmoid')
    >>> proba_cal = model.predict_proba(x_test, calibrated_model=True)
    """

    SUPPORTED_MODELS = ['lgb', 'lightgbm', 'xgb', 'xgboost']

    def __init__(self, model_type, params, model=None):
        """初始化梯度提升模型统一封装类。

        Parameters
        ----------
        model_type : str
            模型类型，可选值：'lgb'、'lightgbm'、'xgb'、'xgboost'
        params : dict
            模型参数字典
        model : estimator, optional
            预加载的模型实例
        """
        if model_type.lower() not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model type: {model_type}. "
                f"Supported types: {self.SUPPORTED_MODELS}"
            )

        self.model_type = model_type.lower()
        self.params = params
        self.model_instance = None
        self.calibrated_model = None   # 新增：存储校准器
        self._init_model(model)

    def _init_model(self, model):
        """初始化底层模型实例。

        Parameters
        ----------
        model : estimator, optional
            预加载的模型实例
        """
        if self.model_type in ['lgb', 'lightgbm']:
            self.model_instance = LightGBMModel(self.params, model=model)
        else:
            self.model_instance = XGBoostModel(self.params, model=model)

    def fit(self, x, y, valx, valy, init_score=None, **kwargs):
        """训练模型。(支持 init_score / base_margin）

        Parameters
        ----------
        x : array-like
            训练集特征
        y : array-like
            训练集标签
        valx : array-like
            验证集特征
        valy : array-like
            验证集标签
        **kwargs : dict
            其他参数，如sample_weight、sample_weight_eval_set等

        Returns
        -------
        self
            返回自身实例，支持链式调用

        Examples
        --------
        >>> model = GradientBoostingModel('lgb', params)
        >>> model.fit(x_train, y_train, x_val, y_val)
        """
        # self.model_instance.fit(x, y, valx, valy, **kwargs)

        if self.model_type in ['lgb', 'lightgbm']:
            self.model_instance.fit(x, y, valx, valy, init_score=init_score, **kwargs)
        else:  # xgb
            self.model_instance.fit(x, y, valx, valy, base_margin=init_score, **kwargs)

        return self

    def fit_calibrated_model(self, X_cal, y_cal, method='sigmoid', cv = 'prefit', **kwargs):
        """使用独立的校准集训练概率校准器。

        基于已训练好的底层模型，训练一个CalibratedClassifierCV校准器。

        Parameters
        ----------
        X_cal : array-like
            校准集特征（不能与训练集重叠）
        y_cal : array-like
            校准集标签
        method : str, optional
            校准方法，可选 'sigmoid' (Platt Scaling) 或 'isotonic'，默认 'sigmoid'
        **kwargs : dict
            额外传递给CalibratedClassifierCV的参数

        Returns
        -------
        self
            返回自身实例
        """
        if self.model_instance.model is None:
            raise ValueError("Base model not trained. Call fit() first.")

        import sklearn
        from distutils.version import LooseVersion

        # 检查 sklearn 版本并选择正确的参数名
        if LooseVersion(sklearn.__version__) >= LooseVersion('1.2'):
            calibrated_clf = CalibratedClassifierCV(
                estimator=self.model_instance.model,
                method=method,
                cv=cv,
                **kwargs
            )
        else:
            # 旧版本使用 base_estimator
            calibrated_clf = CalibratedClassifierCV(
                base_estimator=self.model_instance.model,
                method=method,
                cv=cv,
                **kwargs
            )

        calibrated_clf.fit(X_cal, y_cal)
        self.calibrated_model = calibrated_clf
        return self

    def predict(self, x, calibrated_model=False):
        """使用模型进行预测。

        Parameters
        ----------
        x : array-like
            待预测数据
        calibrated_model : bool, optional
            是否使用校准后的模型进行预测，默认 False

        Returns
        -------
        array-like
            预测结果（类别标签）
        """
        if calibrated_model and self.calibrated_model is not None:
            return self.calibrated_model.predict(x)
        else:
            return self.model_instance.predict(x)

    def predict_proba(self, x, calibrated_model=False):
        """使用模型进行概率预测。

        Parameters
        ----------
        x : array-like
            待预测数据
        calibrated_model : bool, optional
            是否使用校准后的模型进行概率预测，默认 False

        Returns
        -------
        array-like
            预测概率，形状为 (n_samples, n_classes)
        """
        if calibrated_model and self.calibrated_model is not None:
            return self.calibrated_model.predict_proba(x)
        else:
            return self.model_instance.predict_proba(x)

    def get_feature_importance(self):
        """获取特征重要性。

        Returns
        -------
        pd.DataFrame
            特征重要性DataFrame
        """
        return self.model_instance.get_feature_importance()

    def get_best_iteration(self):
        """获取最佳迭代次数。

        Returns
        -------
        int
            早停确定的最佳迭代次数
        """
        return self.model_instance.get_best_iteration()

    def save_model(self, filepath):
        """保存模型到文件。

        Parameters
        ----------
        filepath : str
            模型保存路径
        """
        self.model_instance.save_model(filepath)

    def load_model(self, filepath):
        """从文件加载模型。

        Parameters
        ----------
        filepath : str
            模型文件路径
        """
        self.model_instance.load_model(filepath)
        return self

    def switch_model(self, new_model_type):
        """切换模型类型。

        创建新的底层模型实例，保留参数配置。

        Parameters
        ----------
        new_model_type : str
            新的模型类型，可选值：'lgb'、'lightgbm'、'xgb'、'xgboost'

        Examples
        --------
        >>> model = GradientBoostingModel('lgb', params)
        >>> model.fit(x_train, y_train, x_val, y_val)
        >>> # 切换为XGBoost，使用相同参数
        >>> model.switch_model('xgb')
        >>> model.fit(x_train, y_train, x_val, y_val)
        """
        if new_model_type.lower() not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model type: {new_model_type}. "
                f"Supported types: {self.SUPPORTED_MODELS}"
            )
        self.model_type = new_model_type.lower()
        self._init_model(None)
        self.calibrated_model = None   # 切换模型后重置校准器

# class GradientBoostingModel:
#     """梯度提升模型统一封装类。

#     支持在LightGBM和XGBoost之间切换，
#     提供统一的训练和评估接口。

#     Parameters
#     ----------
#     model_type : str
#         模型类型，可选值：
#         - 'lgb' 或 'lightgbm': 使用LightGBM
#         - 'xgb' 或 'xgboost': 使用XGBoost
#     params : dict
#         模型参数字典
#     model : estimator, optional
#         预加载的模型实例，默认为None

#     Attributes
#     ----------
#     model_type : str
#         当前使用的模型类型
#     params : dict
#         模型参数字典
#     model : estimator
#         底层的模型实例（LightGBMModel或XGBoostModel）

#     Examples
#     --------
#     >>> # 使用LightGBM
#     >>> params = {'n_estimators': 100, 'learning_rate': 0.1}
#     >>> model = GradientBoostingModel('lgb', params)
#     >>> model.fit(x_train, y_train, x_val, y_val)
#     >>> varimp = model.get_feature_importance()

#     >>> # 切换为XGBoost
#     >>> params = {'n_estimators': 100, 'learning_rate': 0.1}
#     >>> model = GradientBoostingModel('xgb', params)
#     >>> model.fit(x_train, y_train, x_val, y_val)
#     >>> varimp = model.get_feature_importance()
#     """

#     SUPPORTED_MODELS = ['lgb', 'lightgbm', 'xgb', 'xgboost']

#     def __init__(self, model_type, params, model=None):
#         """初始化梯度提升模型统一封装类。

#         Parameters
#         ----------
#         model_type : str
#             模型类型，可选值：'lgb'、'lightgbm'、'xgb'、'xgboost'
#         params : dict
#             模型参数字典
#         model : estimator, optional
#             预加载的模型实例
#         """
#         if model_type.lower() not in self.SUPPORTED_MODELS:
#             raise ValueError(
#                 f"Unsupported model type: {model_type}. "
#                 f"Supported types: {self.SUPPORTED_MODELS}"
#             )

#         self.model_type = model_type.lower()
#         self.params = params
#         self.model_instance = None
#         self._init_model(model)

#     def _init_model(self, model):
#         """初始化底层模型实例。

#         Parameters
#         ----------
#         model : estimator, optional
#             预加载的模型实例
#         """
#         if self.model_type in ['lgb', 'lightgbm']:
#             self.model_instance = LightGBMModel(self.params, model=model)
#         else:
#             self.model_instance = XGBoostModel(self.params, model=model)

#     def fit(self, x, y, valx, valy, **kwargs):
#         """训练模型。

#         Parameters
#         ----------
#         x : array-like
#             训练集特征
#         y : array-like
#             训练集标签
#         valx : array-like
#             验证集特征
#         valy : array-like
#             验证集标签
#         **kwargs : dict
#             其他参数，如sample_weight、sample_weight_eval_set等

#         Returns
#         -------
#         self
#             返回自身实例，支持链式调用

#         Examples
#         --------
#         >>> model = GradientBoostingModel('lgb', params)
#         >>> model.fit(x_train, y_train, x_val, y_val)
#         """
#         self.model_instance.fit(x, y, valx, valy, **kwargs)
#         return self
    
#     def predict(self, x):
#         """使用模型进行预测。

#         Parameters
#         ----------
#         x : array-like
#             待预测数据

#         Returns
#         -------
#         array-like
#             预测结果
#         """
#         return self.model_instance.predict(x)

#     def predict_proba(self, x):
#         """使用模型进行概率预测。

#         Parameters
#         ----------
#         x : array-like
#             待预测数据

#         Returns
#         -------
#         array-like
#             预测概率
#         """
#         return self.model_instance.predict_proba(x)

#     def get_feature_importance(self):
#         """获取特征重要性。

#         Returns
#         -------
#         pd.DataFrame
#             特征重要性DataFrame
#         """
#         return self.model_instance.get_feature_importance()

#     def get_best_iteration(self):
#         """获取最佳迭代次数。

#         Returns
#         -------
#         int
#             早停确定的最佳迭代次数
#         """
#         return self.model_instance.get_best_iteration()

#     def save_model(self, filepath):
#         """保存模型到文件。

#         Parameters
#         ----------
#         filepath : str
#             模型保存路径
#         """
#         self.model_instance.save_model(filepath)

#     def load_model(self, filepath):
#         """从文件加载模型。

#         Parameters
#         ----------
#         filepath : str
#             模型文件路径
#         """
#         self.model_instance.load_model(filepath)
#         return self

#     def switch_model(self, new_model_type):
#         """切换模型类型。

#         创建新的底层模型实例，保留参数配置。

#         Parameters
#         ----------
#         new_model_type : str
#             新的模型类型，可选值：'lgb'、'lightgbm'、'xgb'、'xgboost'

#         Examples
#         --------
#         >>> model = GradientBoostingModel('lgb', params)
#         >>> model.fit(x_train, y_train, x_val, y_val)
#         >>> # 切换为XGBoost，使用相同参数
#         >>> model.switch_model('xgb')
#         >>> model.fit(x_train, y_train, x_val, y_val)
#         """
#         if new_model_type.lower() not in self.SUPPORTED_MODELS:
#             raise ValueError(
#                 f"Unsupported model type: {new_model_type}. "
#                 f"Supported types: {self.SUPPORTED_MODELS}"
#             )
#         self.model_type = new_model_type.lower()
#         self._init_model(None)
