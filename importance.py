import pandas as pd
import numpy as np
import math
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
import shap
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance


def mdi():
    data = pd.read_csv('second_stage1housing after outliers.csv')
    xx = data[['hall', 'metro', 'primary', 'size', 'south', 'floor', 'age', 'room', 'park', 'mall',
               'hospital', 'university', 'plot ratio', 'greening ratio', 'bus', 'railway', 'junior_senior', 'cbd']]
    yy = data[['average']]
    x_scale = StandardScaler().fit(xx.values)
    y_scale = StandardScaler().fit(yy.values)
    x_tolist = x_scale.transform(xx.values)
    y = y_scale.transform(yy.values)
    x = pd.DataFrame(x_tolist)
    x_train, x_test, y_train, y_test = train_test_split(x, y.flatten(), test_size=0.3, random_state=0)
    feat_labels = xx.columns[0:]
    print(feat_labels)
    forest = RandomForestRegressor(max_depth=50, n_estimators=200, random_state=0)
    forest.fit(x_train, y_train)
    importance = forest.feature_importances_
    indices = np.argsort(importance)[::-1]
    print("Feature ranking")
    for f in range(x_train.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importance[indices[f]]))


def permutation():
    data = pd.read_csv('second_stage2housing after outliers.csv')
    xx = data[['hall', 'metro', 'primary', 'size', 'south', 'floor', 'age', 'room', 'park', 'mall',
               'hospital', 'university', 'plot ratio', 'greening ratio', 'bus', 'railway', 'junior_senior', 'cbd']]
    yy = data[['average']]
    x_scale = StandardScaler().fit(xx.values)
    y_scale = StandardScaler().fit(yy.values)
    x_tolist = x_scale.transform(xx.values)
    y = y_scale.transform(yy.values)
    x = pd.DataFrame(x_tolist)
    x_train, x_test, y_train, y_test = train_test_split(x, y.flatten(), test_size=0.3, random_state=0)
    feat_labels = xx.columns[0:]
    print(feat_labels)
    forest = RandomForestRegressor(max_depth=50, n_estimators=200, random_state=0)
    forest.fit(x_train, y_train)
    result = permutation_importance(forest, x_train, y_train, n_repeats=10,
                                    random_state=42, n_jobs=2)
    print(result)
    sorted_idx = result.importances_mean.argsort()
    print(sorted_idx)
    fig, ax = plt.subplots()
    ax.boxplot(result.importances[sorted_idx].T,
               vert=False, labels=feat_labels)
    ax.set_title("Permutation Importances (train set)")
    fig.tight_layout()
    plt.show()
    indices = np.argsort(result)[::-1]


def random_forest_shap():
    data = pd.read_csv('second_stage1housing after outliers.csv')
    xx = data[['size', 'south', 'floor', 'age', 'room', 'hall', 'metro', 'bus', 'railway', 'park', 'mall',
               'hospital', 'university', 'cbd', 'junior_senior', 'primary', 'plot ratio', 'greening ratio']]
    yy = data[['average']]
    x_scale = StandardScaler().fit(xx.values)
    y_scale = StandardScaler().fit(yy.values)
    x_tolist = x_scale.transform(xx.values)
    y_data = y_scale.transform(yy.values)
    x_data = pd.DataFrame(x_tolist, columns=['S', 'O', 'F', 'BA', 'N_R', 'N_LD', 'D_MS', 'BS', 'D_RS', 'D_P', 'D_SM',
                                             'D_H', 'UN', 'D_CBD', 'JSN', 'PSN', 'PR', 'GR'])
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=0)
    # x_real = x_scale.inverse_transform(x_train)
    # x_real = pd.DataFrame(x_real, columns=['S', 'O', 'F', 'BA', 'N_R', 'N_LD', 'D_MS', 'BS', 'D_RS', 'D_P', 'D_SM',
    #                                          'D_H', 'UN', 'D_CBD', 'JSN', 'PSN', 'PR', 'GR'])
    # print(x_real)

    forest = ExtraTreesRegressor(max_depth=35, n_estimators=200, random_state=0)
    model = forest.fit(x_train, y_train)
    explainer = shap.TreeExplainer(model, x_train)
    shap_values = explainer(x_train, check_additivity=False)
    # feat_labels = x_data.columns[0:]
    # shap_values = pd.DataFrame(shap_values.values)
    # print(shap_values)

    # x = pd.DataFrame(x_real)
    # print(x)
    # x.to_csv('1_data.csv')
    # shap_values.to_csv('3.csv')

    # shap.initjs()
    # shap.force_plot(base_value=explainer.expected_value, shap_values=shap_values[1].values,
    #                 features=x_train.iloc[1, :].values, feature_names=feat_labels, matplotlib=True, show=True)
    shap.summary_plot(shap_values,  x_train)
    # shap.plots.waterfall(shap_values)
    # shap.plots.scatter(shap_values[:, "metro"], color=shap_values)# 选择具有最强与年龄交互的功能列
    # shap.plots.scatter(shap_values[:, "cbd"], color=shap_values[:, "age"])# 自定义功能列
    # shap.plots.beeswarm(shap_values, max_display=20)# 按最大绝对值进行排序
    # shap.plots.force(shap_values)
    # shap.plots.bar(shap_values)# 全球功能重要性图，其中每个功能的全球重要性被视为该功能在所有给定样本中的平均绝对值。
    # shap.plots.bar(shap_values[0])


if __name__ == "__main__":
    # mdi()
    #permutation()
    random_forest_shap()






