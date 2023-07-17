import inspect
from gray_box_clf.supervised_models import DeepClassification, classifiers_competition, ensemble_targets
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter as sgf
import warnings
import matplotlib.pyplot as plt

warnings.simplefilter("ignore")

df = pd.read_csv("C:\\Users\\George\\PycharmProjects\\Klebsiella\\Tables\\KlbPn_Mean_newVS_LOGO.csv")

X = df.iloc[:, 11:].values

X_der = sgf(X, window_length=13, polyorder=3, deriv=2, mode="nearest")

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

classifiers = [[{'cls': SVC(probability=True), 'name': 'lSVM'}, {'parametrs': {}}],
               [{'cls': SVC(probability=True, kernel="poly", degree=2), 'name': 'qSVM'}, {'parametrs': {}}],
               [{'cls': SVC(probability=True, kernel="poly", degree=3), 'name': 'cSVM'}, {'parametrs': {}}],
               [{'cls': RandomForestClassifier(n_estimators=1000), 'name': 'Random Forest'}, {'parametrs': {}}],
               [{'cls': XGBClassifier(n_estimators=1000), 'name': 'XGBoost'}, {'parametrs': {}}]]

anty = ["Amoxicillin"]
for i in anty:
    print("__________________________________________" + i + "__________________________________________")
    y = df[i].values
    y = np.reshape(y, (y.size, 1))

    df_temp = np.concatenate((X_der.copy(), y), axis=1)
    df_temp = pd.DataFrame(df_temp).dropna()
    X_temp = df_temp.iloc[:, :-1].values
    y_temp = df_temp.iloc[:, -1].values.astype(int)
    model = DeepClassification(X_temp, y_temp, classifier=classifiers[4])
    model.grid_search(sacrifice_rate=0.0, n_features_list=[100, 200, 300, "all"], feature_selection_tech="KL&chi2")
    model.k_folds(shuffle=True, random_state=10, confidence_interval=False)
    model.show_roc(roc_save_location="C:\\Users\\George\\PycharmProjects\\Klebsiella\\" + i + "_Roc_Curve_XGB")
    plt.show()
    print(model.optimal_cut_point_on_roc(delta_max=0.3, plot_point_on_ROC=True))
    plt.show()
    print(model.class_report_)
    print(model.best_parameters_)
    print(model.confusion_mat_)
    print("-----------------------------------------------------------------------------------------------")
    print("-----------------------------------------------------------------------------------------------")
