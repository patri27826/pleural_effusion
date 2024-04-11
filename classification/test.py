import array
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.svm import SVC
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
)

from datetime import datetime
import numpy as np
import shap
import os
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
import lightgbm as lgb
import xgboost as xgb
from autogluon.tabular import TabularPredictor
from sklearn.ensemble import AdaBoostClassifier
import scipy.stats as stats


data = pd.read_csv("clinial_feature.csv")
output_dir = (
    "./results/" + str(datetime.now().strftime("%Y-%m-%d-%H-%M-%S")) + "/"
)
shap_output_dir = output_dir + "shap_fig/"
feature_importance_dir = output_dir + "feat_imp/"
statistic_output_dir = output_dir + "feature_statistic/"

os.makedirs(shap_output_dir, exist_ok=True)
os.makedirs(feature_importance_dir, exist_ok=True)
os.makedirs(statistic_output_dir, exist_ok=True)

data = data.drop(
    [
        # "id",
        # "gender",
        # "age",
        "pH(PL)",
        "Gram Stain",
        "Acid Fast B(PL)",
        # "Appearence",
        # "Color",
        # "SP.Gravity",
        # "Protein",
        # "WBC",
        # "RBC",
        # "Neutrophil%",
        # "Lymphocyte%",
        "Eosinophil%",
        "React Lympho%",
        # "Macrophage",
        # "Mesothel cell",
        # "Glucose (PL)",
        # "T-Protein (PL)",
        # "LDH (PL)",
        # "ADA",
        # "Fluid Status",
        "Etiology",
    ],
    axis=1,
)

macrophage_mapping = {"Few": 0.25, "Some": 0.5, "Many": 0.75}
status_mapping = {"滲出液": 0, "漏出液": 1}
color_map = {"colorless": 0, "Pale Yellow": 1,
             "Yellow": 2, "Red": 3, "Orange": 4}
appearence_map = {"Clear": 0, "Cloudy": 1, "Turbid": 2, "Bloody": 3}
# etiology_mapping = {'肺炎': ,'肝硬化': ,'心臟衰竭': 2,'體液過多': 3,'腎臟衰竭': 2,
#     '肺癌': 1,'攝護腺癌肺部轉移': 1,'膿胸': 5,'胃癌': 1,'淋巴癌': 1,'癌症': 1,
#     '肺膿瘍': 5,'乳癌合併肋膜轉移': 1,'肺結核': 5,'敗血症': 5,'黑色素癌': 1,'乳癌': 1,'腎臟衰竭，體液過多': 3,
#     '不明原因': 5,'急性膽囊炎': 5,'甲狀腺癌合併肋膜轉移': 1
# }
# etiology_mapping = {
#     "肺炎": 0,
#     "肝硬化": 0,
#     "心臟衰竭": 0,
#     "體液過多": 0,
#     "腎臟衰竭": 0,
#     "肺癌": 1,
#     "攝護腺癌肺部轉移": 1,
#     "膿胸": 0,
#     "胃癌": 1,
#     "淋巴癌": 1,
#     "癌症": 1,
#     "肺膿瘍": 0,
#     "乳癌合併肋膜轉移": 1,
#     "肺結核": 0,
#     "敗血症": 0,
#     "黑色素癌": 0,
#     "乳癌": 1,
#     "腎臟衰竭，體液過多": 0,
#     "不明原因": 0,
# }
data["Macrophage"] = data["Macrophage"].map(macrophage_mapping)
data["Mesothel cell"] = data["Mesothel cell"].map(macrophage_mapping)
data['Fluid Status'] = data['Fluid Status'].map(status_mapping)
data['Appearence'] = data['Appearence'].map(appearence_map)
data['Color'] = data['Color'].map(color_map)
# data['Etiology'] = data['Etiology'].map(etiology_mapping)

# 处理缺失值
columns_to_fill = ['age', 'gender', 'Appearence', 'Color', 'Macrophage', 'Mesothel cell',
                   'Glucose (PL)', 'T-Protein (PL)', 'LDH (PL)', 'ADA']  # 要处理的欄位列表
data.loc[:, columns_to_fill] = data.loc[:, columns_to_fill].fillna(
    data[columns_to_fill].median())

# 拆分資料集
X = data.drop(columns=["Maligant"])

# 對資料進行標準化
# scaler = StandardScaler()
scaler = MinMaxScaler()
X_scaled = X.copy()
X[X.columns] = scaler.fit_transform(X_scaled[X.columns])
print(X)

y = data["Maligant"]


xgb_params = {
    "objective": "binary:logistic",
    "n_estimators": 100,
    "learning_rate": 0.1,
    "max_depth": 3,
    "min_child_weight": 1,
    "subsample": 1.0,
    "colsample_bytree": 0.8,
    "gamma": 0,
}
lgb_params = {
    "objective": "binary",
    "num_iterations": 150,
    "learning_rate": 0.1,
    "max_depth": 3,
    "min_data_in_leaf": 20,
    "feature_fraction": 1.0,
    "bagging_fraction": 0.8,
    "lambda_l1": 0,
    "lambda_l2": 0,
}

classifiers = {
    "Random Forest": RandomForestClassifier(
        n_estimators=150,
        max_depth=15,
        min_samples_split=2,
        min_samples_leaf=2,
    ),
    # "Decision Tree": DecisionTreeClassifier(
    #     max_depth=15,
    #     min_samples_split=2,
    #     min_samples_leaf=1,
    #     random_state=42,
    # ),
    "Logistic Regression": LogisticRegression(
        penalty="l2",
        C=0.1,
    ),
    "LightGBM": lgb.LGBMClassifier(**lgb_params),
    "XGBoost": xgb.XGBClassifier(**xgb_params),
    "SVM": SVC(
        kernel="rbf",
        C=10,
        probability=True
    ),
    "AdaBoost": AdaBoostClassifier(
        algorithm="SAMME"
    ),
}


def plot_feature_importance(feature_importance, feature_names, model_name, count):
    plt.figure(figsize=(10, 15))
    sorted_indices = np.argsort(feature_importance)[::-1]
    sorted_feature_importance = feature_importance[sorted_indices]
    sorted_feature_names = [feature_names[i] for i in sorted_indices]

    sorted_feature_importance = sorted_feature_importance[:50]
    sorted_feature_names = sorted_feature_names[:50]

    plt.barh(range(len(sorted_feature_importance)),
             sorted_feature_importance, tick_label=sorted_feature_names)
    plt.xlabel('Feature Importance')
    plt.ylabel('Features')
    plt.title(f"{model_name} Feature Importance")
    plt.tight_layout()
    plt.savefig(
        f'{feature_importance_dir}/{model_name.replace(" ", "_")}_{count}_feature_importance.png')
    # plt.show()

# 定義評估模型的函數


def evaluate_model(
    model_name,
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    count,
):

    # start training
    current_model = model
    current_model.fit(X_train, y_train)
    y_pred = current_model.predict(X_val)

    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
    specificity = tn / (tn + fp)

    y_scores = current_model.predict_proba(X_val)[:, 1]
    auc_roc = roc_auc_score(y_val, y_scores)

    if (
        model_name == "Random Forest"
        or model_name == "Decision Tree"
        or model_name == "LightGBM"
    ):
        explainer = shap.TreeExplainer(current_model)
        shap_values = explainer(X_val)
        shap.plots.bar(shap_values[:, :, 1], max_display=15, show=False)
        # shap.summary_plot(shap_values, X_val, feature_names=X.columns[rfecv.support_], show=False)
        plt.tight_layout()
        plt.savefig(
            f'{shap_output_dir}/{model_name.replace(" ", "_")}_{count}_shap_values.png',
            bbox_inches="tight",
        )
    elif model_name == "XGBoost":  # adaboost no explainer
        explainer = shap.TreeExplainer(current_model)
        shap_values = explainer(X_val)
        shap.plots.bar(shap_values, max_display=15, show=False)
        plt.tight_layout()
        plt.savefig(
            f'{shap_output_dir}/{model_name.replace(" ", "_")}_{count}_shap_values.png',
            bbox_inches="tight",
        )

    elif model_name == "Logistic Regression":
        explainer = shap.LinearExplainer(current_model, X_train)
        shap_values = explainer(X_val)
        shap.plots.bar(shap_values, max_display=15, show=False)
        plt.tight_layout()
        plt.savefig(
            f'{shap_output_dir}/{model_name.replace(" ", "_")}_{count}_shap_values.png',
            bbox_inches="tight",
        )

    plt.clf()

    # draw feature importance
    if isinstance(current_model, RandomForestClassifier) or isinstance(current_model, DecisionTreeClassifier) or isinstance(current_model, lgb.LGBMClassifier) or isinstance(current_model, xgb.XGBClassifier):
        plot_feature_importance(
            current_model.feature_importances_, X_val.columns, model_name, count)
    elif isinstance(current_model, LogisticRegression):
        plot_feature_importance(
            np.abs(current_model.coef_[0]), X_val.columns, model_name, count)

    return accuracy, precision, recall, f1, specificity, auc_roc


# 儲存指標的列表
metrics = ["Accuracy", "Precision", "Recall",
           "Specificity", "F1 Score", "AUC-ROC"]
results = {metric: [] for metric in metrics}

n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

sm = SMOTE(random_state=42, k_neighbors=3)

# 進行 K-Fold 交叉驗證

for classifier_name, classifier in classifiers.items():
    print(f"Evaluating {classifier_name}...")

    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    specificity_scores = []
    f1_scores = []
    auc_roc_scores = []

    count = 1
    for train_index, val_index in kf.split(X):
        X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
        y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]

        (
            accuracy_fold,
            precision_fold,
            recall_fold,
            f1_fold,
            specificity_fold,
            auc_roc_fold,
        ) = evaluate_model(
            classifier_name,
            classifier,
            X_train_fold,
            y_train_fold,
            X_val_fold,
            y_val_fold,
            count,
        )

        accuracy_scores.append(accuracy_fold)
        precision_scores.append(precision_fold)
        recall_scores.append(recall_fold)
        f1_scores.append(f1_fold)
        specificity_scores.append(specificity_fold)
        auc_roc_scores.append(auc_roc_fold)

        count = count + 1

    mean_accuracy = np.mean(accuracy_scores)
    mean_precision = np.mean(precision_scores)
    mean_recall = np.mean(recall_scores)
    mean_specificity = np.mean(specificity_scores)
    mean_f1 = np.mean(f1_scores)
    mean_auc_roc = np.mean(auc_roc_scores)

    std_accuracy = np.std(accuracy_scores)
    std_precision = np.std(precision_scores)
    std_recall = np.std(recall_scores)
    std_specificity = np.std(specificity_scores)
    std_f1 = np.std(f1_scores)
    std_auc_roc = np.std(auc_roc_scores)

    results["Accuracy"].append((mean_accuracy, std_accuracy))
    results["Precision"].append((mean_precision, std_precision))
    results["Recall"].append((mean_recall, std_recall))
    results["Specificity"].append((mean_specificity, std_specificity))
    results["F1 Score"].append((mean_f1, std_f1))
    results["AUC-ROC"].append((mean_auc_roc, std_auc_roc))


# 輸出指標結果
for metric in metrics:
    print(f"{metric} Scores:")
    for i, classifier_name in enumerate(classifiers.keys()):
        mean_score, std_score = results[metric][i]
        print(f"{classifier_name}: {mean_score:.2f} +- {std_score:.2f}")
    print("\n")


# # Prepare data for plotting
# model_scores = {}

# for metric in metrics:
#     model_scores[metric] = []
#     for metric_mean, _ in results[metric]:
#         model_scores[metric].append(metric_mean)

# blue_colors = ['#1f77b4', '#154c79', '#5b82a1', '#13446d', '#061724', '#1f77b8']

# # Create a grouped bar plot
# fig, ax = plt.subplots(figsize=(10, 3))

# bar_width = 0.15
# index = np.arange(len(metrics))

# for i, classifier_name in enumerate(classifiers.keys()):
#     classifier_scores = [model_scores[metric][i] for metric in metrics]
#     ax.bar(index + i * bar_width, classifier_scores, bar_width, label=classifier_name, color=blue_colors[i])

# ax.set_xlabel('Metrics')
# ax.set_ylabel('Score (%)')
# ax.set_title('Random Forest Performance Metrics')
# ax.set_xticks(index + bar_width * (len(classifiers) - 1) / 2)
# ax.set_xticklabels(metrics)

# plt.subplots_adjust(right=0.7)  # 調整整個圖的右邊邊緣
# ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))  # 放置圖例在左上角旁邊
# plt.tight_layout()
# plt.show()
