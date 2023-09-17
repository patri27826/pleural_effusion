import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import numpy as np
import shap
    
# 將資料轉換成DataFrame
data = pd.read_csv('data.csv')
data = data.drop(['age','Etiology', 'id', 'pH(PL)', 'Gram Stain',
                 'Acid Fast B(PL)', 'Eosinophil%', 'React Lympho%'], axis=1)

macrophage_mapping = {"Few": 0.25,"Some": 0.5,"Many": 0.75}
status_mapping = {"滲出液": 0, "漏出液": 1}
color_map = {'Pale Yellow': 0, 'Yellow': 1, 'Red': 2, 'Orange': 3}
appearence_map = {'Clear': 0, 'Cloudy': 1, 'Turbid': 2, 'Bloody': 3}
etiology_mapping = {'肺炎': 5,'肝硬化': 1,'心臟衰竭': 2,'體液過多': 3,'腎臟衰竭': 2,
    '肺癌': 4,'攝護腺癌肺部轉移': 4,'膿胸': 5,'胃癌': 4,'淋巴癌': 4,'癌症': 4,
    '肺膿瘍': 5,'乳癌合併肋膜轉移': 4,'肺結核': 5,'敗血症': 5,'黑色素癌': 4,'乳癌': 4,'腎臟衰竭，體液過多': 3,
    '不明原因': 5,
}
data["Macrophage"] = data["Macrophage"].map(macrophage_mapping)
data["Mesothel cell"] = data["Mesothel cell"].map(macrophage_mapping)
data['Fluid Status'] = data['Fluid Status'].map(status_mapping)
data['Appearence'] = data['Appearence'].map(appearence_map)
data['Color'] = data['Color'].map(color_map)
# data['Etiology'] = data['Etiology'].map(etiology_mapping)

# 处理缺失值
columns_to_fill = ['gender', 'Macrophage', 'Mesothel cell',
                   'Glucose (PL)', 'T-Protein (PL)', 'LDH (PL)', 'ADA']  # 要处理的欄位列表
data.loc[:, columns_to_fill] = data.loc[:,columns_to_fill].fillna(data[columns_to_fill].median())

# 拆分資料集
X = data.drop(columns=['Fluid Status'])
y = data['Fluid Status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 定義分類器名稱、實例以及其參數
classifiers = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42), 
    "SVM": SVC(kernel='linear', C=1.0, random_state=42, probability=True),  # 啟用概率預測
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "Logistic Regression": LogisticRegression(max_iter=1000)
}

# 定義評估模型的函數
def evaluate_model(model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    
    # Calculate Specificity
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
    specificity = tn / (tn + fp)
    
    # Calculate AUC-ROC
    y_scores = model.predict_proba(X_val)[:, 1]
    auc_roc = roc_auc_score(y_val, y_scores)
    
    return accuracy, precision, recall, f1, specificity, auc_roc

# 使用 K-Fold 交叉驗證
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# 儲存指標的列表
metrics = ['Precision', 'Recall', 'Specificity', 'F1 Score', 'AUC-ROC']
results = {metric: [] for metric in metrics}

# 進行 K-Fold 交叉驗證
for classifier_name, classifier in classifiers.items():
    print(f"Evaluating {classifier_name}...")
    
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    specificity_scores = []
    f1_scores = []
    auc_roc_scores = []
    
    for train_index, val_index in kf.split(X_train):
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
        
        accuracy_fold, precision_fold, recall_fold, f1_fold, specificity_fold, auc_roc_fold = evaluate_model(classifier, X_train_fold, y_train_fold, X_val_fold, y_val_fold)
        
        accuracy_scores.append(accuracy_fold)
        precision_scores.append(precision_fold)
        recall_scores.append(recall_fold)
        f1_scores.append(f1_fold)
        specificity_scores.append(specificity_fold)
        auc_roc_scores.append(auc_roc_fold)
    
    mean_precision = np.mean(precision_scores)
    mean_recall = np.mean(recall_scores)
    mean_specificity = np.mean(specificity_scores)
    mean_f1 = np.mean(f1_scores)
    mean_auc_roc = np.mean(auc_roc_scores)
    
    std_precision = np.std(precision_scores)
    std_recall = np.std(recall_scores)
    std_specificity = np.std(specificity_scores)
    std_f1 = np.std(f1_scores)
    std_auc_roc = np.std(auc_roc_scores)
    
    results['Precision'].append((mean_precision, std_precision))
    results['Recall'].append((mean_recall, std_recall))
    results['Specificity'].append((mean_specificity, std_specificity))
    results['F1 Score'].append((mean_f1, std_f1))
    results['AUC-ROC'].append((mean_auc_roc, std_auc_roc))
    
# 輸出指標結果
for metric in metrics:
    print(f"{metric} Scores:")
    for i, classifier_name in enumerate(classifiers.keys()):
        mean_score, std_score = results[metric][i]
        print(f"{classifier_name}: {mean_score:.2f} +- {std_score:.2f}")
    print("\n")

# Prepare data for plotting
model_scores = {}

for metric in metrics:
    model_scores[metric] = []
    for metric_mean, _ in results[metric]:
        model_scores[metric].append(metric_mean)

blue_colors = ['#1f77b4', '#154c79', '#5b82a1', '#13446d', '#061724']

# Create a grouped bar plot
fig, ax = plt.subplots(figsize=(10, 3))

bar_width = 0.15
index = np.arange(len(metrics))

for i, classifier_name in enumerate(classifiers.keys()):
    classifier_scores = [model_scores[metric][i] for metric in metrics]
    ax.bar(index + i * bar_width, classifier_scores, bar_width, label=classifier_name, color=blue_colors[i])

ax.set_xlabel('Metrics')
ax.set_ylabel('Score (%)')
ax.set_title('Random Forest Performance Metrics')
ax.set_xticks(index + bar_width * (len(classifiers) - 1) / 2)
ax.set_xticklabels(metrics)

plt.subplots_adjust(right=0.7)  # 調整整個圖的右邊邊緣
ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))  # 放置圖例在左上角旁邊
plt.tight_layout()
plt.show()