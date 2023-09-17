import pandas as pd
from sklearn.preprocessing import StandardScaler
from visual_func import UMAP, PCA_visual, T_SNE
from sklearn.decomposition import PCA
    
# 將資料轉換成DataFrame
data = pd.read_csv('data.csv')
data = data.drop(['Etiology','id', 'pH(PL)', 'Gram Stain',
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
columns_to_fill = ['gender', 'age', 'Macrophage', 'Mesothel cell',
                   'Glucose (PL)', 'T-Protein (PL)', 'LDH (PL)', 'ADA']  # 要处理的欄位列表
data.loc[:, columns_to_fill] = data.loc[:,columns_to_fill].fillna(data[columns_to_fill].median())

# 將'Etiology'拆分出來作為目標（y）
# y = data['Etiology']
# X = data.drop('Etiology', axis=1)

# 將'Etiology'拆分出來作為目標（y）
# y = data['Fluid Status']
# X = data.drop('Fluid Status', axis=1)

# 標準化特徵
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

