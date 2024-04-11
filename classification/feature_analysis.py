import os
from datetime import datetime
import pandas as pd
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler
# from scipy.stats.morestats import stats
# from statannot import add_stat_annotation
from statannotations.Annotator import Annotator

# def Lev(data1, data2):
#     return stats.levene(data1,data2)


# 讀取檔案分類csv
file_name = './clinial_and_image_feature.csv'  # File name
df = pd.read_csv(file_name)
df = df.drop([
    "id",
    "pH(PL)",
    "Gram Stain",
    "Acid Fast B(PL)",
    "Eosinophil%",
    "React Lympho%",
    "Etiology",
], axis=1)
macrophage_mapping = {"Few": 0.25, "Some": 0.5, "Many": 0.75}
status_mapping = {"漏出液": 0, "漏出液": 1}
color_map = {"Yellow": 0, "Pale Yellow": 1, "Orange": 2, "Red": 3}
appearence_map = {"Clear": 0, "Cloudy": 1, "Turbid": 2, "Bloody": 3}
etiology_mapping = {'肺炎': 5, '肝硬化': 1, '心臟衰竭': 2, '體液過多': 3, '腎臟衰竭': 2,
                    '肺癌': 4, '攝護腺癌肺部轉移': 4, '膿胸': 5, '胃癌': 4, '淋巴癌': 4, '癌症': 4,
                    '肺膿瘍': 5, '乳癌合併肋膜轉移': 4, '肺結核': 5, '敗血症': 5, '黑色素癌': 4, '乳癌': 4, '腎臟衰竭，體液過多': 3,
                    '不明原因': 5,
                    }
etiology_mapping = {
    "肺炎": 0,
    "肝硬化": 0,
    "心臟衰竭": 0,
    "體液過多": 0,
    "腎臟衰竭": 0,
    "肺癌": 1,
    "攝護腺癌肺部轉移": 1,
    "膿胸": 0,
    "胃癌": 1,
    "淋巴癌": 1,
    "癌症": 1,
    "肺膿瘍": 0,
    "乳癌合併肋膜轉移": 1,
    "肺結核": 0,
    "敗血症": 0,
    "黑色素癌": 0,
    "乳癌": 1,
    "腎臟衰竭，體液過多": 0,
    "不明原因": 0,
}
df["Macrophage"] = df["Macrophage"].map(macrophage_mapping)
df["Mesothel cell"] = df["Mesothel cell"].map(macrophage_mapping)
df['Fluid Status'] = df['Fluid Status'].map(status_mapping)
df['Appearence'] = df['Appearence'].map(appearence_map)
df['Color'] = df['Color'].map(color_map)

# 補缺失值
columns_to_fill = ['age', 'gender', 'Appearence', 'Color', 'Macrophage', 'Mesothel cell',
                   'Glucose (PL)', 'T-Protein (PL)', 'LDH (PL)', 'ADA']
df.loc[:, columns_to_fill] = df.loc[:, columns_to_fill].fillna(
    df[columns_to_fill].median())


ouptut_dir = f'./results/feature_statistic_fig/{str(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))}'
os.makedirs(ouptut_dir, exist_ok=True)

# # 新增病人欄位
# df["patient"] = np.nan
# ydict = {}
# now = 0
# for i in range(len(df)):
#     temp = df.loc[i, 'filename'].split(",")
#     temp = df.loc[i, 'filename'].split(" ")
#     if temp[0] in ydict.keys():
#         df.loc[i, 'patient'] = ydict[temp[0]]
#     else:
#         ydict[temp[0]] = int(now)
#         now += 1
#         df.loc[i, 'patient'] = ydict[temp[0]]


'''
0:filename, 1:calss, 2-7:morphological, 8-104:GLCM, 104-129:LBP, 130-205:5GL, 206-210:morphological, 211:patient
# 獲取feature column
'''

Feature = list(df.columns)
# Feature = Feature[2:212]  # + Feature[130:212]
# print(Feature)

'''正規化'''
# scaler = StandardScaler()
# scaler.fit(df[Feature])

# X = scaler.transform(df[Feature])
# df[Feature] = pd.DataFrame(X)
# df[Feature].columns = Feature
# df['class'] = df['class'].astype('int')


# X = df[Feature]
# y = df['class'].astype('int')

# print(df['class'][:129])
# print(df['class'][129:248])
# print(df['class'][248:])

df_stat = []
fo = open(f"{ouptut_dir}/data_stats.txt", "w")
for i in Feature:
    if i == 'Maligant':
        continue
    # temp = df[i]
    # u = df[i].mean()
    # std = df[i].std()

    c1 = df[i][df['Maligant'] == 0]
    c2 = df[i][df['Maligant'] == 1]
    fo.write(
        f"Feature: {i}, Non-Malignant: {c1.mean()} +- {c1.std()}, Maligant: {c2.mean()} +- {c2.std()} \n")
    # m1 = c1.mean()
    # std1 = c1.std()
    # m2 = c2.mean()
    # std2 = c2.std()
    # m3 = c3.mean()
    # std3 = c3.std()
    # df_stat.append([i,m1,m2,m3,std1,std2,std3,str(m1)+ " ± " +str(std1),str(m2)+ " ± " +str(std2),str(m3)+ " ± " +str(std3)])
    # p1 = stats.ttest_ind(c1, c2, equal_var=False)
    # p2 = stats.ttest_ind(c1, c3, equal_var=False)
    # p3 = stats.ttest_ind(c2, c3, equal_var=False)

    order = ['Non-Malignant', 'Malignant']
    # x_pos = np.arange(len(order))
    # CTE = [m1, m2, m3]
    # error = [std1, std2, std3]

    df2 = pd.DataFrame(list(zip(c1, c2)), columns=order)
    # print(df2)
    # X軸是mean，長得像I的是error bar，seaborn在沒調整情況下預設方法是errorbar=('ci', 95)，error bar型態你看一下line
    ax = sns.barplot(data=df2, order=order, capsize=.2,
                     palette=['black', 'silver', 'gray'])
    # 參數參考 ref:https://seaborn.pydata.org/generated/seaborn.barplot.html
    # Error的用處簡單的來說就是一個粗略的方式讓你先去看你的資料群體(本例是良性&惡性群)兩群有沒有獨立=有沒有差異，判斷的方式是看兩群的error bar overlapping的程度，越少overlap代表兩群差異越大，比較有可能有顯著差異的那個*符號，但也只是一個視覺上的判斷來做統計推論(statistical inference)，嚴謹一點還是要採用統計檢定(本例以Mann-Whitney U test)來做判斷
    # yt上講解CI的ref: https://youtu.be/NMcogBp1rnA?si=hcVEh_Ng8JvyMK8N (註:我沒有學過error bars，實際是不是這樣用還是要再多方查證，但我已經看了2個影片都是同個講法所以應該十之八九是這樣，但我現在好累ㄌ剩下你自己查88 )
    # ax.set(xlabel='tumor type', ylabel=i, fontsize=16)
    ax.set_ylabel(i, fontsize=15)
    ax.set_xticklabels(order, fontsize=15)
    # 良性腫瘤的群體和惡性腫瘤的群體在不同的Y特徵上是否有顯著差異(有沒有*號)，有的話代表這個Y特徵可以有效區別這兩種腫瘤的群體
    annotator = Annotator(
        ax, [('Non-Malignant', 'Malignant')], data=df2, order=order)
    # 不是特別頂級但可參考的論文ref: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6963514/ Sec 4.的圖片下方那一段，另外Sec 3.3有Mann-Whitney U test的使用時機介紹也可以看看~
    # 添加顯著性標示，star=*，p-value預設0.05
    annotator.configure(test='Mann-Whitney',
                        text_format='simple', loc='inside')
    # annotator._pvalue_format.pvalue_thresholds =  [[0.001, '****'], [0.01, '***'], [0.05, '**'], [0.1, '*'], [1, 'ns']]
    # For “star” text_format: [[1e-4, “****”], [1e-3, “***”], [1e-2, “**”], [0.05, “*”], [1, “ns”]]. ref: https://statannotations.readthedocs.io/en/latest/statannotations.html
    annotator.apply_and_annotate()

    plt.savefig(ouptut_dir + '/' + i + '.png')
    plt.clf()
