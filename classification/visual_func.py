from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import umap
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd

def UMAP(data):
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(data)

    # Plot the UMAP results
    plt.scatter(embedding[:, 0], embedding[:, 1], c=data['Etiology'], cmap='viridis', s=5)
    plt.colorbar()
    plt.title("UMAP Visualization of Data")
    plt.show()
    
def PCA_visual(x, y):
    # 使用PCA進行降維，這裡設定維度為2
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(x)

    # 將PCA降維後的資料和'Etiology'合併成一個DataFrame
    pca_df = pd.DataFrame(data=X_pca, columns=['PCA1', 'PCA2'])
    pca_df['Etiology'] = y

    # 可以使用散點圖來顯示PCA降維後的資料分布
    plt.figure(figsize=(10, 8))
    for etiology, group in pca_df.groupby('Etiology'):
        plt.scatter(group['PCA1'], group['PCA2'], label=etiology)

    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.title('PCA Visualization of Data')
    plt.legend()
    plt.show()
    
def T_SNE(data):
    tsne = TSNE(n_components=2, random_state=42)
    data_tsne = tsne.fit_transform(data)

    # 绘制t-SNE结果的散点图
    plt.scatter(data_tsne[:, 0], data_tsne[:, 1])
    plt.title('t-SNE Visualization')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.show()