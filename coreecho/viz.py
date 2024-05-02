import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from umap import UMAP

class HelperTSNE:
    def __init__(self, X, **kwargs):
        self.tsne = TSNE(**kwargs)
        self.embs = self.tsne.fit_transform(X)
    
    def __call__(self, labels):
        cmap = sns.cubehelix_palette(as_cmap=True, rot=-0.2)
        
        labels = convert_string_labels_for_plotting(labels)
        
        fig = plt.figure()
        ax = fig.add_subplot()
        points = ax.scatter(self.embs[:,0], self.embs[:,1], c=labels, s=20, cmap=cmap)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(points)
        
        return fig

class HelperUMAP:
    def __init__(self, X, **kwargs):
        self.umap_obj = UMAP(**kwargs)
        self.embs = self.umap_obj.fit_transform(X)
    
    def __call__(self, labels):
        cmap = sns.cubehelix_palette(as_cmap=True, rot=-0.2)
        
        labels = convert_string_labels_for_plotting(labels)
        
        fig = plt.figure()
        ax = fig.add_subplot()
        points = ax.scatter(self.embs[:,0], self.embs[:,1], c=labels, s=20, cmap=cmap)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(points)
        
        return fig

def convert_string_labels_for_plotting(labels):
    contain_str = False
    for label in labels:
        if type(label) == str:
            contain_str = True
            break
    if contain_str == True:
        lbl_to_idx = {label: i for i, label in enumerate(list(set(labels)))}
        return [lbl_to_idx[label] for label in labels]
    else:
        return labels