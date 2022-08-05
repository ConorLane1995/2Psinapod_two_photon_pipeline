import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib as mpl

def main():
    rt_pre = np.load("/media/vtarka/USB DISK/Lab/2P/active_corrected_traces217.npy")
    rt_post = np.load("/media/vtarka/USB DISK/Lab/2P/active_corrected_traces220.npy")

    bf_pre = np.load("/media/vtarka/USB DISK/Lab/2P/Vid_217_BF_labeling_corrected.npy")
    bf_post = np.load("/media/vtarka/USB DISK/Lab/2P/Vid_220_BF_labeling_corrected.npy")

    rt_post = rt_post[:,:len(rt_pre[0])]

    rt = np.concatenate((rt_pre,rt_post))

    labels = np.zeros(len(rt))
    labels[:len(rt_pre)] = 1

    rt_emb = TSNE(n_components=2,init='random',perplexity=10).fit_transform(rt)

    pre = plt.scatter(rt_emb[:len(rt_pre),0],rt_emb[:len(rt_pre),1],c=bf_pre,label="pre",cmap=mpl.cm.jet_r)
    post = plt.scatter(rt_emb[len(rt_pre):,0],rt_emb[len(rt_pre):,1],c=bf_post,marker='*',s=100,label="post",cmap=mpl.cm.jet_r)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.title("tSNE on Raw Calcium Traces - Responsive Cells Only")
    plt.legend([pre,post],["Pre-Psilocybin","Post-Psilocybin"])
    plt.colorbar()
    plt.show()

    print(len(rt_pre))

if __name__=="__main__":
    main()