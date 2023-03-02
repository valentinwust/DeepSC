import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

##############################
##### Utils
##############################

def evaluate_latent_grouped(adata, latent, ctypekey):
    """ Group latent by ctypekey and get means, stds.
    """
    means = np.asarray([latent[adata.obs["celltype"]==ctype].mean(axis=0) for ctype in np.unique(adata.obs["celltype"])]).T
    stds = np.asarray([latent[adata.obs["celltype"]==ctype].std(axis=0) for ctype in np.unique(adata.obs["celltype"])]).T
    ctypenames = np.repeat(np.unique(adata.obs["celltype"])[None], latent.shape[1], axis=0)
    latentdims = np.repeat(np.arange(latent.shape[1])[:,None], np.unique(adata.obs["celltype"]).shape[0], axis=1).astype(str)
    return means, stds, ctypenames, latentdims

##############################
##### Plots
##############################

def plot_latent_bygroup(adata, latent, ctypekey="celltype"):
    """ Plot mean and std of latent representation by cell group.
    """
    means, stds, ctypenames, latentdims = evaluate_latent_grouped(adata, latent, ctypekey)

    clustered = sns.clustermap(means, col_cluster=True)
    plt.close()

    latentind = clustered.dendrogram_row.reordered_ind
    groupind = clustered.dendrogram_col.reordered_ind

    fig, ax = plt.subplots(1,2,figsize=(20,10))

    sns.heatmap(means[latentind][:, groupind], ax=ax[0], center=0, square=False, robust=True)
    ax[0].set_xticklabels(ctypenames[0,groupind], rotation=45, ha="right")
    ax[0].set_yticklabels(latentdims[latentind,0], rotation=0, ha="right")
    ax[0].set_title("Means", size=20)
    
    sns.heatmap(stds[latentind][:, groupind], ax=ax[1], vmin=0, square=False, robust=False)
    ax[1].set_xticklabels(ctypenames[0,groupind], rotation=45, ha="right")
    ax[1].set_yticklabels(latentdims[latentind,0], rotation=0, ha="right")
    ax[1].set_title("Standard Deviations", size=20)

def plot_umap_latent(adata, umapx, umapy, latent, ctypekey="celltype"):
    """ Plot UMAP with celltypes, and all latent dimensions.
    """
    means, stds, ctypenames, latentdims = evaluate_latent_grouped(adata, latent, ctypekey)
    clustered = sns.clustermap(means, col_cluster=True)
    plt.close()
    latentind = clustered.dendrogram_row.reordered_ind
    groupind = clustered.dendrogram_col.reordered_ind
    
    xN, yN = 3, 4+int(np.ceil(latent.shape[1]/3))
    fig = plt.figure(layout=None, facecolor='0.9', figsize=(xN*3, yN*3))
    gs = fig.add_gridspec(nrows=yN, ncols=xN, left=0, right=1,
                          hspace=0.15, wspace=0.05)
    
    axctype = fig.add_subplot(gs[:3, :])
    sns.scatterplot(x=umapx, y=umapy, hue=adata.obs[ctypekey], s=5, ax=axctype)
    
    for i in range(latent.shape[1]):
        ax = fig.add_subplot(gs[4+i//3, i%3])
        sns.scatterplot(x=umapx, y=umapy, hue=latent[:,latentind[i]], cmap="viridis", ax=ax, s=2, legend=False)
        ax.tick_params(left = False, right = False, labelleft = False, labelbottom = False, bottom = False)
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.set_title(str(latentind[i]))

