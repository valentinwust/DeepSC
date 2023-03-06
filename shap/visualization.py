import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

##############################
##### Plot SHAP explanation for single gene and latent dimension
##############################

def plot_latentshap_singlegene(shap_values, eXadata, geneindex, ctypekey, latentdim, geneskey="gene_name", latent=None, layerkey=None, latentkey="X_latent"):
    """ Plot SHAP explanation for single gene and latent dimension.
    """
    fig, ax = plt.subplots(2,2,figsize=(15,15))
    
    exprdata = eXadata.X[:,geneindex] if layerkey is None else eXadata.layers[layerkey][:,geneindex]
    shapdata = shap_values[latentdim][:,geneindex]
    ctype = eXadata.obs[ctypekey]
    
    sns.stripplot(x=ctype,
                  y=shapdata,
                  ax=ax[1,0])
    sns.stripplot(x=exprdata,
                  y=ctype,
                  ax=ax[0,1])
    
    sns.scatterplot(x=exprdata,
                    y=shapdata,
                    hue=ctype,
                    ax=ax[1,1])
    
    r = np.corrcoef(exprdata, shapdata)[1,0]
    
    #left, bottom, width, height = [0.1, 0.6, 0.2, 0.2]
    #axinset = fig.add_axes([left, bottom, width, height])
    axinset = inset_axes(ax[0,0], "100%", "60%", loc="center left", borderpad=0)
    #ax[0,0].get_xaxis().set_visible(False)
    #ax[0,0].get_yaxis().set_visible(False)
    #fig.delaxes(ax[0,0])
    ax[0,0].axis('off')
    
    sns.stripplot(x=ctype,
                  y= eXadata.obsm[latentkey][:,latentdim] if latent is None else latent[:,latentdim],
                  ax=axinset)
    
    ax[1,1].set_yticklabels([])
    ax[0,1].set_xticklabels([])
    #axinset.set_xticklabels([])
    ax[1,1].set_yticks([])
    ax[0,1].set_xticks([])
    axinset.set_xticklabels(axinset.get_xticklabels(), rotation=45, ha="right")
    ax[1,0].set_xticklabels(ax[1,0].get_xticklabels(), rotation=45, ha="right")
    ax[0,1].yaxis.set_label_position("right")
    ax[0,1].yaxis.tick_right()
    
    ax[1,0].set_ylabel("SHAP Value", size=15)
    ax[1,1].set_xlabel("Gene Expression", size=15)
    ax[1,0].set_xlabel("Cell Group", size=12)
    ax[0,1].set_ylabel("Cell Group", size=12)
    axinset.set_xlabel("")
    axinset.set_ylabel(f"Latent Dimension {latentdim}", size=12)
    
    ax[1,1].set_ylim(ax[1,0].get_ylim())
    ax[1,1].set_xlim(ax[0,1].get_xlim())
    
    axinset.set_title(f"Expression x SHAP:\nr={np.round(r,2)}, r²={np.round(r**2,2)}\n\n")
    
    fig.suptitle(f"Explaining Latent Dimension {latentdim} with Gene {eXadata.var[geneskey].iloc[geneindex]}", size=20, y=.92)
    
    plt.subplots_adjust(wspace=.025, hspace=.025)




