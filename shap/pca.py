import anndata
import numpy as np
import scanpy as sc
import shap

##############################
##### PCA with SHAP
##############################

class shapPCA():
    """ Perform PCA on counts, and use linear SHAP explainer to explain the output for specific genes.
    """
    
    def __init__(self, counts, target_sum=1e4, n_comps=50):
        self.adata = anndata.AnnData(np.asarray(counts).copy())
        self.adata.layers["raw_counts"] = counts.copy()
        
        self.prepare_pca_shap(target_sum=target_sum)
    
    def prepare_pca_shap(self, target_sum=1e4, n_comps=50):
        self.adata.layers["pcainput"] = self.adata.layers["raw_counts"].copy()
        sc.pp.normalize_total(self.adata, target_sum=target_sum, layer="pcainput")
        sc.pp.log1p(self.adata, layer="pcainput")
        sc.pp.scale(self.adata, max_value=10, layer="pcainput")
        self.adata.X = self.adata.layers["pcainput"].copy()
        
        sc.tl.pca(self.adata, svd_solver='arpack')
        self.adata.layers["pcaoutput"] = np.matmul(self.adata.obsm["X_pca"],self.adata.varm["PCs"].T)
        
        self.coeff = np.matmul(self.adata.varm["PCs"], self.adata.varm["PCs"].T)
    
    def explain_genes(self, geneindices, explain_indices=None):
        """ Explain genes using SHAP values.
        """
        intercept = np.zeros(len(geneindices))
        explainer = shap.explainers.Linear((coeff[geneindices], intercept), adata.layers["pcainput"])
        if explain_indices is None:
            shap_values = explainer.shap_values(adata.layers["pcainput"])
        else:
            shap_values = explainer.shap_values(adata.layers["pcainput"][explain_indices])
        return shap_values
