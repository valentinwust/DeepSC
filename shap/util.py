import numpy as np

##############################
##### Correlations
##############################

def correlate_expression_latentshap(shap_values, logcountinput):
    """ Correlate gene expression with shap values of latent dimensions.
    """
    singlecorrel = lambda i, j: np.corrcoef(logcountinput[:,i], shap_values[j,:,i])[1,0]
    correls = [[singlecorrel(i,j) for i in range(shap_values.shape[2])] for j in range(shap_values.shape[0])]
    correls = np.asarray(correls)
    return correls