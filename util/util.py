from datetime import datetime
import numpy as np
import pandas as pd

def printwtime(text):
    """ Print text with current time.
    """
    print(datetime.now().strftime("%H:%M:%S"),"-",text)

##############################
##### Sampling
##############################

def sample_indices(Nsample, Ntotal, group=None):
    """ Sample indices, if group is provided does stratified sampling.
    """
    if group is None:
        indices = np.random.choice(np.arange(Ntotal), Nsample, replace=False)
    else:
        if len(group)!=Ntotal:
            raise ValueError("Sampling was provided with the wrong group key!")
        frac = Nsample/Ntotal
        df = pd.DataFrame(group, columns=["group"])
        if (df.groupby("group").apply(len)*frac).min()<2:
            raise ValueError("Trying to sample from small groups, stratified sampling probably isn't appropriate here!")
        sample = df.groupby("group", group_keys=False).apply(lambda x: x.sample(frac=frac))
        indices = np.asarray(sample.index)
        np.random.shuffle(indices)
    return indices

##############################
##### Model Summary
##############################

from torchinfo import summary

def model_summary(model, verbose=1):
    """ Keras style model summary.
    """
    summary(model, (1,model.input_size), verbose=verbose);







