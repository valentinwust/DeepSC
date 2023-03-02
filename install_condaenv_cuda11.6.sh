conda create -n torch python=3.7 -y
conda activate torch
conda config --set channel_priority strict ???
conda install pytorch torchvision torchaudio cuda-toolkit=11.6 -c pytorch -c nvidia
conda install ipykernel  #pip install ipykernel
ipython kernel install --user --name=torch 
conda install scanpy loompy shap matplotlib seaborn