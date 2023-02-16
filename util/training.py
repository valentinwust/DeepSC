import torch
from torch.utils.data import TensorDataset, DataLoader

from ..util import printwtime

def get_RNA_dataloaders(counts, device=None, testshare=0.1, batch_size=64, shuffle=True):
    """ Turn counts into data loaders for training, test set.
    """
    dataset = TensorDataset(counts if device is None else counts.to(device))
    test_size = int(len(dataset)*testshare)
    trainingset, testset = torch.utils.data.random_split(dataset, [ len(dataset)-test_size, test_size])
    trainloader = DataLoader(trainingset, batch_size=batch_size, shuffle=shuffle)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=shuffle)
    return trainloader, testloader

def evaluate_mean_loss(model, loader, device, training=False):
    """ Evaluate loss of model on whole data loader.
    """
    with torch.no_grad():
        if not training is None:
            model.train(training)
        total_loss = 0.0

        for batch in loader:
            k = batch[0].to(device)
            total_loss += model.get_loss(k).sum().item()
    
        return total_loss/len(loader.dataset)

def train_model(model, trainloader, testloader, optimizer, epochs, device, clip_gradients=1., verbose=True):
    """ Train model.
    """
    printwtime(f"Train model {type(model).__name__}")
    printwtime(f"  Optimizer {type(optimizer).__name__} (lr={optimizer.param_groups[0]['lr']}), {epochs} epochs, device {device}.")
    
    history = {"training_loss":[], "test_loss":[], "epoch":[], "lr":[]}
    
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        model.train()
        for batch in trainloader:
            optimizer.zero_grad()
            
            data = batch[0].to(device)
            loss = model.get_loss(data).mean()
            loss.backward()
            if clip_gradients is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_gradients)
            optimizer.step()
            
            running_loss += loss.item()*data.shape[0]
        running_loss = running_loss / len(trainloader.dataset)
        
        evalloss = mean_evaluate_loss(model, testloader, device, False)
        
        history["epoch"].append(epoch)
        history["training_loss"].append(running_loss)
        history["test_loss"].append(evalloss)
        
        if verbose: printwtime(f'  [{epoch + 1}/{epochs}] train loss: {running_loss:.3f}, test loss: {evalloss:.3f}')
        
    return history