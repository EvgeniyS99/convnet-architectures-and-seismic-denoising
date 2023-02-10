import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm, tqdm_notebook
import pickle

def train(model, opt, loss_fn, lr_scheduler, epochs, train_data, val_data, name, path):
    """
    """
    losses = {'train': [], 'val': []}
    metrics = {'train': [], 'val': []}
    device = "cuda:0"
    #lr_scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=3, gamma=0.1)
    max_acc = 0.
    for epoch in range(epochs):
        print('Epoch {}/{}:'.format(epoch, epochs - 1))
        for phase in ['train', 'val']:
            running_acc = 0.
            running_loss = 0.
            
            # train phase
            if phase == 'train':
                dataloader = train_data
                model.train()  
            # eval phase    
            else:
                dataloader = val_data
                model.eval()
                
            # iterate over batch    
            for X_batch, y_batch in tqdm(dataloader, desc=f'{phase} iter:'):
                opt.zero_grad()
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                with torch.set_grad_enabled(phase == 'train'):
                    y_pred = model(X_batch)
                    loss_value = loss_fn(y_pred, y_batch)
                    if phase == 'train':
                        loss_value.backward()
                        opt.step()
                    
                preds = y_pred.argmax(-1)
                corrects = (preds == y_batch).float()
                running_acc += corrects.mean()
                running_loss += loss_value.item()
            
            epoch_loss = running_loss / len(dataloader)
            epoch_acc = running_acc.cpu() / len(dataloader)
            
            if phase == 'train':
                lr_scheduler.step()
            
            losses[phase].append(epoch_loss)
            metrics[phase].append(epoch_acc)
            
            # save model weights
            if phase == 'val' and epoch_acc > max_acc:
                model_weights = model.state_dict()
                torch.save(model_weights, path)
                max_acc = epoch_acc
        
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            print('{} Accuracy: {:.4f}'.format(phase, epoch_acc))
            
    # save losses and metrics        
    with open(f'{name}_losses.pkl', 'wb') as loss_file:
        pickle.dump(losses, loss_file)
    
    with open(f'{name}_metrics.pkl', 'wb') as metrics_file:
        pickle.dump(metrics, metrics_file)
            
    return losses, metrics

def step_scheduler(opt, step_size, gamma):
    return torch.optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma)
