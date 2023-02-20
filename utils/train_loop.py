import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm, tqdm_notebook
import pickle
import torch.nn.functional as F
import numpy as np

def train(model, opt, loss_fn, lr_scheduler, epochs, train_data, val_data, name, path, task, segmentation_metrics=None, device='cpu'):
    """Train loop for classification and segmentations tasks"""
    losses = {'train': [], 'val': []}
    metrics = {'train': [], 'val': []}
    max_metrics = 0.
    
    for epoch in range(epochs):
        # print('Epoch {}/{}:'.format(epoch, epochs - 1))
        for phase in ['train', 'val']:
            running_metrics = 0.
            running_loss = 0.
            
            # train phase
            if phase == 'train':
                dataloader = train_data
                model.train()  
            # eval phase    
            else:
                dataloader = val_data
                model.eval()
                
            loss_during_epoch = []   
            # iterate over batch    
            for i, (X_batch, y_batch) in tqdm(enumerate(dataloader), desc=f'{phase} iter:'):
                opt.zero_grad()
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                with torch.set_grad_enabled(phase == 'train'):
                    y_pred = model(X_batch)
                    loss_value = loss_fn(y_pred, y_batch)
                    if phase == 'train':
                        loss_value.backward()
                        opt.step()
                        
                if task == 'classification':   
                    preds = y_pred.argmax(-1)
                    corrects = (preds == y_batch).float()
                    running_metrics += corrects.mean()
                    
                elif task == 'segmentation':
                    if segmentation_metrics is None:
                        segmentation_metrics = mIoU
                    running_metrics += segmentation_metrics(y_pred, y_batch)
                    
                else:
                    raise ValueError('There is only classification and segmentation tasks')
                
                running_loss += loss_value.item()
                
                # plot metrics and losses during the one epoch
                if phase == 'train':
                    loss_during_epoch.append(loss_value.item())
                    
                    if (i + 1) % 3 == 0:
                        fig, axes = plt.subplots(1, 3, figsize=(12, 6))
                        clear_output(wait=True)
                        
                        axes[0].plot(loss_during_epoch)
                        axes[0].set_xlabel('batch iter')
                        axes[0].set_ylabel('loss during epoch')
                        
                        if losses:
                            axes[1].plot(losses['train'], label='train loss')
                            axes[1].plot(losses['val'], label='val loss')
                            axes[1].set_xlabel('epoch')
                            axes[1].set_ylabel('loss')
                            axes[1].legend()
                            
                        if metrics:
                            axes[2].plot(metrics['train'], label='train miou')
                            axes[2].plot(metrics['val'], label='val miou')
                            axes[2].set_xlabel('epoch')
                            axes[2].set_ylabel('miou')
                            axes[2].legend()
                        
                        plt.show()
            
            epoch_loss = running_loss / len(dataloader)
            
            if task == 'classification':
                running_metrics = running_metrics.cpu()
                
            epoch_metrics = running_metrics / len(dataloader)
            
            if phase == 'train':
                lr_scheduler.step()
            
            # logging losses and matrics
            losses[phase].append(epoch_loss)
            metrics[phase].append(epoch_metrics)
            
            # save model weights
            if phase == 'val' and epoch_metrics > max_metrics:
                model_weights = model.state_dict()
                torch.save(model_weights, path)
                max_metrics = epoch_metrics
        
            # print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            # print('{} Accuracy: {:.4f}'.format(phase, epoch_metrics))
            
    # save losses and metrics        
    with open(f'{name}_losses.pkl', 'wb') as loss_file:
        pickle.dump(losses, loss_file)
    
    with open(f'{name}_metrics.pkl', 'wb') as metrics_file:
        pickle.dump(metrics, metrics_file)
            
    return losses, metrics

def step_scheduler(opt, step_size, gamma):
    return torch.optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma)

def mIoU(pred, true, num_classes=151):

    pred = F.softmax(pred, dim=1)
    pred = pred.argmax(dim=1).squeeze(1)
    pred = pred.cpu() 
    true = true.cpu()
    
    iou = []
    
    for c in range(num_classes):
        target_class = (true == c)
        
        if target_class.sum() == 0:
            continue
            
        pred_class = (pred == c)
        running_intersection = (pred_class * target_class).sum()
        running_union = pred_class.sum() + target_class.sum() - running_intersection
        running_iou = running_intersection / running_union 
        iou.append(running_iou)
    
    return np.mean(iou)