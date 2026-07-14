import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import numpy as np
from IPython.display import clear_output
import torch

def imshow(inp, mean=np.array([0.485, 0.456, 0.406]), std=np.array([0.229, 0.224, 0.225]), title=None, normalize=False, plt_ax=plt):
    """Visualizes tensors"""
    if normalize:
        mean = mean
        std = std
    else:
        mean = 0
        std = 1
        
    inp = std * inp + mean
    
    if len(inp.shape) == 3:
        inp = inp.numpy().astype(int).transpose((1, 2, 0))
        inp = np.clip(inp, 0, 255)
        
    plt_ax.imshow(inp)
    plt_ax.axis('off')
    plt_ax.grid(False)
    
    if title is not None:
        plt_ax.set_title(title)
        
def plot_origin_data(dataset):
    """Plot image and ground truth mask for segmentation task"""
    plt.figure(figsize=(20, 6))

    for i in range(6):
        image, mask = dataset[i]
        plt.subplot(2, 6, i+1)
        plt.axis('off')
        imshow(image)

        plt.subplot(2, 6, i+7)
        plt.axis('off')
        imshow(mask)

def plot_train_set():
    """Visualization of the predicted results on the train set"""
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(10, 10), \
                            sharey=True, sharex=True)
    
    for fig_x in ax.flatten():
        random_samples = int(np.random.uniform(0, len(train_dataset)))
        im, label = train_dataset[random_samples]
        img_label = train_dataset.label_encoder.inverse_transform([label])[0]
        imshow(im.data.cpu(), title=lbl_dict[img_label], plt_ax=fig_x)

def plot_results(model, test_dataset):
    """Visualization of the predicted results on the test set"""
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(12, 12), \
                            sharey=True, sharex=True)
    fig.suptitle('Predicted labels', fontsize='20')
    
    for fig_ax in ax.flatten():
        random_samples = int(np.random.uniform(0, len(test_dataset)))
        im_test, label = test_dataset[random_samples]
        img_label = test_dataset.label_encoder.inverse_transform([label])[0]
        
        #fig_x.add_patch(patches.Rectangle((0, 53), 1, 1,color='white'))
        
        prediction_proba = predict_one_sample(model, im_test.unsqueeze(0))
        predicted_class = prediction_proba.argmax().cpu().item()
        predicted_class = test_dataset.label_encoder.inverse_transform([predicted_class])[0]
        predicted_class = lbl_dict[predicted_class]
        
        imshow(im_test.cpu(), title=predicted_class, plt_ax=fig_ax)
        #fig_ax.text(1, 50, f'{predicted_class}', c='white', fontsize=20)
        
def plot_predicted_mask(loader, model, num_images, height, device):
    """
    Plot images, their ground truth masks and 
    predicted masks
    """
    images, masks = next(iter(loader))
    images = images.to(device)
    
    model.eval()
    
    with torch.no_grad():
        pred = model(images)
        pred = pred.detach().cpu()
        images = images.detach().cpu()
        pred_mask = pred.argmax(dim=1)
        
    fig, axes = plt.subplots(num_images, 3, figsize=(20, height))
    
    axes[0, 0].set_title('Image')
    axes[0, 1].set_title('Ground truth mask')
    axes[0, 2].set_title('Predicted mask')
    
    for i in range(num_images):
        imshow(images[i], plt_ax=axes[i, 0])
        imshow(masks[i], plt_ax=axes[i, 1])
        imshow(pred_mask[i], plt_ax=axes[i, 2])
        
def plot_during_epoch(loss_during_epoch, losses, metrics):
    """Plots losses and metrics during one epoch and also vs epoch"""
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