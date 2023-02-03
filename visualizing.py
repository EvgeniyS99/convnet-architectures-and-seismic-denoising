import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import numpy as np

def imshow(inp, title=None, plt_ax=plt):
    """For visualize tensors"""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt_ax.imshow(inp)
    if title is not None:
        plt_ax.set_title(title)
    plt_ax.grid(False)

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
