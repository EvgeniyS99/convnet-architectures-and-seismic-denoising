import torch
import torch.nn as nn
import torchvision
import numpy as np

def predict_one_sample(model, inputs):
    """
    Returns the probability vector of the sample 
    """
    device = 'cuda:0'
    inputs = inputs.to(device)
    model.eval()
    with torch.no_grad():
        out = model(inputs)
        proba = nn.functional.softmax(out, dim=1)
        
    return proba

def predict(model, test_loader):
    """Returns the probabilities vectors for each sample in test set"""
    device = 'cuda:0'
    with torch.no_grad():
        logits = []
    
        for inputs in test_loader:
            inputs = inputs.to(device)
            model.eval()
            outputs = model(inputs).cpu()
            logits.append(outputs)
            
    probs = nn.functional.softmax(torch.cat(logits), dim=1).numpy()
    
    return probs

def make_predictions(model, images):
    """Return class for each sample in the test set"""
    predictions = np.argmax(predict(model, images), axis=-1)
    
    return predictions
