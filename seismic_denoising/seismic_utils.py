import numpy as np 
from batchflow.plotter import plot
import pickle
import torch
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, MeanSquaredError, UniversalImageQualityIndex

def pearson_correlation_for_matrices(m1, m2):
    
    return np.mean((m1 - m1.mean(axis=-1, keepdims=True)) * (m2 - m2.mean(axis=-1, keepdims=True)), axis=-1) / (m1.std(axis=-1) * m2.std(axis=-1))

def pearson_correlation_for_vectors(v1, v2):
    
    return np.mean((v1 - v1.mean()) * (v2 - v2.mean())) / (v1.std() * v2.std())

def image_to_freq_domain(image):
    
    fft_image = np.log(np.abs(np.fft.fftshift(np.fft.fft2(image))) ** 2)
    return fft_image

def cross_correlation_in_freq_domain(image1, image2, window_shape=None):
    
    x_corr = np.fft.ifft2(np.fft.fft2(image1) * np.fft.fft2(image2))
    return np.real((x_corr - x_corr.mean()) / x_corr.std())

def cross_correlation_grid(image1, image2, window_shape, trace=False):
    """ 
    Compute correlation matrix between two images over window.
    Trace parameter only works when window_shape = (window_shape_x, 1).
    """
    X, Y = image1.shape
    window_shape_x, window_shape_y = window_shape    
    if window_shape_x % 2 != 0:
        before_x = window_shape_x // 2
        after_x = before_x
        if not trace:
            before_y = window_shape_y // 2
            after_y = before_y
        else:
            before_y = after_y = 0
    else:
        before_x = window_shape_x - 1
        if not trace:
            before_y = window_shape_x - 1
            after_x = 0
            after_y = 0
        else:
            before_y = 0
            after_x = after_y = 0
     
    image1_pad = np.pad(image1, pad_width=((before_x, after_x), (before_y, after_y)), mode='mean')
    image2_pad = np.pad(image2, pad_width=((before_x, after_x), (before_y, after_y)), mode='mean')
    patch1 = np.lib.stride_tricks.sliding_window_view(image1_pad, window_shape=(window_shape_x, window_shape_y))
    patch2 = np.lib.stride_tricks.sliding_window_view(image2_pad, window_shape=(window_shape_x, window_shape_y))

    newshape = (patch1.shape[0] * patch1.shape[1], patch1.shape[2] * patch1.shape[3])
    patch1_flatten = patch1.reshape(newshape)
    patch2_flatten = patch2.reshape(newshape)
    correlation = np.nan_to_num(pearson_correlation_for_matrices(patch1_flatten, patch2_flatten).reshape(X, Y))
    
    return correlation

#@njit
def plot_denoised(num_images,
                  images,
                  reconstructed_images,
                  noise,
                  window_shape=(4, 4),
                  mode='images',
                  freq_domain=False,
                  hist=False):
    
    imgs = images.copy()
    reconstructed_imgs = reconstructed_images.copy()
    ns = noise.copy()
    
    if num_images > len(images):
        num_images = len(images)
    
    # if cross_correlation_via_ifft:
    #     cross_correlation = cross_correlation_in_freq_domain
    #     window_shape = None
    #     suptitle='IFFT'
    if mode == 'cc':
        trace = False
    elif mode == 'cc_trace':
        trace = True
        
    all_images = []
    
    for i in range(num_images):
        
        if freq_domain:
            imgs[i] = image_to_freq_domain(imgs[i])
            reconstructed_imgs[i] = image_to_freq_domain(reconstructed_imgs[i])
            ns[i] = image_to_freq_domain(ns[i])
        
        # fixed
        img_min = min(imgs[i].min(), reconstructed_imgs[i].min(), ns[i].min())
        img_max = max(imgs[i].max(), reconstructed_imgs[i].max(), ns[i].max())
        imgs[i] = (imgs[i] - img_min) / (img_max - img_min) 
        reconstructed_imgs[i] = (reconstructed_imgs[i] - img_min) / (img_max - img_min) 
        ns[i] = (ns[i] - img_min) / (img_max - img_min)
            
        if mode == 'images':
            all_images.extend((imgs[i].T,
                               reconstructed_imgs[i].T,
                               ns[i].T)
            )
            title = ['Noisy image', 'Denoised image', 'Noise']
            
        else:
            cc_ir = cross_correlation_grid(imgs[i], reconstructed_imgs[i], window_shape=window_shape, trace=trace)
            cc_rn = cross_correlation_grid(reconstructed_imgs[i], ns[i], window_shape=window_shape, trace=trace)
            cc_in = cross_correlation_grid(imgs[i], ns[i], window_shape=window_shape, trace=trace)
            all_images.extend((cc_ir.T,
                               cc_rn.T,
                               cc_in.T)
            )
            #print(cc_in.min(), cc_rn.min(), cc_ir.min())
            if mode == 'cc':
                title = ['Image/denoised CC', 'Denoised/noise CC', 'Image/noise CC']
            elif mode == 'cc_trace':
                title = ['Image/denoised CC_trace', 'Denoised/noise CC_trace', 'Image/noise CC_trace']
        
            if hist:
                sample_ir = sample_rn = sample_in = np.array([])
                sample_ir = np.concatenate((sample_ir, cc_ir.reshape(-1)))
                sample_rn = np.concatenate((sample_rn, cc_rn.reshape(-1)))
                sample_in = np.concatenate((sample_in, cc_in.reshape(-1)))
     
    if freq_domain:
        suptitle = 'Freq domain'
    else:
        suptitle = 'Origin domain'
    
    if mode in ['cc', 'cc_trace'] or freq_domain:
        cmap = 'seismic'
    else:
        cmap = 'Greys_r'
    
    if mode in ['cc', 'cc_trace']:
        vmin, vmax = -1, 1
    elif mode == 'images' or freq_domain:
        vmin, vmax = 0, 1
     
    if hist:
        plot([sample_ir, sample_rn, sample_in],
             combine='separate',
             mode='histogram',
             n_cols=3,
             title=title)
    else:                          
        plot(all_images,
             title=title * (len(all_images) // 2),
             combine='separate',
             ncols=3,
             figsize=(15, 30),
             title_fontsize=17,
             tight_layout=True,
             suptitle=suptitle,
             suptitle_y=1.05,
             colorbar=True,
             cmap=cmap,
             vmin=vmin,
             vmax=vmax)

def calc_ssim(pred, target):
    """ Calculates SSIM metric """
    pred = torch.tensor(pred)
    target = torch.tensor(target)
    ssim = StructuralSimilarityIndexMeasure()
    result = ssim(pred, target)
    
    return result.item()

def calc_metrics(pred, target, metrics, freq_domain=False):
    """ Calculates the main metrics from the metrics dict """
    if freq_domain:
        pred = image_to_freq_domain(pred)
        target = image_to_freq_domain(target)
        
    pred = torch.tensor(pred)
    target = torch.tensor(target)
    
    results = {}
    for metric in metrics:
        result = metrics[metric](pred, target).item()
        if freq_domain:
            results[f"{metric}_fft"] = result
        else:
            results[metric] = result
        
    return results

def run_val_ppl_and_save_metrics(batch_size, model_name, val_pipeline, val_iters):
    """ Runs validation pipeline and saves calculated metrics to the file """
    val_pipeline.run(n_iters=val_iters, drop_last=False, bar='n')
    metrics = val_pipeline.v('metrics')
    
    with open(f'{model_name}_metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f)
    
    return metrics 
