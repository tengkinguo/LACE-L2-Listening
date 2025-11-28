# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 09:26:13 2025

@author: T
"""

import numpy as np

def compute_ece(y_true, y_prob, n_bins=10):
    """
    Calculates the Expected Calibration Error (ECE).
    
    Args:
        y_true (np.array): True binary labels (0 or 1).
        y_prob (np.array): Predicted probabilities.
        n_bins (int): Number of bins for discretization.
        
    Returns:
        float: The weighted average calibration error.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    
    # Get bin indices for each prediction
    bin_indices = np.digitize(y_prob, bins) - 1
    
    ece = 0.0
    n_samples = len(y_true)
    
    for i in range(n_bins):
        mask = bin_indices == i
        if np.any(mask):
            bin_conf = np.mean(y_prob[mask])
            bin_acc = np.mean(y_true[mask])
            bin_count = np.sum(mask)
            ece += (bin_count / n_samples) * np.abs(bin_acc - bin_conf)
            
    return ece

def reliability_curve(y_true, y_prob, n_bins=10):
    """
    Computes points for the reliability diagram.
    
    Returns:
        tuple: (bin_centers, accuracies, counts)
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1

    bin_centers = []
    accuracies = []
    counts = []

    for b in range(n_bins):
        mask = bin_indices == b
        if np.sum(mask) > 0:
            prob_mean = np.mean(y_prob[mask])
            acc = np.mean(y_true[mask])
            bin_centers.append(prob_mean)
            accuracies.append(acc)
            counts.append(np.sum(mask))
        else:
            bin_centers.append((bins[b] + bins[b + 1]) / 2.0)
            accuracies.append(np.nan)
            counts.append(0)
            
    return np.array(bin_centers), np.array(accuracies), np.array(counts)