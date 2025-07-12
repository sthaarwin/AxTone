"""
Evaluation metrics for tablature generation.

This module provides metrics for evaluating the quality of
generated guitar tablature compared to ground truth.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def tab_accuracy(
    pred: np.ndarray,
    target: np.ndarray,
    threshold: float = 0.5,
    is_fret: bool = False
) -> float:
    """
    Calculate accuracy of tab prediction.
    
    Args:
        pred: Predicted values
        target: Target values
        threshold: Threshold for binary classification (only for string predictions)
        is_fret: Whether this is fret prediction (requires special handling)
        
    Returns:
        Accuracy score
    """
    if is_fret:
        # For fret predictions, only consider positions where a string is actually played
        # Ignore positions where the target is -1 (string not played)
        mask = target >= 0
        if np.sum(mask) == 0:
            return 1.0  # If no strings are played, prediction is perfect by default
            
        return np.mean(pred[mask] == target[mask])
    else:
        # For string predictions (binary classification)
        if isinstance(pred, np.ndarray) and pred.dtype == float:
            # Convert probabilities to binary predictions
            binary_pred = (pred > threshold).astype(int)
        else:
            binary_pred = pred
            
        # Calculate accuracy, ignoring positions where target is -1
        mask = target >= 0
        if np.sum(mask) == 0:
            return 1.0
            
        return np.mean(binary_pred[mask] == target[mask])


def tab_f1_score(
    pred: np.ndarray,
    target: np.ndarray,
    threshold: float = 0.5,
    average: str = 'macro'
) -> float:
    """
    Calculate F1 score for tab prediction.
    
    Args:
        pred: Predicted values
        target: Target values
        threshold: Threshold for binary classification
        average: Averaging method ('micro', 'macro', 'weighted')
        
    Returns:
        F1 score
    """
    # Convert probabilities to binary predictions
    if isinstance(pred, np.ndarray) and pred.dtype == float:
        binary_pred = (pred > threshold).astype(int)
    else:
        binary_pred = pred
        
    # Flatten the arrays for F1 calculation
    pred_flat = binary_pred.flatten()
    target_flat = target.flatten()
    
    # Remove positions where target is -1 (string not played)
    mask = target_flat >= 0
    pred_flat = pred_flat[mask]
    target_flat = target_flat[mask]
    
    # Handle empty arrays
    if len(target_flat) == 0:
        return 1.0
    
    # Calculate F1 score
    return f1_score(target_flat, pred_flat, average=average)


def tab_timing_error(
    pred_onsets: np.ndarray,
    target_onsets: np.ndarray,
    tolerance: float = 0.05
) -> Dict[str, float]:
    """
    Calculate timing errors between predicted and target note onsets.
    
    Args:
        pred_onsets: Predicted onset times (in seconds)
        target_onsets: Target onset times (in seconds)
        tolerance: Tolerance in seconds for matching onsets
        
    Returns:
        Dictionary with timing error metrics:
            - mean_error: Mean absolute error in seconds
            - median_error: Median absolute error in seconds
            - match_rate: Percentage of predicted onsets that match target onsets
    """
    if len(target_onsets) == 0:
        return {
            'mean_error': 0.0,
            'median_error': 0.0,
            'match_rate': 1.0 if len(pred_onsets) == 0 else 0.0
        }
    
    if len(pred_onsets) == 0:
        return {
            'mean_error': float('inf'),
            'median_error': float('inf'),
            'match_rate': 0.0
        }
    
    # For each target onset, find the closest predicted onset
    matched_pred_indices = set()
    errors = []
    
    for target_time in target_onsets:
        # Calculate distances to all predicted onsets
        distances = np.abs(pred_onsets - target_time)
        
        # Find the closest predicted onset
        closest_idx = np.argmin(distances)
        min_distance = distances[closest_idx]
        
        # Only count it as a match if within tolerance
        if min_distance <= tolerance:
            matched_pred_indices.add(closest_idx)
            errors.append(min_distance)
    
    # Calculate error metrics
    if len(errors) > 0:
        mean_error = np.mean(errors)
        median_error = np.median(errors)
    else:
        mean_error = float('inf')
        median_error = float('inf')
    
    # Calculate match rate
    match_rate = len(matched_pred_indices) / len(target_onsets)
    
    return {
        'mean_error': mean_error,
        'median_error': median_error,
        'match_rate': match_rate
    }


def evaluate_tab(
    predictions: Dict[str, np.ndarray],
    targets: Dict[str, np.ndarray],
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Comprehensive evaluation of tablature predictions.
    
    Args:
        predictions: Dictionary with prediction arrays
            - string_probs: String activation probabilities
            - fret_probs: Fret position probabilities
        targets: Dictionary with target arrays
            - string_targets: String activation targets
            - fret_targets: Fret position targets
        threshold: Threshold for binary classification
        
    Returns:
        Dictionary with evaluation metrics:
            - string_accuracy: Accuracy of string predictions
            - fret_accuracy: Accuracy of fret predictions
            - string_f1: F1 score for string predictions
            - fret_f1: F1 score for fret predictions
            - overall_accuracy: Overall accuracy
    """
    # Get predictions
    string_probs = predictions['string_probs']
    fret_probs = predictions['fret_probs']
    
    # Get targets
    string_targets = targets['string_targets']
    fret_targets = targets['fret_targets']
    
    # Convert probabilities to predictions
    string_preds = (string_probs > threshold).astype(float)
    fret_preds = np.argmax(fret_probs, axis=-1)
    
    # Calculate metrics
    string_accuracy = tab_accuracy(string_preds, string_targets, threshold)
    fret_accuracy = tab_accuracy(fret_preds, fret_targets, is_fret=True)
    
    string_f1 = tab_f1_score(string_preds, string_targets, threshold)
    
    # For fret F1, we need to create one-hot encoded targets
    fret_targets_mask = fret_targets >= 0
    fret_f1 = 0.0
    
    if np.sum(fret_targets_mask) > 0:
        # Only calculate F1 for positions where strings are played
        max_fret = fret_probs.shape[-1] - 1
        
        # Create binary targets for each fret position
        fret_targets_binary = np.zeros_like(fret_probs)
        
        # Fill in the correct fret positions
        for i in range(fret_targets.shape[0]):
            for j in range(fret_targets.shape[1]):
                for k in range(fret_targets.shape[2]):
                    if fret_targets[i, j, k] >= 0:
                        fret_targets_binary[i, j, k, fret_targets[i, j, k]] = 1
        
        # Calculate F1 for each string and average
        string_f1_scores = []
        
        for string_idx in range(fret_targets.shape[1]):
            # Get targets for this string
            string_targets = fret_targets_binary[:, string_idx, :, :]
            string_probs = fret_probs[:, string_idx, :, :]
            
            # Flatten the arrays
            string_targets_flat = string_targets.reshape(-1, max_fret + 1)
            string_probs_flat = string_probs.reshape(-1, max_fret + 1)
            
            # Get string mask
            string_mask = np.any(string_targets_flat > 0, axis=1)
            
            if np.sum(string_mask) > 0:
                # Calculate F1 score for this string
                string_targets_filtered = string_targets_flat[string_mask]
                string_probs_filtered = string_probs_flat[string_mask]
                
                string_preds_filtered = np.zeros_like(string_targets_filtered)
                for i in range(len(string_probs_filtered)):
                    pred_fret = np.argmax(string_probs_filtered[i])
                    string_preds_filtered[i, pred_fret] = 1
                
                # Calculate F1 score (average across all frets)
                string_f1 = f1_score(
                    string_targets_filtered.flatten(),
                    string_preds_filtered.flatten(),
                    average='macro'
                )
                string_f1_scores.append(string_f1)
        
        if string_f1_scores:
            fret_f1 = np.mean(string_f1_scores)
    
    # Calculate overall accuracy (weighted combination of string and fret accuracy)
    overall_accuracy = 0.5 * string_accuracy + 0.5 * fret_accuracy
    
    return {
        'string_accuracy': string_accuracy,
        'fret_accuracy': fret_accuracy,
        'string_f1': string_f1,
        'fret_f1': fret_f1,
        'overall_accuracy': overall_accuracy
    }