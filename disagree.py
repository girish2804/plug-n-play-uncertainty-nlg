import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Any
import numpy as np

def get_relative_top_filter(scores: torch.FloatTensor, relative_top: float = 0.1, min_tokens_to_keep: int = 1):
    """
    Creates a boolean mask for filtering tokens based on their relative log probability.
    
    Args:
        scores: The logits/scores tensor
        relative_top: The relative threshold for filtering (e.g. 0.1 means keep tokens with prob >= 0.1*max_prob)
        min_tokens_to_keep: Minimum number of tokens to keep
        
    Returns:
        Boolean mask where False indicates tokens to keep
    """
    scores_normalized = scores.log_softmax(dim=-1) 
    sorted_logits, sorted_indices = torch.sort(scores_normalized, descending=True)
    min_thresh = sorted_logits[..., min_tokens_to_keep-1] 
    probs_max = torch.max(scores_normalized, dim=-1).values
    probs_thresh = probs_max + np.log(relative_top)
    probs_thresh = torch.min(min_thresh, probs_thresh)
    probs_thresh = probs_thresh.unsqueeze(-1)
    return scores_normalized < probs_thresh

def calculate_all_disagreement_metrics(final_logits, premature_logits, 
                                      calculate_metrics=None, relative_top=0.1, min_tokens_to_keep=32, store_logits= False):
    """
    Calculate all disagreement metrics aggregated across all premature layers vs the final layer.
    
    Args:
        final_logits: Logits from the final layer [batch_size, vocab_size]
        premature_logits_list: List of logits from earlier layers [n_layers, batch_size, vocab_size]
        calculate_metrics: List of metrics to calculate (default: all)
        relative_top: The relative threshold for filtering (e.g. 0.1 means keep tokens with prob >= 0.1*max_prob)
        min_tokens_to_keep: Minimum number of tokens to keep
    
    Returns:
        Dictionary of aggregated disagreement metrics
    """
    batch_size, vocab_size = final_logits.shape
    n_layers = len(premature_logits)
    
    premature_logits_list = [premature_logits[x] for x in premature_logits]
    
    # Get filter mask based on final layer logits
    # Note: The get_relative_top_filter returns True for tokens to filter out
    # So we need to invert it to get tokens to keep
    filter_mask = ~get_relative_top_filter(final_logits, 
                                         relative_top=relative_top, 
                                         min_tokens_to_keep=min_tokens_to_keep)
    
    # Convert final logits to probabilities
    final_probs = F.softmax(final_logits, dim=-1)
    final_log_probs = F.log_softmax(final_logits, dim=-1)
    
    # Apply mask to final probabilities and renormalize
    masked_final_probs = final_probs * filter_mask
    normalization_factor = masked_final_probs.sum(dim=-1, keepdim=True)
    normalized_final_probs = masked_final_probs / (normalization_factor + 1e-10)
    normalized_final_log_probs = torch.log(normalized_final_probs + 1e-10)  # Need epsilon for log of zero

    results = {}
    
    # Store unfiltered logits in results
    if store_logits:
        results['unfiltered_final_logits'] = final_logits.clone().detach().to('cpu')
        results['unfiltered_premature_logits'] = [logits.clone().detach().to('cpu') for logits in premature_logits_list]
    
    # Also store count of tokens used after filtering
        results['tokens_kept_count'] = torch.sum(filter_mask[0]).item()
        results['tokens_kept_percentage'] = (torch.sum(filter_mask[0]).item() / vocab_size) * 100
    
    # Define which metrics to calculate (all by default)
    if calculate_metrics is None:
        calculate_metrics = ['jsd', 'kl', 'entropy_mixture', 'mutual_info', 
                           'epkl', 'cv', 'bhattacharyya', 'tvd']
    
    # Initialize metric aggregation across all layers
    metric_values = {metric: [] for metric in calculate_metrics}
    
    # Process each premature layer
    for layer_idx, premature_logits in enumerate(premature_logits_list):
        # Convert premature logits to probabilities and log probabilities
        premature_probs = F.softmax(premature_logits, dim=-1)
        premature_log_probs = F.log_softmax(premature_logits, dim=-1)
        
        # Apply the same token mask and renormalize
        masked_premature_probs = premature_probs * filter_mask
        premature_norm_factor = masked_premature_probs.sum(dim=-1, keepdim=True)
        normalized_premature_probs = masked_premature_probs / (premature_norm_factor + 1e-10)
        normalized_premature_log_probs = torch.log(normalized_premature_probs + 1e-10)  # Need epsilon for log of zero
        
        # Calculate each requested metric for this layer
        for metric in calculate_metrics:
            if metric == 'jsd':
                # 1. Jensen-Shannon Divergence
                m_probs = 0.5 * (normalized_final_probs + normalized_premature_probs)
                m_log_probs = torch.log(m_probs + 1e-10)
                
                # KL(final || m)
                kl1 = F.kl_div(m_log_probs, normalized_final_probs, reduction='none')
                # kl1 = torch.sum(normalized_final_probs * kl1, dim=-1)
                
                # KL(premature || m)
                kl2 = F.kl_div(m_log_probs, normalized_premature_probs, reduction='none')
                # kl2 = torch.sum(normalized_premature_probs * kl2, dim=-1)
                
                values = 0.5 * (kl1 + kl2)
                values = values.mean(-1)
                metric_values[metric].append(values.mean().item())
                
            elif metric == 'kl':
                # 2. Kullback-Leibler Divergence using standard PyTorch function
                # Note: F.kl_div expects log-probabilities for the first argument
                values = F.kl_div(normalized_premature_log_probs, normalized_final_probs, reduction='none')
                # values = torch.sum(normalized_final_probs * values, dim=-1)
                values = values.mean(-1)
                metric_values[metric].append(values.mean().item())
                
            elif metric == 'entropy_mixture':
                # 3. Entropy of Mixture Distribution
                m_probs = 0.5 * (normalized_final_probs + normalized_premature_probs)
                values = -torch.sum(m_probs * torch.log(m_probs + 1e-10), dim=-1)
                metric_values[metric].append(values.mean().item())
                
                # Also calculate entropy of individual distributions for this layer
                entropy_final = -torch.sum(normalized_final_probs * normalized_final_log_probs, dim=-1)
                entropy_premature = -torch.sum(normalized_premature_probs * normalized_premature_log_probs, dim=-1)
                
                # Store entropy of final layer (same for all comparisons)
                if layer_idx == 0:
                    results['entropy_final'] = entropy_final.mean().item()
                
            elif metric == 'mutual_info':
                # 4. Mutual Information
                # I(X;Y) = H(X) + H(Y) - H(X,Y)
                entropy_final = -torch.sum(normalized_final_probs * normalized_final_log_probs, dim=-1)
                entropy_premature = -torch.sum(normalized_premature_probs * normalized_premature_log_probs, dim=-1)
                
                # For joint entropy approximation
                joint_entropy = -torch.sum(normalized_final_probs * normalized_premature_log_probs, dim=-1)
                values = entropy_final + entropy_premature - joint_entropy
                metric_values[metric].append(values.mean().item())
                
            elif metric == 'epkl':
                # 5. Expected Pairwise KL Divergence using standard PyTorch functions
                kl_forward = F.kl_div(normalized_premature_log_probs, normalized_final_probs, reduction='none')
                # kl_forward = torch.sum(normalized_final_probs * kl_forward, dim=-1)
                
                kl_backward = F.kl_div(normalized_final_log_probs, normalized_premature_probs, reduction='none')
                # kl_backward = torch.sum(normalized_premature_probs * kl_backward, dim=-1)
                
                values = 0.5 * (kl_forward + kl_backward)
                values = values.mean(-1)
                metric_values[metric].append(values.mean().item())
                
            elif metric == 'cv':
                # 6. Coefficient of Variation for Token Probabilities
                # Stack probabilities to calculate mean and std
                stacked_probs = torch.stack([normalized_final_probs, normalized_premature_probs], dim=0)
                mean_probs = torch.mean(stacked_probs, dim=0)  # [batch_size, vocab_size]
                std_probs = torch.std(stacked_probs, dim=0)    # [batch_size, vocab_size]
                
                # Calculate CV = std/mean for each token
                cv_values = std_probs / (mean_probs + 1e-10)
                
                # Apply the token mask to CV values
                masked_cv = cv_values * filter_mask
                
                # Aggregate across vocabulary dimension
                values = torch.sum(masked_cv, dim=-1) / (torch.sum(filter_mask, dim=-1) + 1e-10)
                metric_values[metric].append(values.mean().item())
                
            elif metric == 'bhattacharyya':
                # 7. Bhattacharyya Distance
                bc_coeff = torch.sum(torch.sqrt(normalized_final_probs * normalized_premature_probs), dim=-1)
                values = -torch.log(bc_coeff + 1e-10)
                metric_values[metric].append(values.mean().item())
                
            elif metric == 'tvd':
                # 8. Total Variation Distance
                values = 0.5 * torch.sum(torch.abs(normalized_final_probs - normalized_premature_probs), dim=-1)
                metric_values[metric].append(values.mean().item())
    
    # Aggregate metrics across all layers
    for metric in calculate_metrics:
        # Average across all layers
        results[metric] = sum(metric_values[metric]) / len(metric_values[metric])
        
        # Also provide min and max across layers
        results[metric + '_min'] = min(metric_values[metric])
        results[metric + '_max'] = max(metric_values[metric])
    
    return results