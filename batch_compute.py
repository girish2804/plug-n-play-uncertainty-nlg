import os
import pandas as pd
import torch
import pickle as pkl
import json
import torch.nn as nn
import numpy as np
import tqdm
import traceback
import csv
from sklearn.metrics import roc_auc_score
import gc  # Add garbage collection

from utils import compute_semantic_clusters, compute_semantic_entropy, compute_semantic_entropy_new, prepare_results
from args import Args

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
args = Args()
auth_token = 'set huggingface authorization token'
device_llm = 'cuda:0'
device_deberta = 'cuda:0'
removed_sample_ids = []
base_path = '/notebooks/clarification/SDLG/modeified_sdlg/result/truthful-llama/'
base_path_cluster = '/notebooks/clarification/SDLG/modeified_sdlg/result/truthful-llama/'
base_path_final = '/notebooks/clarification/SDLG/modeified_sdlg/'
num_instances = 817 # dataset instances
model_type = args.deberta_model

correctness_metric = ["rougeL", "bleurt", "rouge1"][1]
num_total_gens = 10
auroc_keys_baseline = ["normalised_semantic_entropy", "unnormalised_semantic_entropy"] 
auroc_keys = ["normalised_semantic_entropy", "unnormalised_semantic_entropy", #"epistemic_entropy", 
               'entropy_final', 'jsd', 'jsd_min', 'jsd_max', 'kl', 'kl_min', 'kl_max', 'entropy_mixture', 'entropy_mixture_min', 'entropy_mixture_max', 'mutual_info', 'mutual_info_min', 'mutual_info_max', 'epkl', 'epkl_min', 'epkl_max', 'cv', 'cv_min', 'cv_max', 'bhattacharyya', 'bhattacharyya_min', 'bhattacharyya_max', 'tvd', 'tvd_min', 'tvd_max'] 
correctness_threshold_list = [-0.05, -0.1, -0.15, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1.0, 0, 0.03, 0.07, 0.1, .13, .16, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
run_keys = ['sdlg'] #, 'baseline']
compute_cleaned = args.compute_cleaned

csv_header = [
    # "run_id",
    "run_key",
    "num_gens",
    "correctness_threshold",
    "auroc_norm_sem_ent",
    "auroc_unnorm_sem_ent",
    "auroc_ours_norm_sem_ent",
    "auroc_ours_unnorm_sem_ent",
    'entropy_final', 'jsd', 'jsd_min', 'jsd_max', 'kl', 'kl_min', 'kl_max', 'entropy_mixture', 'entropy_mixture_min', 'entropy_mixture_max', 'mutual_info', 'mutual_info_min', 'mutual_info_max', 'epkl', 'epkl_min', 'epkl_max', 'cv', 'cv_min', 'cv_max', 'bhattacharyya', 'bhattacharyya_min', 'bhattacharyya_max', 'tvd', 'tvd_min', 'tvd_max',
    "auroc_kuhn_norm_ent",
    "auroc_kuhn_unnorm_ent",
]

csv_filepath = os.path.join(base_path_final, 'auroc_truthful-llama.csv')
if not os.path.exists(csv_filepath):
    with open(csv_filepath, 'w', newline='') as f:  # Use newline='' to avoid extra blank rows
        writer = csv.writer(f)
        writer.writerow(csv_header)  # Write the header

for run_key in run_keys:

    # BATCH PROCESSING - Process instances in smaller batches
    batch_size = 100  # Adjust this based on your memory constraints

    # Storage for all results across ALL batches - INITIALIZE ONCE
    all_bleurt_scores = []
    all_batch_questions = []
    all_semantic_entropies_final = []
    all_semantic_entropies_kuhn_final = []
    all_semantic_entropies_ours_final = []

    print(f"Processing {num_instances} instances in batches of {batch_size}")

    # ===================================================================
    # PHASE 1: BATCH PROCESSING - Collect all entropy data
    # ===================================================================
    for batch_start in range(0, num_instances, batch_size):
        batch_end = min(batch_start + batch_size, num_instances)
        current_batch_size = batch_end - batch_start

        print(f"Processing batch {batch_start}-{batch_end-1} ({current_batch_size} instances)")

        # Load batch data
        list_results_dict, list_bleurt, dataset_size = prepare_results(num_samples=batch_end,
                                                                        run_key=run_key,
                                                                        metric=correctness_metric,
                                                                        start_sample_id=batch_start,
                                                                        base_path=base_path_cluster)

        # Clean up large data immediately after loading
        for results_dict in list_results_dict:
            for x in results_dict[run_key]['epistem_entropies']:
                del x['unfiltered_premature_logits']
                del x['unfiltered_final_logits']
                del x['tokens_kept_count']
                del x['tokens_kept_percentage']

        # Initialize batch-level collections for THIS batch only
        batch_bleurt_scores = []
        batch_questions = []
        batch_semantic_entropies, batch_semantic_entropies_kuhn = [], []
        batch_semantic_entropies_ours = []

        # We only need the final num_gens iteration, so no need for the loop
        num_gens = num_total_gens
        list_num_semantic_clusters, list_num_generations = [], []

        # iterate over instances in this batch
        for i, results_dict in enumerate(list_results_dict):
            try:
                # ---------- create mask of considered generations
                if num_gens < len(results_dict[run_key]['generations']):
                    boolean_mask = [True] * num_gens + [False] * (len(results_dict[run_key]['generations']) - num_gens)
                else:
                    boolean_mask = [True] * len(results_dict[run_key]['generations'])

                boolean_mask = torch.tensor(boolean_mask)

                if run_key in ("sdlg",'dola_sdlg'):
                    mask = torch.tensor([1] + [gen['token_likelihood'] for gen in results_dict[run_key]['generations'][1:]])
                    assert torch.all(mask > 0) and torch.all(mask[1:] < 1), f"mask: {mask}"
                elif run_key == "baseline":
                    mask = boolean_mask
                # ----------

                list_num_generations.append(torch.sum(boolean_mask).item())

                all_considered_generations, all_considered_likelihoods = [], []
                for m, included in enumerate(boolean_mask):
                    if included:
                        all_considered_generations.append(results_dict[run_key]['generations'][m])
                        all_considered_likelihoods.append(results_dict[run_key]['likelihoods'][m])

                if results_dict[run_key][f'semantic_pairs_{model_type}']['semantic_pairs'].shape[0] > 1:
                    semantic_pairs = results_dict[run_key][f'semantic_pairs_{model_type}']['semantic_pairs'][boolean_mask, :][:, boolean_mask]
                    if compute_cleaned:
                        cleaned_semantic_pairs = results_dict[run_key][f'semantic_pairs_{model_type}']['cleaned_semantic_pairs'][boolean_mask, :][:, boolean_mask]
                else:
                    # deal with single generations
                    assert boolean_mask.item() == True, f"mask: {boolean_mask}"
                    semantic_pairs = results_dict[run_key][f'semantic_pairs_{model_type}']['semantic_pairs']
                    if compute_cleaned:
                        cleaned_semantic_pairs = results_dict[run_key][f'semantic_pairs_{model_type}']['cleaned_semantic_pairs']

                # compute symmetric adjacency matrix
                semantic_pairs = semantic_pairs & semantic_pairs.T
                assert np.array_equal(semantic_pairs, semantic_pairs.T)
                if compute_cleaned:
                    cleaned_semantic_pairs = cleaned_semantic_pairs & cleaned_semantic_pairs.T
                    assert np.array_equal(cleaned_semantic_pairs, cleaned_semantic_pairs.T)

                # compute semantic clusters
                semantic_difference = compute_semantic_clusters(generations=all_considered_generations, 
                                                                        cleaned_semantic_pairs=cleaned_semantic_pairs if compute_cleaned else None,
                                                                        semantic_pairs=semantic_pairs,
                                                                        compute_cleaned=compute_cleaned)
                list_num_semantic_clusters.append(torch.unique(semantic_difference["semantic_clusters"]).shape[0])

                # compute semantic entropy
                weights = boolean_mask[boolean_mask].to(torch.float32) if run_key == "baseline" else torch.nn.functional.normalize(mask[boolean_mask].to(torch.float32), p=1, dim=0)
                my_weights = torch.nn.functional.normalize(results_dict[run_key]['seq_level_impo'][boolean_mask].to(torch.float32), p=1, dim=0)

                semantic_entropy_1 = compute_semantic_entropy(weights=weights,
                                             mc_estimate_over_clusters=False,
                                             neg_log_likelihoods=all_considered_likelihoods, 
                                             semantic_difference=semantic_difference,
                                             epistem_entropy_list = results_dict[run_key]['epistem_entropies'],
                                             compute_cleaned=compute_cleaned)

                semantic_entropy_kuhn = compute_semantic_entropy(weights=weights,
                                                 mc_estimate_over_clusters=True,
                                                 neg_log_likelihoods=all_considered_likelihoods, 
                                                 semantic_difference=semantic_difference,
                                                 epistem_entropy_list = results_dict[run_key]['epistem_entropies'],
                                                 compute_cleaned=compute_cleaned)

                semantic_entropy_ours = compute_semantic_entropy_new(weights=my_weights,
                                             mc_estimate_over_clusters=False,
                                             neg_log_likelihoods=all_considered_likelihoods, 
                                             semantic_difference=semantic_difference,
                                             epistem_entropy_list = results_dict[run_key]['epistem_entropies'],
                                             compute_cleaned=compute_cleaned)

                # If we reach here, all computations succeeded, so add to batch collections
                batch_bleurt_scores.append(list_bleurt[i])
                batch_questions.append(results_dict['question'])
                batch_semantic_entropies.append(semantic_entropy_1)
                batch_semantic_entropies_kuhn.append(semantic_entropy_kuhn)
                batch_semantic_entropies_ours.append(semantic_entropy_ours)

            except Exception as e:
                print(f"Error in batch {batch_start}, instance {i}: {str(e)}")
                print(f"Skipping sample {batch_start + i}")
                # Skip this sample entirely - don't add anything to any collection
                continue

        # ACCUMULATE batch results in final collections
        all_bleurt_scores.extend(batch_bleurt_scores)
        all_batch_questions.extend(batch_questions)
        all_semantic_entropies_final.extend(batch_semantic_entropies)
        all_semantic_entropies_kuhn_final.extend(batch_semantic_entropies_kuhn)
        all_semantic_entropies_ours_final.extend(batch_semantic_entropies_ours)

        # CRITICAL: Clean up batch data immediately
        del list_results_dict, list_bleurt
        del batch_bleurt_scores, batch_semantic_entropies, batch_semantic_entropies_kuhn, batch_semantic_entropies_ours
        gc.collect()  # Force garbage collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear GPU cache

        print(f"Completed batch {batch_start}-{batch_end-1}. Total samples processed so far: {len(all_bleurt_scores)}")
    
    # ===================================================================
    # PHASE 2: VERIFICATION AND SAVING
    # ===================================================================
    print(f"Final verification: Total bleurt scores = {len(all_bleurt_scores)}, Expected = {num_instances}")
    print(f"Final verification: Total entropy samples = {len(all_semantic_entropies_final)}, Expected = {num_instances}")
    print(f"Final verification: Total kuhn entropy samples = {len(all_semantic_entropies_kuhn_final)}, Expected = {num_instances}")
    print(f"Final verification: Total ours entropy samples = {len(all_semantic_entropies_ours_final)}, Expected = {num_instances}")
    
    with open(os.path.join(base_path, f'{run_key}_label_truthful-llama.pkl'), 'wb') as f:
            pkl.dump(all_bleurt_scores,f)
    
    all_semantic_entropies_ = all_semantic_entropies_ours_final + all_semantic_entropies_final + all_semantic_entropies_kuhn_final
    with open(os.path.join(base_path, f'{run_key}_entropies_truthful-llama.pkl'), 'wb') as f:
        pkl.dump(all_semantic_entropies_, f)
            
    print('------------------------------------', max(all_bleurt_scores), min(all_bleurt_scores))
    
    # ===================================================================
    # PHASE 3: AUROC CALCULATION USING ALL SAMPLES
    # ===================================================================
    print(f"Starting AUROC calculations with {len(all_bleurt_scores)} total samples...")
    
    for r, correctness_threshold in enumerate(correctness_threshold_list):
        print(f'Processing threshold: {correctness_threshold} with {len(all_bleurt_scores)} samples')
        try:
            list_correct_labels = torch.logical_not((torch.tensor(all_bleurt_scores) >= correctness_threshold))
            
            aurocs, aurocs_kuhn = {}, {}
            aurocs_ours = {}
            
            # Verify data integrity before AUROC calculation
            print(f"  - Calculating AUROCs for {len(list_correct_labels)} samples")
            print(f"  - all_semantic_entropies_final: {len(all_semantic_entropies_final)} samples")
            print(f"  - all_semantic_entropies_kuhn_final: {len(all_semantic_entropies_kuhn_final)} samples") 
            print(f"  - all_semantic_entropies_ours_final: {len(all_semantic_entropies_ours_final)} samples")
            
            # Check data structure consistency
            for i, item in enumerate(all_semantic_entropies_ours_final):
                if not isinstance(item, dict):
                    print(f"Item {i} is not a dict: {type(item)}")
                    break
            
            # Calculate AUROCs for baseline methods
            for k, key in enumerate(auroc_keys_baseline):
                aurocs[key] = roc_auc_score(list_correct_labels,
                                            [d[key] for d in all_semantic_entropies_final])

                aurocs_kuhn[key] = roc_auc_score(list_correct_labels,
                                                    [d[key] for d in all_semantic_entropies_kuhn_final])
            
            # Calculate AUROCs for our methods
            for k, key in enumerate(auroc_keys):  
                aurocs_ours[key] = roc_auc_score(list_correct_labels,
                                            [d[key] for d in all_semantic_entropies_ours_final])
           
            # Write results to CSV
            with open(csv_filepath, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([
                    run_key,
                    num_total_gens,
                    correctness_threshold,
                    aurocs.get("normalised_semantic_entropy", 11.0), 
                    aurocs.get("unnormalised_semantic_entropy", 11.0),
                    aurocs_ours.get("normalised_semantic_entropy", 11.0),
                    aurocs_ours.get("unnormalised_semantic_entropy", 11.0),
                    aurocs_ours.get("entropy_final", 11.0),
                    aurocs_ours.get("jsd", 11.0),
                    aurocs_ours.get("jsd_min", 11.0),
                    aurocs_ours.get("jsd_max", 11.0),
                    aurocs_ours.get("kl", 11.0),
                    aurocs_ours.get("kl_min", 11.0),
                    aurocs_ours.get("kl_max", 11.0),
                    aurocs_ours.get("entropy_mixture", 11.0),
                    aurocs_ours.get("entropy_mixture_min", 11.0),
                    aurocs_ours.get("entropy_mixture_max", 11.0),
                    aurocs_ours.get("mutual_info", 11.0),
                    aurocs_ours.get("mutual_info_min", 11.0),
                    aurocs_ours.get("mutual_info_max", 11.0),
                    aurocs_ours.get("epkl", 11.0),
                    aurocs_ours.get("epkl_min", 11.0),
                    aurocs_ours.get("epkl_max", 11.0),
                    aurocs_ours.get("cv", 11.0),
                    aurocs_ours.get("cv_min", 11.0),
                    aurocs_ours.get("cv_max", 11.0),
                    aurocs_ours.get("bhattacharyya", 11.0),
                    aurocs_ours.get("bhattacharyya_min", 11.0),
                    aurocs_ours.get("bhattacharyya_max", 11.0),
                    aurocs_ours.get("tvd", 11.0),
                    aurocs_ours.get("tvd_min", 11.0),
                    aurocs_ours.get("tvd_max", 11.0),
                    aurocs_kuhn.get("normalised_semantic_entropy", 11.0),
                    aurocs_kuhn.get("unnormalised_semantic_entropy", 11.0),
                    ])
                    
            print(f"  - Successfully calculated AUROCs for threshold {correctness_threshold}")
            
        except Exception as e:
            print(f'Error at threshold {correctness_threshold}: {str(e)}')
            # traceback.print_exc()

    # Final cleanup
    del all_bleurt_scores, all_semantic_entropies_final, all_semantic_entropies_kuhn_final, all_semantic_entropies_ours_final
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    print(f"Completed processing for run_key: {run_key}")

print("All processing completed!")