import os
import random
import numpy as np
import torch
import math
from tqdm import tqdm
import pickle
from transformers import AutoModelForSequenceClassification, AutoTokenizer, OPTForCausalLM
from accelerate import dispatch_model
from seq_lvl_imp import sentence_importance

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

@torch.no_grad()
def remove_invalid_ids(generation, 
                       invalid_ids):
    for invalid in invalid_ids:
        if invalid in generation:
            generation = generation[:torch.where(generation == invalid)[0][0]]
    return generation


@torch.no_grad()
def clean_generation(generation):
    strings_to_filter_on = ['A:', 'A;', 'answer:',  'Answer:', 'Answers:', 'answers:', 'ANSWER:',
                            'Q:', 'Q;', 'question:', 'Question:', 'Questions:', 'questions:', 'QUESTION:']

    for stop_word in strings_to_filter_on:
        stop_word_index = generation.find(stop_word)
        if stop_word_index != -1:
            generation = generation[:stop_word_index]
    
    generation = generation.strip()
    
    return generation


@torch.no_grad()
def generate_text(args, 
                  model, 
                  tokenizer, 
                  input_ids, 
                  len_prompt, 
                  decoding_method, 
                  device):
    input_ids = input_ids.to(device).reshape(1, -1) if args.dataset == 'trivia_qa' else input_ids.to(device)

    if decoding_method == 'dola':
        generation_ids = model.generate(input_ids,
                                        # num_beams=args.num_beams_most_likely,
                                        num_return_sequences=args.num_return_sequences_most_likely,
                                        do_sample= False,
                                        dola_layers = 'high',
                                        # repetition_penalty = 1.2,
                                        temperature=args.temperature_most_likely,
                                        top_p=args.top_p_most_likely,
                                        max_new_tokens= args.max_length_of_generated_sequence,
                                        eos_token_id=args.eos_token_ids,)

    if decoding_method == "most_likely":
        generation_ids = model.generate(input_ids,
                                        num_beams=args.num_beams_most_likely,
                                        num_return_sequences=args.num_return_sequences_most_likely,
                                        do_sample=args.do_sample_most_likely,
                                        temperature=args.temperature_most_likely,
                                        top_p=args.top_p_most_likely,
                                        # repetition_penalty = 1.2,
                                        max_new_tokens = args.max_length_of_generated_sequence,
                                        eos_token_id=args.eos_token_ids,)
    elif decoding_method == 'baseline':
        generation_ids = model.generate(input_ids,
                                        num_beams=args.num_beams_baseline,
                                        num_beam_groups=args.num_beam_groups_baseline,
                                        diversity_penalty=args.diversity_penalty_baseline,
                                        num_return_sequences=args.num_return_sequences_baseline,
                                        do_sample=args.do_sample_baseline,
                                        temperature=args.temperature_baseline,
                                        top_p=args.top_p_baseline,
                                        max_new_tokens= args.max_length_of_generated_sequence,
                                        eos_token_id=args.eos_token_ids,)
    elif decoding_method == 'sdlg':
        generation_ids = model.generate(input_ids,
                                        num_beams=args.num_beams_sdlg * args.num_return_sequences_sdlg,
                                        num_return_sequences=args.num_return_sequences_sdlg,
                                        do_sample=args.do_sample_sdlg,
                                        temperature=args.temperature_sdlg,
                                        top_p=args.top_p_sdlg,
                                        # repetition_penalty = 1.2,
                                        max_new_tokens =  args.max_length_of_generated_sequence,
                                        eos_token_id=args.eos_token_ids,)

    generation_ids = generation_ids.detach().to('cpu')

    generation_ids_list, generation_text_list, cleaned_generation_ids_list, cleaned_generation_text_list = list(), list(), list(), list()
    
    pad_token_id = tokenizer.pad_token_id

    for i in range(len(generation_ids)):
        
        generation_to_add = generation_ids[i][len_prompt:]
        generation_to_add = generation_to_add[generation_to_add != pad_token_id] # remove pad_token_ids

        generation_to_add = remove_invalid_ids(generation_to_add, args.invalid_ids)
        generation_ids_list.append(generation_to_add)
        generation_text = tokenizer.decode(generation_to_add, skip_special_tokens=True).strip()
        # generation_text = tokenizer.decode(generation_to_add[0], skip_special_tokens=True).strip()
        generation_text_list.append(generation_text)

        cleaned_generation_text = clean_generation(generation_text)
        cleaned_generation_text_list.append(cleaned_generation_text)
        cleaned_generation_ids_list.append(generation_to_add if cleaned_generation_text == generation_text else \
            tokenizer.encode(cleaned_generation_text, add_special_tokens=False, return_tensors='pt')[0])

    return {
        'generation_ids': generation_ids_list,
        'generation_text': generation_text_list,

        'cleaned_generation_ids': cleaned_generation_ids_list,
        'cleaned_generation_text': cleaned_generation_text_list,
    }


@torch.no_grad()
def prepare_generated_text(generation_ids, 
                           generation_text, 
                           cleaned_generation_ids, 
                           cleaned_generation_text, 
                           **kwargs):
    list_generation_dicts = []


    for i in range(len(generation_ids)):
        generation_dict = {
            'generation_ids': [generation_ids[i]],
            'generation_text': [generation_text[i]],
            'cleaned_generation_ids': [cleaned_generation_ids[i]],
            'cleaned_generation_text': [cleaned_generation_text[i]],
        }
            
        # add other kwargs (word_idx, new_token, token_likelihood)
        for kwarg in kwargs.keys():
            generation_dict[kwarg] = kwargs[kwarg]
        
        list_generation_dicts.append(generation_dict)

    return list_generation_dicts


@torch.no_grad()
def compute_correctness(args, 
                        reference_answers, 
                        incorrect_answers, 
                        most_likely_generation_text, 
                        exact_match_metric=None, 
                        rouge=None, 
                        bleurt=None):
    correctness_dict = {}

    if exact_match_metric is not None:
        exact_match = 0.0
        for answer in reference_answers:
            results = exact_match_metric.compute(predictions=[most_likely_generation_text],
                                                references=[answer],
                                                ignore_case=True,
                                                ignore_punctuation=True)
            exact_match = max(results['exact_match'], exact_match)
        
        correctness_dict['exact_match'] = exact_match
            
    if rouge is not None:
        rouge1, rouge2, rougeL = 0.0, 0.0, 0.0
        for answer in reference_answers:
            rouge_results = rouge.compute(predictions=[most_likely_generation_text], 
                                        references=[answer])
            rouge1 = max(rouge_results['rouge1'].item(), rouge1)
            rouge2 = max(rouge_results['rouge2'].item(), rouge2)
            rougeL = max(rouge_results['rougeL'].item(), rougeL)

        incorrect_rouge1, incorrect_rouge2, incorrect_rougeL = 0.0, 0.0, 0.0
        for incorrect_answer in incorrect_answers:
            rouge_results = rouge.compute(predictions=[most_likely_generation_text], 
                                        references=[incorrect_answer])
            incorrect_rouge1 = max(rouge_results['rouge1'].item(), incorrect_rouge1)
            incorrect_rouge2 = max(rouge_results['rouge2'].item(), incorrect_rouge2)
            incorrect_rougeL = max(rouge_results['rougeL'].item(), incorrect_rougeL)

        if len(incorrect_answers) != 0:
            correctness_dict['rouge1-diff'] = rouge1 - incorrect_rouge1
            correctness_dict['rouge2-diff'] = rouge2 - incorrect_rouge2
            correctness_dict['rougeL-diff'] = rougeL - incorrect_rougeL
        correctness_dict['rouge1'] = rouge1
        correctness_dict['rouge2'] = rouge2
        correctness_dict['rougeL'] = rougeL

    if bleurt is not None:
        scores_true = max(bleurt.compute(predictions=[most_likely_generation_text] * len(reference_answers), references=reference_answers)['scores'])

        correctness_dict['bleurt'] = scores_true
        if len(incorrect_answers) != 0:
            scores_false = max(bleurt.compute(predictions=[most_likely_generation_text] * len(incorrect_answers), 
                                              references=incorrect_answers)['scores'])
            correctness_dict['bleurt-diff'] = scores_true - scores_false

    return correctness_dict


@torch.no_grad()
def compute_likelihood(prompt, 
                       generation, 
                       model, 
                       device, 
                       compute_cleaned=False, 
                       store_logits=True):
        
    # Note: This computation of NLL follows the impementation of Kuhn et al. (2023)
    list_average_neg_log_likelihoods, list_neg_log_likelihood = [], []
    list_cleaned_average_neg_log_likelihood, list_cleaned_neg_log_likelihood = [], []
    list_generation_logits, list_cleaned_generation_logits = [], []

    # iterate over all generations -> "generation_ids" is list of generations
    for i in range(len(generation['generation_ids'])): 

        generation_ids = generation['generation_ids'][i]

        generation_input = torch.hstack([prompt, generation_ids]).to(device)

        target_ids = generation_input.clone()
        target_ids[:len(prompt)] = -100
        model_output = model(torch.reshape(generation_input, (1, -1)), labels=target_ids)
        average_neg_log_likelihood = model_output['loss'].item()
        neg_log_likelihood = average_neg_log_likelihood * (len(generation_ids))

        list_average_neg_log_likelihoods.append(average_neg_log_likelihood)
        list_neg_log_likelihood.append(neg_log_likelihood)

        # compute logits
        if store_logits:
            generation_logits = model_output["logits"][0, len(prompt)-1:-1, :].detach().to('cpu') 
            # shift by 1 since token probs at last token of prompt already belong to first token of generation
            list_generation_logits.append(generation_logits)
            assert generation_logits.shape[0] == generation_ids.shape[0]

        if compute_cleaned:

            cleaned_generation_ids = generation['cleaned_generation_ids'][i]

            if torch.equal(cleaned_generation_ids, generation_ids) or \
                generation['cleaned_generation_text'][i] == generation['generation_text'][i]:
                cleaned_average_neg_log_likelihood = average_neg_log_likelihood
                cleaned_neg_log_likelihood = neg_log_likelihood
                if store_logits:
                    cleaned_generation_logits = generation_logits
            elif generation['cleaned_generation_text'][i] == '':
                # Note: setting nll to ngative infinity (zero likelihood) if cleaned generation is empty
                cleaned_average_neg_log_likelihood = float('-inf')
                cleaned_neg_log_likelihood = float('-inf')
                if store_logits:
                    cleaned_generation_logits = []
            else:
                # Note: computation of NNL follows tutorial: https://huggingface.co/docs/transformers/perplexity
                generation_input = torch.hstack([prompt, cleaned_generation_ids]).to(device)
                target_ids = generation_input.clone()
                target_ids[:len(prompt)] = -100
                model_output = model(torch.reshape(generation_input, (1, -1)), labels=target_ids)
                cleaned_average_neg_log_likelihood = model_output['loss'].item()
                cleaned_neg_log_likelihood = cleaned_average_neg_log_likelihood * (len(cleaned_generation_ids))

                # compute logits
                if store_logits:
                    cleaned_generation_logits = model_output["logits"][0, len(prompt)-1:-1, :].to('cpu')
            
            if store_logits:
                list_cleaned_generation_logits.append(cleaned_generation_logits.detach().to('cpu'))
            list_cleaned_average_neg_log_likelihood.append(cleaned_average_neg_log_likelihood)
            list_cleaned_neg_log_likelihood.append(cleaned_neg_log_likelihood)

    return {
        'average_neg_log_likelihood': list_average_neg_log_likelihoods,
        'neg_log_likelihood': list_neg_log_likelihood,
        'generation_logits': list_generation_logits,
        
        'cleaned_average_neg_log_likelihood': list_cleaned_average_neg_log_likelihood,
        'cleaned_neg_log_likelihood': list_cleaned_neg_log_likelihood,
        'cleaned_generation_logits': list_cleaned_generation_logits,
    }


@torch.no_grad()
def prepare_likelihood(average_neg_log_likelihood, 
                       neg_log_likelihood,
                       generation_logits=None, 
                       cleaned_average_neg_log_likelihood=None, 
                       cleaned_neg_log_likelihood=None, 
                       cleaned_generation_logits=None, 
                       compute_cleaned=False,
                       store_logits=True):
    list_likelihood_dicts = []
    if isinstance(average_neg_log_likelihood, list):
        for i in range(len(average_neg_log_likelihood)):
            
            likelihood_dict = {
                'average_neg_log_likelihood': [average_neg_log_likelihood[i]],
                'neg_log_likelihood': [neg_log_likelihood[i]],
                'generation_logits': [generation_logits[i]] if store_logits else [],
                'cleaned_average_neg_log_likelihood': [cleaned_average_neg_log_likelihood[i]] if compute_cleaned else [],
                'cleaned_neg_log_likelihood': [cleaned_neg_log_likelihood[i]] if compute_cleaned else [],
                'cleaned_generation_logits': [cleaned_generation_logits[i]] if compute_cleaned and store_logits else [],
                
            }

            list_likelihood_dicts.append(likelihood_dict)
    else:
        assert not isinstance(neg_log_likelihood, list) and \
               not isinstance(cleaned_neg_log_likelihood, list) and \
               not isinstance(cleaned_average_neg_log_likelihood, list)
        
        likelihood_dict = {
            'average_neg_log_likelihood': [average_neg_log_likelihood],
            'neg_log_likelihood': [neg_log_likelihood],
            'generation_logits': [generation_logits] if store_logits else [],
            'cleaned_average_neg_log_likelihood': [cleaned_average_neg_log_likelihood] if compute_cleaned else [],
            'cleaned_neg_log_likelihood': [cleaned_neg_log_likelihood] if compute_cleaned else [],
            'cleaned_generation_logits': [cleaned_generation_logits] if compute_cleaned and store_logits else [],
        }

        list_likelihood_dicts.append(likelihood_dict)

    return list_likelihood_dicts

def prepare_results(num_samples, 
                    run_key, 
                    metric=None, 
                    start_sample_id=0, 
                    base_path='results'):

    list_results_dict = []
    dataset_size = 0
    list_correct_labels = []

    for i in tqdm(range(num_samples), total=num_samples):
        if i < start_sample_id:
            continue
        
        try:
            with open(os.path.join(base_path, f'results_dict_{i}.pkl'), 'rb') as f:
                results_dict = pickle.load(f)
        except:
            continue

        if len(results_dict[run_key]['generations'][0]['generation_ids'][0]) == 0:
            continue

        prepared_generation = []
        prepared_likelihoods = []
        for generations, likelihoods in zip(results_dict[run_key]['generations'], results_dict[run_key]['likelihoods']):
            prepared_generation += prepare_generated_text(**generations)
            prepared_likelihoods += prepare_likelihood(**likelihoods)
        results_dict[run_key]['generations'] = prepared_generation
        results_dict[run_key]['likelihoods'] = prepared_likelihoods

            
        if metric is not None:
            list_correct_labels.append(results_dict["correctness_dict"][metric])

        list_results_dict.append(results_dict)
        dataset_size += 1
        
        del prepared_generation, prepared_likelihoods, results_dict

    return list_results_dict, list_correct_labels, dataset_size

# ---------------------------------------------------------------------------------------------------------------------------------------

@torch.no_grad()
def compute_semantic_pairs(generations, 
                           deberta_tokenizer, 
                           deberta_model, 
                           question, 
                           device, 
                           compute_cleaned=False):
    
    semantic_pairs = [np.zeros(shape=(len(generations), len(generations)), dtype=bool),
                      np.zeros(shape=(len(generations), len(generations)), dtype=bool)]

    for i, generation_i  in enumerate(generations):
        for j, generation_j in enumerate(generations):

            if i == j:
                semantic_pairs[0][i, j] = True
                semantic_pairs[1][i, j] = True
                continue

            list_iterations = ['generation_text', 'cleaned_generation_text'] if compute_cleaned else ['generation_text']
            for k, genration_key in enumerate(list_iterations):

                # Note: if cleaned generation text is identiacal to generation text, we don't have to check again
                if genration_key == 'cleaned_generation_text' and generation_i['generation_text'][0] == generation_i['cleaned_generation_text'][0] \
                    and generation_j['generation_text'][0] == generation_j['cleaned_generation_text'][0]:
                    semantic_pairs[1][i, j] = semantic_pairs[0][i, j]
                else:
                    
                    if generation_i[genration_key][0].lower() == generation_j[genration_key][0].lower():
                        semantic_pairs[k][i, j] = True
                    else:
                        qa_i = question + ' ' + generation_i[genration_key][0]
                        qa_j = question + ' ' + generation_j[genration_key][0]

                        input_sequence = qa_i + ' [SEP] ' + qa_j
                        encoded_input = deberta_tokenizer.encode(input_sequence, padding=True)
                        prediction = deberta_model(torch.tensor([encoded_input], device=device))['logits']
                        predicted_label = torch.argmax(prediction, dim=1)

                        semantic_pairs[k][i, j] = True if predicted_label != 0 else False
            
    return {
        'semantic_pairs': semantic_pairs[0],
        'cleaned_semantic_pairs': semantic_pairs[1] if compute_cleaned else [],
    }


@torch.no_grad()
def compute_batched_semantic_pairs(generations, 
                                   deberta_tokenizer, 
                                   deberta_model, 
                                   question, 
                                   device, 
                                   compute_cleaned=False, 
                                   batch_size=32):
    
    semantic_pairs = [np.zeros(shape=(len(generations), len(generations)), dtype=bool),
                      np.zeros(shape=(len(generations), len(generations)), dtype=bool)]

    list_iterations = ['generation_text', 'cleaned_generation_text'] if compute_cleaned else ['generation_text']
    for k, genration_key in enumerate(list_iterations):

        if genration_key == 'cleaned_generation_text':
            if False not in [g['generation_text'][0] == g['cleaned_generation_text'][0] for g in generations]:
                semantic_pairs[1] = semantic_pairs[0]
                break

        num_batches = range(math.ceil(len(generations) / batch_size))

        for batch_i in num_batches:
            for batch_j in num_batches:
                model_input =  []
                # ensemble batch
                for generation_i in generations[batch_size*batch_i:batch_size*(batch_i+1)]:
                    for generation_j in generations[batch_size*batch_j:batch_size*(batch_j+1)]:

                        qa_i = question + ' ' + generation_i[genration_key][0]
                        qa_j = question + ' ' + generation_j[genration_key][0]

                        input_sequence = qa_i + ' [SEP] ' + qa_j
                        model_input.append(input_sequence)

                # compute predictions
                try:
                    # try batched input
                    encoded_input = deberta_tokenizer(model_input, return_tensors='pt', padding=True).to(device)
                    prediction = deberta_model(**encoded_input)['logits']
                    predicted_labels = torch.argmax(prediction, dim=1)
                except:
                    # Note: if batched input is OOM, compute predictions for each pair individually
                    predicted_labels = []
                    for input_sequence in model_input:
                        encoded_input = deberta_tokenizer.encode(input_sequence, padding=True)
                        encoded_input = torch.tensor([encoded_input], device=device)
                        prediction = deberta_model(encoded_input)['logits']
                        predicted_labels.append(torch.argmax(prediction, dim=1).item())

                # assign values to semantic_pairs
                index = 0
                for i in range(len(generations[batch_size*batch_i:batch_size*(batch_i+1)])):
                    for j in range(len(generations[batch_size*batch_j:batch_size*(batch_j+1)])):
                        semantic_pairs[k][batch_size*batch_i + i, batch_size*batch_j + j] = True if predicted_labels[index] != 0 else False
                        index += 1
    
    return {
        'semantic_pairs': semantic_pairs[0],
        'cleaned_semantic_pairs': semantic_pairs[1] if compute_cleaned else [],
    }


@torch.no_grad()
def compute_semantic_clusters(generations, 
                              semantic_pairs, 
                              cleaned_semantic_pairs, 
                              compute_cleaned=False):

    semantic_clusters = [list(range(0, len(generations))), list(range(0, len(generations)))]
    for i, generation_i  in enumerate(generations):
        for j, generation_j in enumerate(generations):

            if j <= i:
                continue

            list_iterations = ['generation_text', 'cleaned_generation_text'] if compute_cleaned else ['generation_text']
            for k, genration_key in enumerate(list_iterations):

                # [cleaned] if cleaned generation text is identiacal to generation text, we don't have to check again
                if genration_key == 'cleaned_generation_text' and generation_i['generation_text'][0] == generation_i['cleaned_generation_text'][0] \
                    and generation_j['generation_text'][0] == generation_j['cleaned_generation_text'][0]:
                    semantic_clusters[1] = semantic_clusters[0]
                    break

                # [all] if the clusters are already the same, we don't have to check again
                if semantic_clusters[k][j] == semantic_clusters[k][i]:
                    continue
                # [all] if generation text is identical, directly assign to same cluster
                elif generation_i[genration_key][0].lower() == generation_j[genration_key][0].lower():
                    semantic_clusters[k][j] = semantic_clusters[k][i]
                # [all] otherwise, check semantic_pairs
                else:
                    if genration_key == 'generation_text':
                        if semantic_pairs[i, j] == True and semantic_pairs[j, i] == True:
                            # Note: equivalent to: if predicted_label != 0 and reverse_predicted_label != 0
                            semantic_clusters[k][j] = semantic_clusters[k][i]
                    else:
                        if cleaned_semantic_pairs[i, j] == True and cleaned_semantic_pairs[j, i] == True:
                            semantic_clusters[k][j] = semantic_clusters[k][i]
    
    return {
        'semantic_clusters': torch.tensor(semantic_clusters[0]),
        'cleaned_semantic_clusters': torch.tensor(semantic_clusters[1]) if compute_cleaned else torch.tensor([]),
        }


@torch.no_grad()
def compute_semantic_entropy_new(weights,
                            mc_estimate_over_clusters,
                            neg_log_likelihoods,
                            semantic_difference,
                            epistem_entropy_list,
                            compute_cleaned=False):
    results = []
    gamma = 1e-9

    list_iterations = ['', 'cleaned_'] if compute_cleaned else ['']

    for cleaned in list_iterations:
        for nll_key in ["average_neg_log_likelihood", "neg_log_likelihood"]:
            for d in neg_log_likelihoods:
                assert len(d[cleaned+nll_key]) == 1, "only single likelihoods supported. prepare likelihoods!"

            # compute log(p(y|x,w)): converting NLL to LL, and handling NaN and negative infinity values
            log_likelihoods = torch.tensor(
                [-1e12 if math.isnan(d[cleaned+nll_key][0]) or d[cleaned+nll_key][0] == float('-inf') else -d[cleaned+nll_key][0] for d in neg_log_likelihoods]
            ) 
            assert torch.all(log_likelihoods <= 0), f"likelihood bigger than 1!"

            # scale LL by weights (all ones for baseline)
            log_likelihoods += torch.log(weights)

            aggregated_log_likelihoods = []
            aggregated_weights = []
            
            # compute p(c|x,w): aggregate LL over clusters
            for semantic_set_id in torch.unique(semantic_difference[cleaned+'semantic_clusters']):
                aggregated_log_likelihoods.append(torch.logsumexp(log_likelihoods[semantic_difference[cleaned+'semantic_clusters'] == semantic_set_id], dim=0))
                aggregated_weights.append(torch.sum(weights[semantic_difference[cleaned+'semantic_clusters'] == semantic_set_id], dim=0))

            aggregated_log_likelihoods = torch.tensor(aggregated_log_likelihoods)
            aggregated_weights = torch.tensor(aggregated_weights)

            # softmax: normalizing + transforming from log space to probability space 
            aggregated_normalized_likelihoods = torch.softmax(aggregated_log_likelihoods, dim=0)

            if mc_estimate_over_clusters:
                entropy = - torch.sum(aggregated_log_likelihoods, dim=0) / torch.tensor(aggregated_log_likelihoods.shape[0])
            else:
                entropy = - torch.sum(aggregated_log_likelihoods * aggregated_normalized_likelihoods, dim=0)

            assert not torch.isinf(entropy).any(), \
                f"semantic_entropy is inf. entropy: {entropy}, aggregated_log_likelihoods: {aggregated_log_likelihoods}, log_likelihoods: {log_likelihoods}"
            assert not torch.isnan(entropy).any(), \
                f"semantic_entropy is nan. entropy: {entropy}, aggregated_log_likelihoods: {aggregated_log_likelihoods}, log_likelihoods: {log_likelihoods}"
            
            results.append(entropy.item())
    
    # Calculate average for each metric from epistem_entropy_list
    avg_metrics = {}
    
    # Initialize dictionaries to store sums and counts for each metric
    if epistem_entropy_list and len(epistem_entropy_list) > 0 and isinstance(epistem_entropy_list[0], dict):
        # Get all metric keys from the first dictionary
        all_metrics = epistem_entropy_list[0].keys()
        
        # Calculate sum for each metric
        for metric in all_metrics:
            # Filter out any items where the metric might be missing
            valid_values = [item[metric] for item in epistem_entropy_list if metric in item]
            if valid_values:
                avg_metrics[f"{metric}"] = sum(valid_values) / len(valid_values)
    
    # Build the final result dictionary
    result_dict = {
        "normalised_semantic_entropy": results[0],
        "unnormalised_semantic_entropy": results[1],
    }
    
    # Add cleaned results if computed
    if compute_cleaned:
        result_dict.update({
            "cleaned_normalised_semantic_entropy": results[2],
            "cleaned_unnormalised_semantic_entropy": results[3],
        })
    
    # Add all averaged metrics from epistem_entropy_list
    result_dict.update(avg_metrics)
    
    return result_dict

@torch.no_grad()
def compute_semantic_entropy(weights, 
                             mc_estimate_over_clusters, 
                             neg_log_likelihoods, 
                             semantic_difference,
                             epistem_entropy_list,
                             compute_cleaned=False):
    results = []
    gamma = 1e-9

    list_iterations = ['', 'cleaned_'] if compute_cleaned else ['']

    for cleaned in list_iterations:
        for nll_key in ["average_neg_log_likelihood", "neg_log_likelihood"]:

            for d in neg_log_likelihoods:
                assert len(d[cleaned+nll_key]) == 1, "only single likelihoods supported. prepare likelihoods!"

            # compute log(p(y|x,w)): converting NLL to LL, and handling NaN and negative infinity values
            log_likelihoods = torch.tensor(
                [-1e12 if math.isnan(d[cleaned+nll_key][0]) or d[cleaned+nll_key][0] == float('-inf') else -d[cleaned+nll_key][0] for d in neg_log_likelihoods]
            ) 
            assert torch.all(log_likelihoods <= 0), f"likelihood bigger than 1!"

            # scale LL by weights (all ones for baseline)
            log_likelihoods += torch.log(weights)

            aggregated_log_likelihoods = []
            aggregated_weights = []
            
            # compute p(c|x,w): aggregate LL over clusters
            for semantic_set_id in torch.unique(semantic_difference[cleaned+'semantic_clusters']):
                aggregated_log_likelihoods.append(torch.logsumexp(log_likelihoods[semantic_difference[cleaned+'semantic_clusters'] == semantic_set_id], dim=0))
                aggregated_weights.append(torch.sum(weights[semantic_difference[cleaned+'semantic_clusters'] == semantic_set_id], dim=0))

            aggregated_log_likelihoods = torch.tensor(aggregated_log_likelihoods)
            aggregated_weights = torch.tensor(aggregated_weights)

            # softmax: normalizing + transforming from log space to probability space 
            aggregated_normalized_likelihoods = torch.softmax(aggregated_log_likelihoods, dim=0)

            if mc_estimate_over_clusters:
                entropy = - torch.sum(aggregated_log_likelihoods, dim=0) / torch.tensor(aggregated_log_likelihoods.shape[0])
            else:
                entropy = - torch.sum(aggregated_log_likelihoods * aggregated_normalized_likelihoods, dim=0)

            assert not torch.isinf(entropy).any(), \
                f"semantic_entropy is inf. entropy: {entropy}, aggregated_log_likelihoods: {aggregated_log_likelihoods}, log_likelihoods: {log_likelihoods}"
            assert not torch.isnan(entropy).any(), \
                f"semantic_entropy is nan. entropy: {entropy}, aggregated_log_likelihoods: {aggregated_log_likelihoods}, log_likelihoods: {log_likelihoods}"
            results.append(entropy.item())
        
    return {
        "normalised_semantic_entropy": results[0],
        "unnormalised_semantic_entropy": results[1],  
        "cleaned_normalised_semantic_entropy": results[2] if compute_cleaned else [],
        "cleaned_unnormalised_semantic_entropy": results[3] if compute_cleaned else [],
    }

# @torch.no_grad()
def compute_semantic_paris_new(base_path,
                           model_type, 
                           deberta_tokenizer, 
                           deberta_model, 
                           num_instances, 
                           device,
                           nli_model_sent = 'sentence-transformers/sentence-t5-large',
                           offset= 0
                          ):
    
    for method in ['sdlg', 'baseline']: 
    # for method in ['sdlg']: 
        print('method: ', method)
        removed_sample_ids = []
        for i in tqdm(range(0, num_instances)):
            i = i + offset
            try:
                with open(os.path.join(base_path, f'results_dict_{i}.pkl'), 'rb') as f:
                    results_dict = pickle.load(f)
            except:
                if i not in removed_sample_ids:
                    removed_sample_ids.append(i)
                continue
            if len(results_dict[method]['generations']) == 0 or len(results_dict[method]['generations'][0]['generation_ids'][0]) == 0:
                if i not in removed_sample_ids:
                    removed_sample_ids.append(i)
                continue
            prepared_generation = []
            prepared_likelihoods = []
            for generations, likelihoods in zip(results_dict[method]['generations'], results_dict[method]['likelihoods']):
                prepared_generation += prepare_generated_text(**generations)
                prepared_likelihoods += prepare_likelihood(**likelihoods)
            results_dict[method]['generations'] = prepared_generation
            results_dict[method]['likelihoods'] = prepared_likelihoods
            sdlg_gen_list = [x['generation_text'][0] for x in results_dict[method]['generations']]
            
            if (f'semantic_pairs_{model_type}' not in results_dict[method].keys() or 
                results_dict[method][f'semantic_pairs_{model_type}']['semantic_pairs'].shape[0] != len(results_dict[method]['generations'])):
                
            pairs = compute_batched_semantic_pairs(generations=results_dict[method]["generations"],
                                                   deberta_tokenizer=deberta_tokenizer, 
                                                   deberta_model=deberta_model, 
                                                   question=results_dict['question'], 
                                                   device=device,
                                                   compute_cleaned=False,
                                                   batch_size=32)

            results_dict[method][f'semantic_pairs_{model_type}'] = pairs

            target_pairs = torch.tensor(pairs['semantic_pairs'].astype(int), dtype=torch.float32, device='cuda:0')
            seq_level_imp = sentence_importance(sdlg_gen_list[0], sdlg_gen_list[1:], nli_model_sent, target_pairs)

            # NEW: Weight down by generation likelihood
            # Extract numerical likelihood values from the prepared likelihood dictionaries
            # Using average_neg_log_likelihood and converting to positive likelihood
            likelihood_values = []
            for likelihood_dict in prepared_likelihoods:
                avg_neg_log_likelihood = likelihood_dict['average_neg_log_likelihood'][0]
                # Convert negative log-likelihood to likelihood: likelihood = exp(-neg_log_likelihood)
                likelihood = torch.exp(torch.tensor(-avg_neg_log_likelihood))
                likelihood_values.append(likelihood)

            # Convert to tensor and ensure same device as seq_level_imp
            generation_likelihoods = torch.stack(likelihood_values).to(seq_level_imp.device)

          # Add small epsilon to avoid division by zero
            # epsilon = 1e-8
            # generation_likelihoods = generation_likelihoods + epsilon

            imp_weight =  generation_likelihoods / seq_level_imp 
            imp_weight = seq_level_imp.detach().to('cpu')
            results_dict[method]['seq_level_impo'] = imp_weight

            with open(os.path.join(base_path, f'results_dict_{i}.pkl'), 'wb') as f:
                pickle.dump(results_dict, f)