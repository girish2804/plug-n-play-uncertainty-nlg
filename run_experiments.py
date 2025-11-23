import os
# os.environ["HF_HOME"] = ...                     # set accordingly
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # set accordingly

import pickle
import yaml

import numpy as np
import datasets
import evaluate
import torch
from tqdm import tqdm

from torch.utils.data import DataLoader, Dataset

from args import Args
from utils import seed_everything, compute_correctness, compute_semantic_paris_new
from utils import generate_text, prepare_generated_text, compute_likelihood, prepare_likelihood
from sdlg import generate_semantically_diverse_output_sequences
from transformers import AutoTokenizer,AutoModelForSequenceClassification,AutoModelForCausalLM, OPTForCausalLM
from datasets import load_dataset
import traceback
# from bleurt import score as bleurt_score

CUDA_ID_LLM = 0                 # set accordingly
CUDA_ID_DEBERTA = CUDA_ID_LLM   # set accordingly

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
def encode(examples, few_shot):
    # return tokenizer(f'''fact: {examples['fact1']}\n Considering the given fact. Answer the question by continuing the sentence. Provide your reasoning along with your answer.\n '', truncation=False, padding=False)
    # return tokenizer(f'''Question: {examples['question']}\n Choices: {', '.join(examples['choices']['text'])}\n Answer:''', truncation=False, padding=False)
    return tokenizer(f'''Q: {examples['question']} A:''', truncation=False, padding=False)
    # return tokenizer(examples['knowledge'] + ' Q: ' + examples['question'] + ' A:', truncation=False, padding=False)
    # return tokenizer(few_shot + ' Q: ' + examples['question'] + ' A:', truncation=False, padding=False)

def encode_and_format_dataset(dataset, few_shot):
    dataset = dataset.map(lambda examples: encode(examples, few_shot), batched=False, load_from_cache_file=False)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'], output_all_columns=True)
    return dataset

def get_results(args, base_path, llm_model, tokenizer, device_llm, 
                                deberta_model, deberta_tokenizer, device_deberta, dataset):
    
    # existing setup code...
    squad_metric = evaluate.load("squad")
    rouge = evaluate.load('rouge')
    exact_match_metric = evaluate.load("exact_match")
    bleurt = evaluate.load("bleurt")
    
    deberta_embeddings = deberta_model.deberta.embeddings.word_embeddings(
        torch.tensor([list(range(0, deberta_tokenizer.vocab_size))]).to(device_deberta)
    ).squeeze().detach()
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    cnt = 0
    
    for b, batch in enumerate(tqdm(dataloader)):
        try:
            # Your existing processing code...
            prompt = batch['input_ids'][0].to('cpu')
            question = batch["question"][0]
            # question =batch["question_stem"][0]
            detached_input_ids = batch['input_ids'].detach().to('cpu')
            
            results_dict = {'input_ids': detached_input_ids,
                           'question': question,
                           'correctness_dict': {},
                           'dola_sdlg': {'generations': [],
                                   'likelihoods': [],
                                   'epistem_entropies':[]},
                           'sdlg': {'generations': [],
                                   'likelihoods': [],
                                   'epistem_entropies':[]},
                           'baseline': {'generations': [],
                                       'likelihoods': []}
                           }
            
            most_likely_generation = generate_text(args=args, 
                                     model=llm_model, 
                                     tokenizer=tokenizer, 
                                     input_ids=batch['input_ids'], 
                                     len_prompt=len(prompt), 
                                     decoding_method='most_likely', 
                                     device=device_llm)
            
            # most_likely_generation_dola = generate_text(args=args, 
            #                                      model=llm_model, 
            #                                      tokenizer=tokenizer, 
            #                                      input_ids=batch['input_ids'], 
            #                                      len_prompt=len(prompt), 
            #                                      decoding_method='dola', 
            #                                      device=device_llm)
            
            reference_answers = batch['best_answer']
            # reference_answers = batch['solution']
            # i = 0
            # if batch['answerKey'][0] == 'A':
            #     i = 0
            # elif batch['answerKey'][0] == 'B':
            #     i = 1
            # elif batch['answerKey'][0] == 'C':
            #     i = 2
            # elif batch['answerKey'][0] == 'D':
            #     i =3
            # else:
            #     print('answer key error', batch['answerKey'][0])
            # reference_answers = list(batch['choices']['text'][i]) #+ ', ' + batch['fact1'][0]
            # reference_answers = batch['correct_answer']  #+ ', ' + batch['support'][0]
   
            
            incorrect_answers = []
            
            correctness_dict = compute_correctness(args=args, 
                                                 reference_answers=reference_answers, 
                                                 incorrect_answers=incorrect_answers, 
                                                 most_likely_generation_text=most_likely_generation['generation_text'][0], 
                                                 exact_match_metric=exact_match_metric, 
                                                 rouge=rouge,
                                                 bleurt=bleurt)
            
            results_dict['correctness_dict'] = correctness_dict
            
            # compute likelihood
            most_likely_generation_likelihoods = compute_likelihood(prompt=prompt, 
                                                                    generation=most_likely_generation, 
                                                                    model=llm_model, 
                                                                    device=device_llm, 
                                                                    compute_cleaned=args.compute_cleaned, 
                                                                    store_logits=args.store_logits)
            # most_likely_generation_likelihoods_dola = compute_likelihood(prompt=prompt, 
            #                                                         generation=most_likely_generation_dola, 
            #                                                         model=llm_model, 
            #                                                         device=device_llm, 
            #                                                         compute_cleaned=args.compute_cleaned, 
            #                                                         store_logits=args.store_logits)
            ### (2) sample addtional output sequences
            
            # # sdlg with dola (2.0)
            # results_dict['dola_sdlg']['generations'].append(most_likely_generation_dola)
            # results_dict['dola_sdlg']['likelihoods'].append(most_likely_generation_likelihoods_dola)
            
            # results_dict = generate_semantically_diverse_output_sequences(results_dict=results_dict, 
            #                                                             deberta_model=deberta_model, 
            #                                                             deberta_tokenizer=deberta_tokenizer, 
            #                                                             device_deberta=device_deberta,
            #                                                             deberta_embeddings=deberta_embeddings,
            #                                                             model=llm_model, 
            #                                                             tokenizer=tokenizer, 
            #                                                             device_llm=device_llm,
            #                                                             input_ids=batch['input_ids'], 
            #                                                             prompt=prompt,
            #                                                             question=question, 
            #                                                             initial_generation=most_likely_generation_dola,
            #                                                             initial_likelihood=most_likely_generation_likelihoods_dola,
            #                                                             key = 'dola_sdlg',
            #                                                             args=args) 

            # (2.1) SDLG
            results_dict['sdlg']['generations'].append(most_likely_generation)
            results_dict['sdlg']['likelihoods'].append(most_likely_generation_likelihoods)
            
            results_dict = generate_semantically_diverse_output_sequences(results_dict=results_dict, 
                                                                        deberta_model=deberta_model, 
                                                                        deberta_tokenizer=deberta_tokenizer, 
                                                                        device_deberta=device_deberta,
                                                                        deberta_embeddings=deberta_embeddings,
                                                                        model=llm_model, 
                                                                        tokenizer=tokenizer, 
                                                                        device_llm=device_llm,
                                                                        input_ids=batch['input_ids'], 
                                                                        prompt=prompt,
                                                                        question=question, 
                                                                        initial_generation=most_likely_generation,
                                                                        initial_likelihood=most_likely_generation_likelihoods,
                                                                        key = 'sdlg',
                                                                        args=args)      

            # (2.2) MS
            assert args.num_total_generations % args.num_return_sequences_baseline == 0
            results_dict['baseline']['generations'].append(most_likely_generation)
            results_dict['baseline']['likelihoods'].append(most_likely_generation_likelihoods)

            for i in range(int(args.num_total_generations / args.num_return_sequences_baseline)):
                baseline_generation = generate_text(args=args, 
                                                    model=llm_model, 
                                                    tokenizer=tokenizer, 
                                                    input_ids=batch['input_ids'], 
                                                    len_prompt=len(prompt), 
                                                    decoding_method='baseline', 
                                                    device=device_llm)

                results_dict['baseline']['generations'].append(baseline_generation)
                results_dict['baseline']['likelihoods'].append(compute_likelihood(prompt=prompt, 
                                                                                generation=baseline_generation, 
                                                                                model=llm_model, 
                                                                                device=device_llm, 
                                                                                compute_cleaned=args.compute_cleaned, 
                                                                                store_logits=args.store_logits))
            for x in results_dict['sdlg']['generations']:
                print(x['generation_text'])
            print('sdlg^-----------------------------------')
            
            k = int(batch['question_id'])
            
            with open(os.path.join(base_path, f'results_dict_{k}.pkl'), 'wb') as outfile:
                pickle.dump(results_dict, outfile)
                      
        except Exception as error:
            cnt += 1
            print("Error", cnt)
            traceback.print_exc()


if __name__ == '__main__':

    args = Args()
    args.run_id = 'truthful-llama'
    base_path = os.path.join('result', args.run_id)
    if not os.path.exists(base_path):
        os.makedirs(base_path, exist_ok=True)
    
    if os.path.exists(os.path.join(base_path, f'config.yaml')):
        with open(os.path.join(base_path, f'config.yaml'), 'r') as file:
            existing_args = yaml.load(file, Loader=yaml.FullLoader)
        changes = False

        for k, v in existing_args.items():
            if k not in args.__dict__:
                print(f"new arg: {k}")
                changes = True
            elif v != args.__dict__[k]:
                print(f"arg {k} changed from {v} to {args.__dict__[k]}")
                changes = True
        if changes:
            exit()
        print("continuing existing run ...")
    else:
        print("starting new run ...")

    # save args
    args.args_to_yaml(base_path)

    print("run_id", args.run_id)

    seed_everything(seed=args.seed_value)

    # prepare model & tokenizer
    device_llm = "mps" if torch.backends.mps.is_built() else f"cuda:{CUDA_ID_LLM}" if torch.cuda.is_available() else "cpu"
    print("device_llm: ", device_llm)
    device_deberta = "mps" if torch.backends.mps.is_built() else f"cuda:{CUDA_ID_DEBERTA}" if torch.cuda.is_available() else "cpu"
    print("device_deberta: ", device_deberta)
    
    auth_token = 'set your huggingface authorization token'
    
    args.llm_model =   ["mistralai/Mistral-7B-v0.1" ,'meta-llama/Llama-2-7b-hf'][1]
    # args.llm_model = 'facebook/opt-13b'
    
    model_type_deberta = ["deberta-base-mnli", "deberta-large-mnli", "deberta-xlarge-mnli", "deberta-v2-xlarge-mnli", "deberta-v2-xxlarge-mnli"][1]
    deberta_tokenizer = AutoTokenizer.from_pretrained(f"microsoft/{model_type_deberta}")
    deberta_model = AutoModelForSequenceClassification.from_pretrained(f"microsoft/{model_type_deberta}", device_map = 'auto')

    tokenizer = AutoTokenizer.from_pretrained(args.llm_model, token = auth_token)
    
    # llm_model = OPTForCausalLM.from_pretrained(args.llm_model,
    #                                         torch_dtype=torch.bfloat16, 
    #                                         # attn_implementation="flash_attention_2",
    #                                         # use_cache=True
    #                                           )
    # llm_model = llm_model.to(device_llm)
    
    llm_model = AutoModelForCausalLM.from_pretrained(args.llm_model, torch_dtype=torch.bfloat16, token = auth_token).to(device_llm)
    llm_model.resize_token_embeddings(len(tokenizer))

    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.add_special_tokens({'bos_token': '</s>'})
    tokenizer.add_special_tokens({'eos_token': '</s>'})
    tokenizer.add_special_tokens({'unk_token': '</s>'})
    
    dataset = load_dataset('truthfulqa/truthful_qa','generation')
    values = []
    for i in range(0,len(dataset['validation'])):
        values.append(i)
    dataset_ = dataset['validation'].add_column('question_id', values)
    
    few_shot = 'This is a bot that correctly answers questions. \n'
    for sample in dataset_.select(range(51,54)):
        question = sample['question']
        answer = sample['best_answer']
        if answer[-1] != ".":
            answer += "."
        few_shot += 'Q: ' + question + ' A: ' + answer + ' '
    
    # dataset = load_dataset('tasksource/ScienceQA_text_only')
    # values = []
    # for i in range(0,len(dataset['test'])):
    #     values.append(i)
    # dataset_ = dataset['test'].add_column('question_id', values)
  
    # dataset = load_dataset('allenai/openbookqa','additional')
    # values = []
    # for i in range(0,len(dataset['test'])):
    #     values.append(i)
    # dataset_ = dataset['test'].add_column('question_id', values)
    
    num =len(dataset_)
    # dataset_ = dataset_.select(range(0,5))
    dataset_ = encode_and_format_dataset(dataset_, few_shot)
    
    get_results(args=args,
                base_path=base_path, 
                llm_model=llm_model, 
                tokenizer=tokenizer, 
                device_llm=device_llm, 
                deberta_model=deberta_model, 
                deberta_tokenizer=deberta_tokenizer, 
                device_deberta=device_deberta, 
                dataset=dataset_)

    compute_semantic_paris_new(base_path=base_path, 
                        model_type=args.deberta_model, 
                        deberta_tokenizer=deberta_tokenizer, 
                        deberta_model=deberta_model, 
                        num_instances=len(dataset_),
                        # num_instances = 1000,
                        device=device_deberta,
                          offset = 0)
