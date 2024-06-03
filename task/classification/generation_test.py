# text classification
# test

# 라이브러리
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false' # tokenizer들이 모든 cpu에 할당되지 않도록 방지
import sys
import logging
import argparse
from tqdm.auto import tqdm
from sklearn.metrics import f1_score
import pandas as pd
import torch
import json
from torch.nn.functional import softmax
torch.set_num_threads(2) # tokenizer들이 모든 cpu에 할당되지 않도록 방지
import torch.nn as nn
from torch.utils.data import DataLoader
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model.classification.model import ClassificationModel
from model.classification.dataset import CustomDataset
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM, T5ForSequenceClassification, T5Config
from utils.utils import TqdmLoggingHandler, write_log, get_huggingface_model_name, get_wandb_exp_name, get_torch_device
from utils.metric import f1_cal, gold_answer, answer_accuracy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def classifier_model(args: argparse.Namespace) -> tuple: # (test_acc_cls, test_f1_cls)
    device = get_torch_device(args.device)
    
    # logger 및 tensorboard writer 정의
    logger = logging.getLogger(__name__)
    if len(logger.handlers) > 0:
        logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter(" %(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.propagate = False
    
    # model 불러오기
    write_log(logger, 'Building model')
    model_name = get_huggingface_model_name(args.model_type)
    classifier_tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = T5Config.from_pretrained(model_name)
    config.num_labels = 1
    classifier = T5ForSequenceClassification.from_pretrained(model_name, config=config)


    # 모델 weights 로드
    write_log(logger, "Loading model weights")
    load_model_name = os.path.join(args.model_path, args.task, args.task_dataset, args.model_type, 'final_model.pt')
    classifier = classifier.to('cpu')
    print(load_model_name)
    checkpoint = torch.load(load_model_name, map_location =torch.device('cpu'))
    for key in list(checkpoint['model'].keys()):
        if 'model.' in key:
            checkpoint['model'][key.replace('model.', '')] = checkpoint['model'].pop(key)

    classifier.load_state_dict(checkpoint['model'])
    classifier = classifier.to(device)
    write_log(logger, f'Loaded model weights from {load_model_name}')
    del checkpoint
    classifier.eval()
    # Wandb 로드
    if args.use_wandb:
        import wandb
        from wandb import AlertLevel
        wandb.init(project = args.proj_name,
                   name = get_wandb_exp_name(args) + f' - Test',
                   config = args,
                   notes = args.description,
                   tags = ['TEST',
                           f'Dataset: {args.task_dataset}',
                           f'Model: {args.model_type}'])
    return classifier, classifier_tokenizer



def load_model(model_name, bnb_config):
    n_gpus = torch.cuda.device_count()
    max_memory = "10000MB"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map='auto', # dispatch efficiently the model on the available ressources
        max_memory = {0: "10000MB"},
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Needed for LLaMA tokenizer
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def create_bnb_config():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    return bnb_config


def classifier_question(classifier, classifier_tokenizer, question):
    question_token = classifier_tokenizer(question, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = classifier(**question_token)
    probabilities = torch.sigmoid(outputs.logits.squeeze())
    predicted_class_idx = (probabilities > 0.5).long().item()
    
    return predicted_class_idx

def classifier_llm_generation(args):
    dir_path = './dataset/' + args.task_dataset
    data_path = dir_path + '/test_wiki_retriever.json'
    data_file = open(data_path, 'r')
    data = json.load(data_file)
    pred_ans = []
    
    # classifier, classifier_tokenizer = classifier_model(args)
    # classifier.eval()
    model_name = "meta-llama/Llama-2-7b-chat-hf" 
    bnb_config = create_bnb_config()
    model, tokenizer = load_model(model_name, bnb_config)
    model.eval()
    for idx, question in enumerate(tqdm(data)):
        with torch.no_grad():
            question['pred_answer'] = []
            question['accuracy'] = []
            my_question = question["question"]
            document = question["sparse_retrieval"][0]
            doc = ''
            is_retrieved = 0
            for i in document:
                # for j in i:
                    doc += i          
            if args.task_dataset == 'triviaqa':
                ans = question['answer']
                # retrieval o
                if is_retrieved == 0:
                    inputs = 'Please answer the question: \n Question: '+ my_question + '\nAnswer: '
                elif is_retrieved == 1:
                    inputs = 'Please refer to the given document and answer the question: \n Question: '+ my_question + '\nRelate passage: ' + doc + '\nAnswer:'

                inp_ = tokenizer(inputs, return_tensors='pt').to(device)

                outputs = model.generate(**inp_)
                
                answer = tokenizer.decode(outputs[0])
            
            elif args.task_dataset == 'naturalqa':
                ans = question['answer']
                # retrieval o
                if is_retrieved == 0:
                    inputs = 'Please answer the question: \n Question: '+ my_question + '\nAnswer: '
                elif is_retrieved == 1:
                    inputs = 'Please refer to the given document and answer the question: \n Question: '+ my_question + '\nRelate passage: ' + doc + '\nAnswer:'

                inp_ = tokenizer(inputs, return_tensors='pt').to(device)

                outputs = model.generate(**inp_)
                
                answer = tokenizer.decode(outputs[0])

            elif args.task_dataset == 'squad':
                ans = question['answer']
                # retrieval o
                if is_retrieved == 0:
                    inputs = 'Please answer the question: \n Question: '+ my_question + '\nAnswer: '
                elif is_retrieved == 1:
                    inputs = 'Please refer to the given document and answer the question: \n Question: '+ my_question + '\nRelate passage: ' + doc + '\nAnswer:'

                inp_ = tokenizer(inputs, return_tensors='pt').to(device)

                outputs = model.generate(**inp_)
                
                answer = tokenizer.decode(outputs[0])
                
            answer = answer.split('Answer:')[1] if 'Answer:' in answer else 'No answer found'


            pred_ans.append([ans, answer])
            # print(pred_ans[idx])
            
            question['pred_answer'].append(answer)
            question['accuracy'].append(answer_accuracy(answer, ans))

    if args.task_dataset == 'triviaqa':

        f1 = f1_cal(args, pred_ans)
        print("f1: ", f1)
        data[0]['f1_score']=[]
        data[0]['f1_score'].append(f1)
        all_accuracy = gold_answer(args, pred_ans)
        print("all_accuracy: ",all_accuracy)
        data[0]['all_accuracy']=[]
        data[0]['all_accuracy'].append(all_accuracy)

    elif args.task_dataset == 'naturalqa':

        f1 = f1_cal(args, pred_ans)
        print("f1: ", f1)
        data[0]['f1_score']=[]
        data[0]['f1_score'].append(f1)
        all_accuracy = gold_answer(args, pred_ans)
        print("all_accuracy: ",all_accuracy)
        data[0]['all_accuracy']=[]
        data[0]['all_accuracy'].append(all_accuracy)

    elif args.task_dataset == 'squad':

        f1 = f1_cal(args, pred_ans)
        print("f1: ", f1)
        data[0]['f1_score']=[]
        data[0]['f1_score'].append(f1)
        all_accuracy = gold_answer(args, pred_ans)
        print("all_accuracy: ",all_accuracy)
        data[0]['all_accuracy']=[]
        data[0]['all_accuracy'].append(all_accuracy)

    with open(dir_path + '/base_test_results_' + args.model_type+ '.json', 'w') as f:
        json.dump(data, f, indent=4)
    
