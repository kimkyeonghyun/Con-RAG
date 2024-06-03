import torch
import json
from tqdm import tqdm
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from captum.attr import (
    ShapleyValues, 
    LLMAttribution, 
    TextTemplateInput,
)
from captum.attr import visualization as viz

from utils.metric import f1_cal, gold_answer, answer_accuracy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_name, bnb_config):

    max_memory = "10000MB"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto", # dispatch efficiently the model on the available ressources
        max_memory = {0: max_memory},
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

model_name = "meta-llama/Llama-2-7b-chat-hf" 
bnb_config = create_bnb_config()
model, tokenizer = load_model(model_name, bnb_config)
model.eval()

def question_classification(args):
    dir_path = './dataset/' + args.task_dataset
    data_path = dir_path + '/train_wiki_retriever.json'
    data_file = open(data_path, 'r')
    data = json.load(data_file)
    pred_ans = []
    no_doc_list = []
    ret_doc_list = []

    # prefix = 'All answer must directly address the question asked. For instance, if a date or a name is requested, include only that specific information in your answer.'
    for idx, question in enumerate(tqdm(data[20000:40000])):
        with torch.no_grad():
            question['pred_answer'] = []
            question['accuracy'] = []

            my_question = question["question"]
            document = question["sparse_retrieval"][0]
            doc = ''
            for i in document:
                # for j in i:
                    doc += i            

            if args.task_dataset == 'triviaqa':
                ans = question['answer']
                # retrieval o
                inputs = 'Please refer to the given document and answer the question: \n Question: '+ my_question + '\nRelate passage: ' + doc + '\nAnswer:'
                # retrieval x
                # inputs = prefix + 'Please refer to the given document and answer the question: \n Question: '+ my_question + '\nAnswer: '
                inp_ = tokenizer(inputs, return_tensors='pt').to(device)

                outputs = model.generate(**inp_)
                
                answer = tokenizer.decode(outputs[0])
                
                answer = answer.split('Answer:')[1] if 'Answer:' in answer else 'No answer found'
                template = 'Please refer to the given document and answer the question: \n Question: '+ '{}' + '\nRelate passage: ' + '{}' + '\nAnswer:'

            elif args.task_dataset == 'naturalqa':
                ans = question['answer']
                # retrieval o
                inputs = 'Please refer to the given document and answer the question: \n Question: '+ my_question + '?' + '\nRelate passage: ' + doc + '\nAnswer:'
                # retrieval x
                # inputs = prefix + 'Please refer to the given document and answer the question: \n Question: '+ my_question + '\nAnswer: '
                inp_ = tokenizer(inputs, return_tensors='pt').to(device)

                outputs = model.generate(**inp_)
                
                answer = tokenizer.decode(outputs[0])
                
                answer = answer.split('Answer:')[1] if 'Answer:' in answer else 'No answer found'
                template = 'Please refer to the given document and answer the question: \n Question: '+ '{}' + '\nRelate passage: ' + '{}' + '\nAnswer:'

            elif args.task_dataset == 'squad':
                ans = question['answer']
                # retrieval o
                inputs = 'Please refer to the given document and answer the question: \n Question: '+ my_question + '?' + '\nRelate passage: ' + doc + '\nAnswer:'
                # retrieval x
                # inputs = prefix + 'Please refer to the given document and answer the question: \n Question: '+ my_question + '\nAnswer: '
                inp_ = tokenizer(inputs, max_length=4096, return_tensors='pt').to(device)

                outputs = model.generate(**inp_)
                
                answer = tokenizer.decode(outputs[0])
                
                answer = answer.split('Answer:')[1] if 'Answer:' in answer else 'No answer found'
                template = 'Please refer to the given document and answer the question: \n Question: '+ '{}' + '\nRelate passage: ' + '{}' + '\nAnswer:'


            if answer_accuracy(answer, ans) > 0:
                question_attr, doc_attr = classify_question(template, my_question, doc, ans)
                if question_attr > doc_attr: 
                    no_doc_list.append(my_question)
                else:
                    ret_doc_list.append(my_question)
            pred_ans.append([ans, answer])

            question['pred_answer'].append(answer)
            question['accuracy'].append(answer_accuracy(answer, ans))

    if args.task_dataset == 'triviaqa':
        # rouge_score = rouge_cal(args, pred_ans)
        # print(rouge_score)
        # data[0]['rouge_score']=[]
        # data[0]['rouge_score'].append(rouge_score)
        f1 = f1_cal(args, pred_ans)
        print("f1: ", f1)
        data[0]['f1_score']=[]
        data[0]['f1_score'].append(f1)
        exat = gold_answer(args, pred_ans)
        print("accuracy: ",exat)
        data[0]['exat']=[]
        data[0]['exat'].append(exat)
        # accuracy = accuracy_cal(args, pred_ans)
        # data[0]['accuracy']=[]
        # data[0]['accuracy'].append(accuracy)
        # print("accuracy: ", accuracy)
    elif args.task_dataset == 'naturalqa':
        f1 = f1_cal(args, pred_ans)
        print("f1: ", f1)
        data[0]['f1_score']=[]
        data[0]['f1_score'].append(f1)
        exat = gold_answer(args, pred_ans)
        print("accuracy: ",exat)
        data[0]['exat']=[]
        data[0]['exat'].append(exat)
    question_class_path = dir_path + '/train_question_classification.json'
    question_file = open(question_class_path, 'r')
    question_list = json.load(question_file)
    question_list['no_doc'] += no_doc_list
    question_list['ret_doc'] += ret_doc_list

    
    with open(dir_path + '/retrieval_train_results_' + args.model_type + '-7B.json', 'w') as f:
        json.dump(data, f, indent=4)
    with open(dir_path + '/train_question_classification.json', 'w') as f:
        json.dump(question_list, f, indent = 4)
    
    
def classify_question(template, my_question, doc, answer):
    sv = ShapleyValues(model) 
    llm_attr = LLMAttribution(sv, tokenizer)
    inp = TextTemplateInput(
        template = template, 
        values=[my_question, doc]
        )
    attr_res1 = llm_attr.attribute(inp, target=answer)
    return attr_res1.seq_attr[0], attr_res1.seq_attr[1]


