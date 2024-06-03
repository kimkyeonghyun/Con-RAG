
import bitsandbytes as bnb
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,AutoModelForSeq2SeqLM
import sys
import string
import re
from collections import Counter
import matplotlib.pyplot as plt
sys.path.append("/home/aoboyang/local/captum")
from captum.attr import (
    FeatureAblation, 
    ShapleyValues,
    LayerIntegratedGradients, 
    LLMAttribution, 
    LLMGradientAttribution, 
    TextTokenInput, 
    TextTemplateInput,
    ProductBaselines,
)
from captum.attr import visualization as viz

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_name, bnb_config):
    n_gpus = torch.cuda.device_count()
    max_memory = "10000MB"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto", # dispatch efficiently the model on the available ressources
        max_memory = {i: max_memory for i in range(n_gpus)},
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)

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
print(model)

inputs = ['Who won Super Bowl XX?']
answer = ['Chicago Bears']
doc = ["""Super Bowl XX was an American football game between the National Football Conference (NFC) champion Chicago Bears and the American Football Conference (AFC) champion New England Patriots to decide the National Football League (NFL) champion for the 1985 season. The Bears defeated the Patriots by the score of 46\u201310, capturing their first NFL championship since 1963, three years prior to the birth of the Super Bowl. Super Bowl XX was played on January 26, 1986, at the Louisiana Superdome in New Orleans."""]
input_ ='Questoin: '+ inputs[0] + '\nRelated Doc: ' + doc[0] + '\nAnswer:'
inp_ = tokenizer(input_, return_tensors='pt').to("cuda")

with torch.no_grad():
    outputs = model.generate(inp_["input_ids"], max_new_tokens=1024,)
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # print(prediction)
target = prediction.split('Answer:')[1]
print('-------')

print(target)
print('-------')

# lig = LayerIntegratedGradients(model, layer=model.lm_head) 
# llm_attr = LLMGradientAttribution(lig, tokenizer)
# inp1 = TextTokenInput(
#     input_, 
#     tokenizer,
#     skip_tokens=[1]
#     )
# attr_res1 = llm_attr.attribute(inp1, target=answer[0])
# fig, ax = attr_res1.plot_token_attr(show=False)
# fig.savefig('./llm.png')


template = 'Please refer to the given document and answer the question: \n Question: '+ '{}' + '\nRelate passage: ' + '{}' + '\nAnswer:'

sv = ShapleyValues(model) 
llm_attr = LLMAttribution(sv, tokenizer)
inp = TextTemplateInput(
    template = template, 
    values=[inputs[0], doc[0]]
    )
attr_res1 = llm_attr.attribute(inp, target=answer[0])
attr_res1.plot_seq_attr(show=True)


        
