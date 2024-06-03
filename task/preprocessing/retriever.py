# from pyserini.search.lucene import LuceneSearcher
import json
# import nltk
# nltk.download('punkt')
# from nltk.tokenize import sent_tokenize
# import spacy
from txtai.embeddings import Embeddings
import os
# nlp = spacy.load("en_core_web_md")
# searcher = LuceneSearcher.from_prebuilt_index('wikipedia-dpr')
embeddings = Embeddings(path = 'intfloat/e5-base')
embeddings.load(provider="huggingface-hub", container="neuml/txtai-wikipedia")
os.environ['CUDA_VISIBLE_DEVICES']='2'
def information_retrieval(query):
    
    hits = embeddings.search(query, 5)
    paragraphs = []
    for hit in hits:
        paragraphs.append(hit['text'])
    return paragraphs

def get_wiki_pages(data_path, output_path):
    output_option = 'sparse_retrieval'
    data_file = open(data_path, 'r')
    data = json.load(data_file)
    for idx, case in enumerate(data):
        print(f'Processing {idx+1}/{len(data)}')
        question = case['question']  # Assuming 'question' key holds the query text
        case[output_option] = []
        paragraphs = information_retrieval(question)
        case[output_option].append(paragraphs)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)


# def get_gold_pages(data_path, output_path):
#     output_option = 'gold_text'
#     data_file = open(data_path, 'r')
#     data = json.load(data_file)
#     for idx, case in enumerate(data):
#         print(f'Processing {idx+1}/{len(data)}')
#         question = case['question']  # Assuming 'question' key holds the query text
#         case[output_option] = []
#         paragraphs = []
#         for i in case['doc']:
#             with open(i, 'r', encoding='utf-8') as file:
#                 file_content = file.read()
#                 sentences = sent_tokenize(file_content, language='english')
#                 first_three_sentences = sentences[:3]
#             paragraphs.append(first_three_sentences)
#         case[output_option].append(paragraphs)
#         # print(case)

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)  # 데이터를 새 JSON 파일에 다시 덤프

# Example usage
def retriever(args):
    get_wiki_pages('./dataset/' + args.task_dataset + '/train.json', './dataset/' + args.task_dataset + '/train_wiki_retriever.json')
    get_wiki_pages('./dataset/' + args.task_dataset + '/test.json', './dataset/' + args.task_dataset + '/test_wiki_retriever.json')



