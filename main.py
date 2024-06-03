# 3주차 text classification
# IMDB 데이터셋을 Custom Dataset을 활용하여 Load

# 라이브러리 
import time
import argparse

from utils.arguments import ArgParser
from utils.utils import check_path, set_random_seed

def main(args: argparse.Namespace) -> None:
    # Set random see
    if args.seed is not None:
        set_random_seed(args.seed)

    start_time = time.time()

    # 경로 존재 확인
    for path in []:
        check_path(path)

    # 할 job 얻기
    if args.job == None:
        raise ValueError('Please specify the job to do.')
    else:
        if args.task == 'preprocessing':
            if args.job == 'preprocessing':
                from task.preprocessing.data_preprocessing import preprocessing as job
            elif args.job == 'generation':
                from task.preprocessing.question_classification import question_classification as job
            elif args.job == 'retriever':
                from task.preprocessing.retriever import retriever as job
        elif args.task == 'classification':
            if args.job == 'preprocessing':
                from task.classification.question_preprocessing import preprocessing as job
            elif args.job in ['training', 'resume_training']:
                from task.classification.train import training as job
            elif args.job == 'testing':
                from task.classification.test import testing as job
            elif args.job == 'generation_test':
                from task.classification.generation_test import classifier_llm_generation as job
            elif args.job == 'num_classification':
                from task.classification.classification_num import num_classification as job
            else:
                raise ValueError(f'Invalid job: {args.job}')
        else:
            raise ValueError(f'Invalid task: {args.task}')
        
    # job 하기
    job(args)

    elapsed_time = time.time() - start_time
    print(f'Completed {args.job}; Time elapsed: {elapsed_time / 60:.2f} minutes')

if __name__ == '__main__':
    parser = ArgParser()
    args = parser.get_args()

    main(args)