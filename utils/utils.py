# text classification
# utils

# 라이르러리
import os
import sys
import time
import tqdm
import random
import logging
import argparse
import numpy as np
import torch
import torch.nn.functional as F

# 경로가 존재하는지 확인하고 없으면 경로를 생성
def check_path(path: str):
    '''
    Check if the path exists and create it if not.
    '''
    if not os.path.exists(path):
        os.makedirs(path)

# 난수 생성
def set_random_seed(seed: int):
    '''
    Set random seed for repruducibility.
    '''
    torch.manual_seed(seed) # CPU를 위한 난수 생성
    torch.cuda.manual_seed_all(seed) # GPU를 위한 난수 생성
    np.random.seed(seed)
    random.seed(seed)

# CPU와 GPU 중 어느 것을 사용할 것인지 정하는 함수
def get_torch_device(device: str):
    if device is not None:
        get_torch_device.device = device
    
    if 'cuda' in get_torch_device.device: # This also supprots Rocm by amd gpu.
        if torch.cuda.is_available():
            return torch.device(get_torch_device.device) # 멀티 gpu환경-하나만 있을 때는 cuda:0
        else:
            print('No GPU found. Using CPU.')
            return torch.device('cpu')
    elif 'mps' in device: # mac에서는 pytorch 1.12이상이 필요
        if not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print('MPS not available because the current Pytorch install'
                      ' was not built with MPS enabled.')
                print('Using CPU.')
            else:
                print('MPS not available because the current MacOS version'
                      ' is not 12.3+ and/or you do not have an MPS-enabled'
                      ' device on this machine.')
                print('Using CPU.')
            return torch.device('cpu')
        else:
            return torch.device(get_torch_device.device)
    elif 'cpu' in device:
        return torch.device('cpu')
    else:
        print('No such device found. Using CPU.')
        return torch.device('cpu')

# 로깅: 발생하는 이벤트를 시간 순서대로 기록하는 기능
class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level = logging.DEBUG):
        super().__init__(level)
        self.stream = sys.stdout 

    def flush(self):
        self.acquire()
        try:
            if self.stream and hasattr(self.stream, 'flush'):
                self.stream.flush() 
        finally:
            self.release()
    
    def emit(self, record):
        try:
            msg = self.format(record) # record에 저장된 로그 기록을 받아 문자열로 변환
            tqdm.tqdm.write(msg, self.stream) # msg, 로그 기록를 출력
            self.flush()
        except (KeyboardInterrupt, SystemExit, RecursionError):
            raise
        except Exception:
            self.handleError(record)

# logger: 파이썬 내장모듈 - 실행 과정에서 발생하는 이벤트나 메시지를 기록, 오류 디버깅, 사용자 활동 추적 등
def write_log(logger, message):
    if logger:
        logger.info(message) # 로깅 레벨: 로그의 우선순위를 결정하는데 사용되는 순위
        # info: 프로그램 작동이 예상대로 진행되고 있는지 트래킹하기 위해 사용

def get_tb_exp_name(args: argparse.Namespace):
    """
    tensorboard 실험을 위한 실험명 가져오기 
    """

    ts = time.strftime('%Y - %b - %d - %H: %M: %S', time.localtime()) # strftime: 주어진 format에 따란 시간을 문자로 변환
    # 년도 월 이름, 일, 시간, 분, 초, time.localtime(): 현재 시간을 변환
    exp_name = str()
    exp_name += "%s - " % args.task.upper()
    exp_name += "%s - " % args.proj_name

    if args.job in ['training', 'resume_training']:
        exp_name += 'TRAIN - '
        exp_name += 'MODEL = %s - ' % args.model_type.upper()
        exp_name += 'DATA = %s - ' % args.task_dataset.upper()
        exp_name += 'DESC = %s - ' % args.description
    
    elif args.job == 'testing':
        exp_name += 'TEST - '
        exp_name += 'MODEL = %s - ' % args.model_type.upper()
        exp_name += 'DATA = %s - ' % args.task_dataset.upper()
        exp_name += 'DESC = %s - ' % args.description
    exp_name += 'TS = %s' % ts

    return exp_name

def get_wandb_exp_name(args: argparse.Namespace):
    """
    weight와 biases 실험에 대한 실험명 가져오기
    """

    exp_name = str()
    exp_name += '%s - ' % args.task.upper()
    exp_name += '%s / ' % args.task_dataset.upper()
    exp_name += '%s' % args.model_type.upper()

    return exp_name

def get_huggingface_model_name(model_type: str) -> str:
    name = model_type.lower()
    # huggingface: 개방형 소스 프로젝트들 제공, 최신 자연어 처리 모델 및 기술을 제공

    if name == 't5-small':
        return 'google-t5/t5-small'
    elif name == 't5-base':
        return 'google-t5/t5-base'
    elif name == 't5-large':
        return 'google-t5/t5-large'



    
def parse_bool(value: str):
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')