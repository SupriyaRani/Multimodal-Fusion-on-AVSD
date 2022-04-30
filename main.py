import json
import pickle
import logging
import numpy as np
import torch
import torch.utils.data
from torch.utils.data import Dataset
from itertools import chain
import tarfile
from argparse import ArgumentParser
from utils.meta.dataset import get_dataset, build_input_from_segments
from utils.meta.VideoGPT2 import *
from utils.meta.train import SPECIAL_TOKENS, SPECIAL_TOKENS_DICT

if __name__ =="__main__":
    parser = ArgumentParser()
    parser.add_argument("--test_set", type=str, default="E:/Supriya/Final project/AVSD/data/dataset/DSTC7-AVSD/OfficialData/train_set4DSTC7-AVSD.json")
    parser.add_argument("--model_checkpoint", type=str, default="gpt2", help="Path, url or short name of the model")
    parser.add_argument("--model", type=str, default="gpt2", help="Path, url or short name of the model")
    parser.add_argument("--max_history", type=int, default=3, help="Number of previous exchanges to keep in history")
    args = parser.parse_args()
    for arg in vars(args):
        print("{}={}".format(arg, getattr(args, arg)))

    tokenizer_class = GPT2Tokenizer if "gpt2" == args.model else OpenAIGPTTokenizer
    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)
    tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)
    #logging.info('Loading test data from ' + args.test_set)
    test_data = json.load(open('E:/Supriya/Final project/AVSD/data/dataset/DSTC7-AVSD/OfficialData/train_set4DSTC7-AVSD.json','r'))
    test_dataset = get_dataset(tokenizer, args.test_set, undisclosed_only=False, n_history=args.max_history)
    print(test_dataset)