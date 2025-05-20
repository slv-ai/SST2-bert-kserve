import os
import numpy as np 
import pandas as pd 
from datasets import load_dataset
import torch 
from torch.utils.data import DataLoader
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import accuracy_score
import argparse
import json

torch.manual_seed(42)
np.random.seed(42)

def load_and_prepare_data():
    print("loading the SST-2 dataset...")
    dataset=load_dataset("glue","sst2")
    train_data=dataset["train"]
    test_data=dataset["test"]
    return train_data,test_data

def tokenize_data(tokenizer,texts,labels,max_length=128):
    encodings=tokenizer
