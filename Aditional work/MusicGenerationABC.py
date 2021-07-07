from tqdm.auto import tqdm

import argparse

import torch
from torch.utils.data import DataLoader

from accelerate import Accelerator

import datasets
from datasets import load_dataset, load_metric

import transformers
from transformers import AdamW, AutoModelForSequenceClassification, AutoTokenizer, get_scheduler, set_seed, DataCollatorWithPadding, AutoModelWithLMHead


def training_function(config, args):

    # Hyper-parameters for model selection and training process
    lr = config['lr']
    seed = int(config['seed'])
    checkpoint = config['checkpoint']
    num_epochs = int(config['num_epochs'])
    batch_size = int(config['batch_size'])
    
    save = args.save
    save_directory = args.directory
  
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    model = AutoModelWithLMHead.from_pretrained(checkpoint)


def main():

    checkpoint = "distilgpt2"


if __name__ == '__main__':
    transformers.logging.set_verbosity_warning()
    datasets.logging.set_verbosity_warning()
    main()