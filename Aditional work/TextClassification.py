from tqdm.auto import tqdm

import argparse

import torch
from torch.utils.data import DataLoader

from accelerate import Accelerator

import datasets
from datasets import load_dataset, load_metric

import transformers
from transformers import AdamW, AutoModelForSequenceClassification, AutoTokenizer, get_scheduler, set_seed, DataCollatorWithPadding

def training_function(config, args):

    # Hyper-parameters for model selection and training process
    lr = config['lr']
    seed = int(config['seed'])
    checkpoint = config['checkpoint']
    num_epochs = int(config['num_epochs'])
    batch_size = int(config['batch_size'])
    
    save = args.save
    save_directory = args.directory

    # Initialize accelerator
    accelerator = Accelerator(fp16=args.fp16, cpu=args.cpu)

    # HF initialization
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    print('Importing and creating dataset...')
    raw_datasets = load_dataset('csv', data_files='../data/complaints_cleaned.csv')
    # Need to find which is the best metric for this text classification task

    # Dataset preprocessing
    print('Preprocessing dataset...')
    def tokenize_function(model_input):
        return tokenizer(model_input['Narrative'], truncation=True)

    print(raw_datasets)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(['Narrative'])
    tokenized_datasets = tokenized_datasets.rename_column('Product', 'labels')
    tokenized_datasets.set_format('pytorch')

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_dataloader = DataLoader(
        
    )


def main():
    parser = argparse.ArgumentParser(description="Example of training a HF model with accelerator")
    parser.add_argument("--fp16", type=bool, default=False, help="If passed, will use FP16 training.")
    parser.add_argument("--cpu", type=bool, default=False, help="If passed, will train on the CPU.")
    parser.add_argument("--save", type=bool, default=True, help="If True (default), the model will be saved after the training process.")
    parser.add_argument("--directory", type=str, default="./model_saved", help="Path to directory where model will be saved.")

    args = parser.parse_args()

    config = {'checkpoint': 'bert-base-uncased', 'lr': 5e-5, 'num_epochs': 3, 'seed': 42, 'batch_size': 16}

    training_function(config, args)

if __name__ == '__main__':
    transformers.logging.set_verbosity_warning()
    datasets.logging.set_verbosity_warning()
    main()