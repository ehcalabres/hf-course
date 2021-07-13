from tqdm.auto import tqdm

import argparse

import torch
from torch.utils.data import DataLoader

from accelerate import Accelerator

import datasets
from datasets import load_dataset, load_metric, Sequence, Value

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
    # It's not needed to load the train and test split this way, it ca be done with the function
    # raw_datasets.train_test_split(test_size=0.2) when the dataset has been loaded with all the 
    # data available without being splited before.
    raw_datasets = load_dataset('csv', data_files={'train': '../data/complaints_split/train.csv',
                                                    'test': '../data/complaints_split/test.csv'},
                                        split={'train': 'train[:10%]',
                                                'test': 'test[:10%]'})
    metric = load_metric('glue', 'mnli')
    # Need to find which is the best metric for this text classification task

    # Dataset preprocessing
    print('Preprocessing dataset...')
    def tokenize_function(model_input):
        return tokenizer(model_input['Narrative'], truncation=True, max_length=512, padding='max_length')

    raw_datasets = raw_datasets.remove_columns(['Unnamed: 0'])

    cols = raw_datasets["train"].column_names
    raw_datasets = raw_datasets.map(lambda x : {"labels": [x[c] for c in cols if c != "Narrative"]})

    cols = raw_datasets['train'].column_names
    cols.remove('labels')

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True, remove_columns=cols)
    tokenized_datasets.set_format('pytorch')

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    new_features_train = tokenized_datasets['train'].features.copy()
    new_features_train['labels'] = Sequence(Value('float'))
    tokenized_datasets['train'] = tokenized_datasets['train'].cast(new_features_train)

    new_features_test = tokenized_datasets['test'].features.copy()
    new_features_test['labels'] = Sequence(Value('float'))
    tokenized_datasets['test'] = tokenized_datasets['test'].cast(new_features_test)

    train_dataloader = DataLoader(
        tokenized_datasets['train'], shuffle=True, batch_size=batch_size, collate_fn=data_collator
    )

    eval_dataloader = DataLoader(
        tokenized_datasets['test'], shuffle=True, batch_size=batch_size, collate_fn=data_collator
    )

    set_seed(seed)

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, problem_type='multi_label_classification', num_labels=18)

    model = model.to(accelerator.device)

    optimizer = AdamW(params=model.parameters(), lr=lr)

    train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
        train_dataloader, eval_dataloader, model, optimizer
    )

    lr_scheduler = get_scheduler(
        'linear',
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_epochs * len(train_dataloader)
    )

    print('Model training...')
    print('Starting training for epoch 0')
    # Progress bar
    progress_bar = tqdm(range(len(train_dataloader) + len(eval_dataloader)))
    for epoch in range(num_epochs):
        if epoch != 0:
            accelerator.print(f"\nStarting training for epoch {epoch}\n")
            progress_bar.reset()

        model.train()
        for step, batch in enumerate(train_dataloader):
            batch.to(accelerator.device)

            outputs = model(**batch)
            
            loss = outputs.loss
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            progress_bar.update(1)

        model.eval()
        for step, batch in enumerate(eval_dataloader):
            batch.to(accelerator.device)
            with torch.no_grad():
                outputs = model(**batch)
            
            predictions = outputs.logits.argmax(dim=-1)

            print(accelerator.gather(predictions))
            print(accelerator.gather(batch['labels']))

            metric.add_batch(
                predictions=accelerator.gather(predictions),
                references=accelerator.gather(batch['labels'])
            )

            progress_bar.update(1)

        eval_metric = metric.compute()

        accelerator.print(f"\nEpoch {epoch}:", eval_metric)

    # Saving model
    if save:
        print('Saving model...')
        model.save_pretrained(save_directory)
    
    print('Training process completed succesfuly.')


def main():
    parser = argparse.ArgumentParser(description="Example of training a HF model with accelerator")
    parser.add_argument("--fp16", type=bool, default=False, help="If passed, will use FP16 training.")
    parser.add_argument("--cpu", type=bool, default=False, help="If passed, will train on the CPU.")
    parser.add_argument("--save", type=bool, default=True, help="If True (default), the model will be saved after the training process.")
    parser.add_argument("--directory", type=str, default="./model_saved", help="Path to directory where model will be saved.")

    args = parser.parse_args()

    config = {'checkpoint': 'mrm8488/bert-tiny-finetuned-yahoo_answers_topics', 'lr': 5e-5, 'num_epochs': 3, 'seed': 42, 'batch_size': 16}
    config = {'checkpoint': 'bert-base-uncased', 'lr': 5e-5, 'num_epochs': 3, 'seed': 42, 'batch_size': 2}

    training_function(config, args)

if __name__ == '__main__':
    transformers.logging.set_verbosity_warning()
    datasets.logging.set_verbosity_warning()
    main()
