from tqdm.auto import tqdm

import argparse

import torch
from torch.utils.data import DataLoader

from accelerate import Accelerator

import datasets
from datasets import load_dataset, load_metric

import transformers
from transformers import AdamW, AutoModelForSequenceClassification, AutoTokenizer, get_scheduler, set_seed, DataCollatorWithPadding

# MAIN TRAINING FUNCTION
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
    print('Downloading and creating dataset...')
    raw_datasets = load_dataset('glue', 'mrpc')
    metric = load_metric('glue', 'mrpc')

    # Dataset preprocessing
    print('Preprocessing dataset...')
    def tokenize_function(model_input):
        return tokenizer(model_input['sentence1'], model_input['sentence2'], truncation=True)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(['sentence1', 'sentence2', 'idx'])
    tokenized_datasets = tokenized_datasets.rename_column('label', 'labels')
    tokenized_datasets.set_format('pytorch')

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_dataloader = DataLoader(
        tokenized_datasets['train'], shuffle=True, batch_size=batch_size, collate_fn=data_collator
    )

    eval_dataloader = DataLoader(
        tokenized_datasets['validation'], batch_size=batch_size, collate_fn=data_collator
    )

    set_seed(seed)

    # Model and optimizers creation
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

    # When sending tensors to devices manually, this line needs to be placed before the optimizer creation,
    # otherwise it will not work with TPUs
    model = model.to(accelerator.device)

    # Optimizer creation
    optimizer = AdamW(params=model.parameters(), lr=lr)

    # Preparation of all our assets to be used with accelerator. It's important to place all the assets in
    # the exact same order than were passed to optimizer.prepare() function in order to be unpacked well.
    train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
        train_dataloader, eval_dataloader, model, optimizer
    )

    # Learning rate scheduler creation
    lr_scheduler = get_scheduler(
        'linear',
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_epochs * len(train_dataloader)
    )

    # Model training
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

    config = {'checkpoint': 'bert-base-uncased', 'lr': 5e-5, 'num_epochs': 3, 'seed': 42, 'batch_size': 16}

    training_function(config, args)

if __name__ == '__main__':
    transformers.logging.set_verbosity_error()
    datasets.logging.set_verbosity_error()
    main()