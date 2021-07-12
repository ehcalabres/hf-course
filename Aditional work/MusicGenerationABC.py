#
# Work based on prevoius research that can be found on my personal repository.
#
# AI-MuG: https://github.com/ehcalabres/ai-mug
#

# +-------------------+
# | Utility imports   |
# +-------------------+

import regex as re
import numpy as np

# +----------------------------+
# | Transformers (HF) imports  |
# +----------------------------+

import argparse

import datasets
from datasets import load_dataset, load_metric

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments, pipeline

def training_function(config, args):

    # Hyper-parameters for model selection and training process
    lr = config['lr']
    seed = int(config['seed'])
    checkpoint = config['checkpoint']
    num_epochs = int(config['num_epochs'])
    batch_size = int(config['batch_size'])
    train_file_path = config['dataset']['train']
    test_file_path = config['dataset']['test']
    
    save = args.save
    save_directory = args.directory
  
    print('Importing, preprocessing and creating dataset...')
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    train_dataset = TextDataset(
        file_path=train_file_path,
        tokenizer=tokenizer,
        block_size=128
    )

    test_dataset = TextDataset(
        file_path=test_file_path,
        tokenizer=tokenizer,
        block_size=128
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    print('Initializing model...')
    model = AutoModelForCausalLM.from_pretrained(checkpoint)

    training_arguments = TrainingArguments(
        output_dir=save_directory,
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_steps = 400,
        save_steps=800,
        warmup_steps=500,
        prediction_loss_only=True
    )

    trainer = Trainer(
        model=model,
        args=training_arguments,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )

    print('Model training...')
    trainer.train()

    # Saving model
    if save:
        print('Saving model...')
        trainer.save_model(save_directory)

    print('Training process completed succesfuly.')


def inference_function(config, args):

    compositor = AutoModelForCausalLM.from_pretrained(args.directory)
    music_compositor = pipeline('text-generation', model=compositor, tokenizer=config['checkpoint'], config={'max_length': 800})

    new_composition = music_compositor("X:1\n", max_length=300, num_return_sequences=5)

    print("Compositions generated:")
    for element in new_composition:
        print("----------------------")
        print(element['generated_text'])
        print("----------------------")


def main():

    parser = argparse.ArgumentParser(description="Example of training a HF model with accelerator")
    parser.add_argument("--fp16", type=bool, default=False, help="If passed, will use FP16 training.")
    parser.add_argument("--cpu", type=bool, default=False, help="If passed, will train on the CPU.")
    parser.add_argument("--save", type=bool, default=False, help="If passed, the model will be saved after the training process.")
    parser.add_argument("--directory", type=str, default="./model_saved", help="Path to directory where model will be saved.")
    parser.add_argument("--inference", type=bool, default=False, help="If passed, will try to generate a new sample with the text provided with the parameter --inference-init (default='X:1')")
    parser.add_argument("--inference-text", type=str, default="X:1", help="Initial text from where the model will start generating.")

    args = parser.parse_args()

    config = {
        'checkpoint': 'distilgpt2', 
        'dataset': {
            'train': '../data/abc_dataset/irish_music - original.abc',
            'test': '../data/abc_dataset/irish_music_test.abc'},
        'lr': 5e-5, 
        'num_epochs': 10, 
        'seed': 42, 
        'batch_size': 4
    }

    if not args.inference:
        training_function(config, args)
    else:
        inference_function(config, args)

if __name__ == '__main__':
    transformers.logging.set_verbosity_warning()
    datasets.logging.set_verbosity_warning()
    main()
