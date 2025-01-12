import os
import time
from datetime import datetime
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)
import torch


splits = {
    'train': 'openassistant_best_replies_train.jsonl',
    'test':  'openassistant_best_replies_eval.jsonl'
}

model_checkpoint = "bert-base-uncased"

output_dir = "./Models/Bert/v3/bert-finetuned-mlm"

default_num_train_epochs = 3
default_train_batch_size = 8
default_eval_batch_size = 8
default_learning_rate = 5e-5
default_weight_decay = 0.01
seed = 42

do_hpo = True  

n_hpo_trials = 10

def hp_space_optuna(trial):
    """
    Optuna parameter search space.#
    Adjust to suit your search needs.
    """
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 5),
        "seed": trial.suggest_categorical("seed", [42, 1234, 2021]),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32]),
        "per_device_eval_batch_size": trial.suggest_categorical("per_device_eval_batch_size", [8, 16, 32]),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
    }

def compute_objective(metrics):
    """
    Objective function for hyperparameter search.
    We want to minimize the validation loss. The Trainer by default returns 'eval_loss'.
    """
    return metrics["eval_loss"]

def main():
    set_seed(seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading dataset from Hugging Face...")
    
    train_data = pd.read_json(f"hf://datasets/timdettmers/openassistant-guanaco/{splits['train']}", lines=True)
    test_data = pd.read_json(f"hf://datasets/timdettmers/openassistant-guanaco/{splits['test']}", lines=True)

    train_dataset = Dataset.from_dict({"text": train_data["text"].tolist()})
    test_dataset = Dataset.from_dict({"text": test_data["text"].tolist()})

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of test samples:     {len(test_dataset)}")

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=128
        )

    print("Tokenizing dataset...")
    train_tokenized = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    test_tokenized = test_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    def model_init():
        return AutoModelForMaskedLM.from_pretrained(model_checkpoint)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=100,
        num_train_epochs=default_num_train_epochs,
        per_device_train_batch_size=default_train_batch_size,
        per_device_eval_batch_size=default_eval_batch_size,
        learning_rate=default_learning_rate,
        weight_decay=default_weight_decay,
        seed=seed,
    )

    trainer = Trainer(
        model_init=model_init if do_hpo else None,  
        model=None if do_hpo else AutoModelForMaskedLM.from_pretrained(model_checkpoint),
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=test_tokenized,
        data_collator=data_collator,
    )

    if do_hpo:
        print("\n=== Running hyperparameter search ===")
        best_run = trainer.hyperparameter_search(
            direction="minimize",
            hp_space=hp_space_optuna,
            compute_objective=compute_objective,
            n_trials=n_hpo_trials
        )
        print("Best hyperparameters found:", best_run.hyperparameters)

        for k, v in best_run.hyperparameters.items():
            setattr(trainer.args, k, v)

        trainer.model = trainer.model_init()
    else:
        print("Skipping hyperparameter search...")

    print("\n=== Starting training ===")
    trainer.train()

    print("\n=== Evaluating on test dataset ===")
    eval_metrics = trainer.evaluate()
    print(f"Eval metrics: {eval_metrics}")
    if "eval_loss" in eval_metrics:
        perplexity = torch.exp(torch.tensor(eval_metrics["eval_loss"]))
        print(f"Perplexity: {perplexity.item():.2f}")

    print(f"\n=== Saving final model to {output_dir} ===")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("All done!")

if __name__ == "__main__":

    start_time = time.time()
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    main()

    end_time = time.time()
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    time_taken = end_time - start_time
    print(f"Time taken to execute the loop: {time_taken:.2f} seconds")