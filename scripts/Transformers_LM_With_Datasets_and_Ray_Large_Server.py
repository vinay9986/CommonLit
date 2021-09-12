#!/usr/bin/env python

num_cpus=1
num_gpus=1
model_name="roberta-large"
keep_checkpoints_num=1

block_size=256
train_num_shards=2
perturbation_interval=1
n_trials=16


import torch
torch.cuda.empty_cache()
import transformers
import datasets

print(f"Running on transformers v{transformers.__version__}, datasets v{datasets.__version__}, torch v{torch.__version__}")

import ray
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining

from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling, TrainingArguments, Trainer
from torch.nn.functional import cross_entropy

import numpy as np
import pandas as pd
import math
import copy
import random

import os

print("Creating directories")
for d in ['results', 'logs', 'ray_results', 'models']:
    if not os.path.exists(d):
        os.makedirs(d)
print("Directories created")

ray.shutdown()
ray.init(log_to_driver=False, ignore_reinit_error=True, num_cpus=num_cpus, num_gpus=num_gpus, include_dashboard=False)

tokenizer=AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["text"])

def get_model():
    return AutoModelForMaskedLM.from_pretrained(model_name)

pd_train=pd.read_csv('/home/ubuntu/train/preprocessed_train_data.csv', low_memory=False)
columns = pd_train.columns.tolist()
columns.remove('text')
pd_train.drop(columns=columns, inplace=True, axis=1)

train_testeval=(datasets.Dataset.from_pandas(df=pd_train)).train_test_split(train_size=.75, test_size=.25)
test_eval=train_testeval['test'].train_test_split(train_size=.75, test_size=.25)
ds=datasets.DatasetDict({
    'train': train_testeval['train'],
    'test': test_eval['test'],
    'valid': test_eval['train']})
tokenized_datasets=ds.map(tokenize_function, batched=True, num_proc=14, remove_columns=["text"])
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples={k: sum(examples[k], []) for k in examples.keys()}
    total_length=len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
    total_length=(total_length // block_size) * block_size
    # Split by chunks of max_len.
    result={
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"]=result["input_ids"].copy()
    return result

lm_datasets=tokenized_datasets.map(group_texts, batched=True, batch_size=1000, num_proc=14)

training_args = TrainingArguments(
    output_dir="/home/ubuntu/train/results",
    overwrite_output_dir = True,
    learning_rate=1e-5,                             # config
    do_train=True,
    do_eval=True,
    evaluation_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model = 'eval_perplexity',
    greater_is_better = False,
    num_train_epochs=3,                             # config
    max_steps=-1,                                   # config
    per_device_train_batch_size=32,                 # config
    per_device_eval_batch_size=32,                  # config
    warmup_steps=0,                                 # config
    weight_decay=0.1,                               # config
    logging_dir="/home/ubuntu/train/logs",
    logging_steps=10,
    skip_memory_metrics=True,
    report_to="none",
    fp16=True,
    seed=12,                                        # config
)

def compute_perplexity(pred):
    logits = torch.from_numpy(pred.predictions)
    labels = torch.from_numpy(pred.label_ids)
    loss = cross_entropy(logits.view(-1, tokenizer.vocab_size), labels.view(-1))
    try:
        perplexity = math.exp(loss)
    except OverflowError:
        perplexity = float("inf")
    return {'perplexity': perplexity, 'calculated_loss': loss}

trainer = Trainer(
    model_init=get_model,
    args=training_args,
    train_dataset=lm_datasets['train'].shard(index=1, num_shards=train_num_shards),
    eval_dataset=lm_datasets['valid'].shard(index=1, num_shards=10),
    tokenizer = tokenizer,
    data_collator = data_collator,
    compute_metrics=compute_perplexity
)

tune_config = {
    "max_steps": -1,
    "per_device_eval_batch_size": 32,
    "num_train_epochs": 3,
}

scheduler = PopulationBasedTraining(
    time_attr="training_iteration",
    metric="eval_perplexity",
    mode="min",
    perturbation_interval=perturbation_interval,
    require_attrs=True,
    hyperparam_mutations={
        "learning_rate": tune.loguniform(1e-6, 1e-1),
        "per_device_train_batch_size": tune.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 
                                                15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                                                27, 28, 29, 30, 31, 32]),
        "warmup_steps": tune.randint(1, 10001),
        "weight_decay": tune.uniform(1e-1, 0.6),
        "seed": tune.randint(1, 1000),
    })

def objective_perplexity(metrics):
    return metrics['eval_perplexity']

best_run = trainer.hyperparameter_search(
    hp_space=lambda _: tune_config,
    compute_objective=objective_perplexity,
    direction='minimize',
    backend="ray",
    n_trials=n_trials,
    resources_per_trial={
        "cpu": num_cpus,
        "gpu": num_gpus
    },
    scheduler=scheduler,
    keep_checkpoints_num=keep_checkpoints_num,
    checkpoint_score_attr="training_iteration",
    stop=None,
    local_dir="/home/ubuntu/train/ray_results",
    name="tune-clrp-mlm",
    log_to_file=True
)

for n, v in best_run.hyperparameters.items():
    setattr(trainer.args, n, v)

trainer.train()
trainer.save_model('/home/ubuntu/train/models')

print('*'*100)
print(best_run)
print('*'*100)

ray.shutdown()