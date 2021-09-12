#!/usr/bin/env python

import torch
torch.cuda.empty_cache()
import transformers
import datasets

print(f"Running on transformers v{transformers.__version__}, datasets v{datasets.__version__}, torch v{torch.__version__}")

import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import PopulationBasedTraining

from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling, TrainingArguments, Trainer
from torch.nn.functional import cross_entropy
from tqdm.notebook import tqdm

import pandas as pd
import math
import copy

import os
import pathlib
import datetime


ray.shutdown()
ray.init(log_to_driver=False, ignore_reinit_error=True, num_cpus=1, num_gpus=1, include_dashboard=False)

data_dir_name="../data/ignore/"
model_name="distilroberta-base"
task_name="clrp-mlm"
task_data_dir=os.path.join(data_dir_name, task_name.upper())

tokenizer=AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["text"])

def get_model():
    return AutoModelForMaskedLM.from_pretrained(model_name)

pd_train=pd.read_csv('../data/preprocessed_train_data.csv', low_memory=False)
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
block_size=240
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
    output_dir="../data/mlm/output_for_preprocessed_train_data/with_HPO_results",
    do_train=True,
    do_eval=True,
    evaluation_strategy="epoch",
    load_best_model_at_end=True,
    logging_dir="../data/mlm/output_for_preprocessed_train_data/with_HPO_logs",
    skip_memory_metrics=True,
    report_to="none",
    eval_accumulation_steps=1,
    fp16=True,
)

def compute_perplexity(pred):
    logits = torch.from_numpy(pred.predictions)
    labels = torch.from_numpy(pred.label_ids)
    loss = cross_entropy(logits.view(-1, tokenizer.vocab_size), labels.view(-1))
    return {'perplexity': math.exp(loss), 'calculated_loss': loss}


trainer = Trainer(
    model_init=get_model,
    args=training_args,
    train_dataset=lm_datasets['train'],
    eval_dataset=lm_datasets['valid'],
    tokenizer = tokenizer,
    data_collator = data_collator,
    compute_metrics=compute_perplexity
)

tune_config = {
    "per_device_eval_batch_size": 24,
    "max_steps": -1,
}

scheduler = PopulationBasedTraining(
    time_attr="training_iteration",
    metric="eval_perplexity",
    mode="min",
    perturbation_interval=3,
    require_attrs=True,
    hyperparam_mutations={
        "learning_rate": tune.loguniform(1e-6, 1e-1),
        "num_train_epochs": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "per_device_train_batch_size": tune.randint(1, 25),
        "warmup_steps": [10, 100, 500, 1000],
        "weight_decay": tune.uniform(0.0, 0.6),
        "seed": tune.randint(1, 1000),
    })

def objective_perplexity(metrics):
    return metrics['eval_perplexity']

best_run = trainer.hyperparameter_search(
    hp_space=lambda _: tune_config,
    compute_objective=objective_perplexity,
    direction='minimize',
    backend="ray",
    n_trials=10,
    resources_per_trial={
        "cpu": 1,
        "gpu": 1
    },
    scheduler=scheduler,
    keep_checkpoints_num=1,
    checkpoint_score_attr="training_iteration",
    stop=None,
    local_dir="../data/mlm/output_for_preprocessed_train_data/ray_results",
    name="tune-clrp-mlm",
    log_to_file=True
)

print(best_run)

for n, v in best_run.hyperparameters.items():
    setattr(trainer.args, n, v)

trainer.train()

trainer.save_model('../data/mlm/output_for_preprocessed_train_data/with_HPO_model')



