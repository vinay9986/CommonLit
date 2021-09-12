#!/usr/bin/env python
# Change model save path in the last line
num_cpus=1
num_gpus=1
model_name="roberta-large"
keep_checkpoints_num=1

train_num_shards=2
perturbation_interval=1
n_trials=10


import torch
torch.cuda.empty_cache()
import transformers
import datasets

print(f"Running on transformers v{transformers.__version__}, datasets v{datasets.__version__}, torch v{torch.__version__}")

import ray
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

import numpy as np
import pandas as pd
import random
import copy
import os

from sklearn.metrics import mean_squared_error

# print("Creating directories")
# for d in ['results', 'logs', 'ray_results', 'models']:
#     if not os.path.exists(d):
#         os.makedirs(d)
# print("Directories created")

ray.shutdown()
ray.init(log_to_driver=False, ignore_reinit_error=True, num_cpus=num_cpus, num_gpus=num_gpus, include_dashboard=False)

tokenizer = AutoTokenizer.from_pretrained(model_name, problem_type="regression")

def tokenize_function(examples):
    return tokenizer.batch_encode_plus(examples["text"], truncation=True, padding=True, max_length=256)

def get_model():
    return AutoModelForSequenceClassification.from_pretrained('/mnt/1TB/workspace/comp/Commonlit_kaggle_1/data/mlm/output_for_preprocessed_train_data/without_HPO/models/roberta_large', num_labels=1, problem_type="regression")

pd_train = pd.read_csv('../data/preprocessed_train_data.csv', low_memory=False)
columns = pd_train.columns.tolist()
columns.remove('text')
columns.remove('labels')
pd_train.drop(columns=columns, inplace=True, axis=1)

train_testeval=(datasets.Dataset.from_pandas(df=pd_train)).train_test_split(train_size=.75, test_size=.25)
test_eval=train_testeval['test'].train_test_split(train_size=.75, test_size=.25)
ds=datasets.DatasetDict({
    'train': train_testeval['train'],
    'test': test_eval['test'],
    'valid': test_eval['train']})
cols = ds["train"].column_names
cols.remove("labels")
tokenized_datasets=ds.map(tokenize_function, batched=True, remove_columns=cols, num_proc=2)

tokenized_datasets.set_format("torch")
tokenized_datasets = (tokenized_datasets
          .map(lambda x : {"float_labels": x["labels"].to(torch.float)}, remove_columns=["labels"])
          .rename_column("float_labels", "labels"))

training_args = TrainingArguments(
    output_dir="../data/regression/output_for_preprocessed_train_data/with_HPO/results",
    overwrite_output_dir = True,
    learning_rate=1e-5,                             # config
    do_train=True,
    do_eval=True,
    evaluation_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model = 'eval_rmse',
    greater_is_better = False,
    num_train_epochs=3,                             # config
    max_steps=-1,                                   # config
    per_device_train_batch_size=2,                  # config
    per_device_eval_batch_size=10,                  # config
    warmup_steps=0,                                 # config
    weight_decay=0.1,                               # config
    logging_dir="../data/regression/output_for_preprocessed_train_data/with_HPO/logs",
    logging_steps=10,
    skip_memory_metrics=True,
    report_to="none",
    fp16=True,
    seed=12,                                        # config
)

def compute_rmse(eval_pred):
    predictions, labels = eval_pred
    return {'rmse': mean_squared_error(labels, predictions.reshape(-1), squared=False)}

trainer = Trainer(
    model_init=get_model,
    args=training_args,
    train_dataset=tokenized_datasets['train'].shard(index=1, num_shards=train_num_shards),
    eval_dataset=tokenized_datasets['valid'],
    tokenizer = tokenizer,
    compute_metrics=compute_rmse
)

tune_config = {
    "max_steps": -1,
    "per_device_eval_batch_size": 10,
    "per_device_train_batch_size": 1,
    "num_train_epochs": 3,
}

scheduler = PopulationBasedTraining(
    time_attr="training_iteration",
    metric="eval_rmse",
    mode="min",
    perturbation_interval=perturbation_interval,
    require_attrs=True,
    hyperparam_mutations={
        "learning_rate": tune.loguniform(1e-6, 1e-1),
#        "per_device_train_batch_size": tune.randint(1, 3),
        "warmup_steps": tune.randint(1, 10001),
        "weight_decay": tune.uniform(1e-2, 1),
        "seed": tune.randint(1, 1000),
    })

def objective_rmse(metrics):
    return metrics['eval_rmse']

best_run = trainer.hyperparameter_search(
    hp_space=lambda _: tune_config,
    compute_objective=objective_rmse,
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
    local_dir="../data/regression/output_for_preprocessed_train_data/with_HPO/ray_results",
    name="tune-clrp-regression",
    log_to_file=True
)

for n, v in best_run.hyperparameters.items():
    setattr(trainer.args, n, v)

trainer.train()
trainer.save_model('../data/regression/output_for_preprocessed_train_data/with_HPO/models/roberta_large_1')

print('*'*100)
print(best_run)
print('*'*100)

ray.shutdown()
