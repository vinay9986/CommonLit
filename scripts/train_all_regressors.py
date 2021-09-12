#!/usr/bin/env python

import transformers
import datasets
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

import numpy as np
import pandas as pd
import random
import copy
import os
import gc
from tqdm import tqdm
from sklearn.metrics import mean_squared_error

import logging
logging.basicConfig(filename='train_all_regressors.log', level=logging.INFO, filemode='w', datefmt='%m/%d/%Y %I:%M:%S %p', format='%(asctime)s | %(levelname)s | %(message)s')

msg0 = "Running on transformers v"+str(transformers.__version__)+", datasets v"+str(datasets.__version__)+" and torch v"+str(torch.__version__)
logging.info(msg0)

pd_train = pd.read_csv('../data/preprocessed_train_data.csv', low_memory=False)
columns = pd_train.columns.tolist()
columns.remove('text')
columns.remove('labels')
pd_train.drop(columns=columns, inplace=True, axis=1)


data_seed = 912
train_testeval=(datasets.Dataset.from_pandas(df=pd_train)).train_test_split(train_size=.75, test_size=.25, seed=data_seed)
test_eval=train_testeval['test'].train_test_split(train_size=.75, test_size=.25, seed=data_seed)
ds=datasets.DatasetDict({
    'train': train_testeval['train'],
    'test': test_eval['test'],
    'valid': test_eval['train']})

model_ckpt = "roberta-large"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt, problem_type="regression")
def tokenize_and_encode(examples):
    return tokenizer.batch_encode_plus(examples["text"], truncation=True, padding=True, max_length=256)

cols = ds["train"].column_names
cols.remove("labels")
ds_enc = ds.map(tokenize_and_encode, batched=True, remove_columns=cols, num_proc=2)
ds_enc.set_format("torch")
ds_enc = (ds_enc
          .map(lambda x : {"float_labels": x["labels"].to(torch.float)}, remove_columns=["labels"])
          .rename_column("float_labels", "labels"))

model_paths = [
    '/mnt/1TB/workspace/comp/Commonlit_kaggle_1/data/mlm/output_for_preprocessed_train_data/without_HPO/models/roberta_large',
    '/mnt/1TB/workspace/comp/Commonlit_kaggle_1/data/mlm/output_for_preprocessed_train_data/with_HPO/models/roberta-large-1',
    '/mnt/1TB/workspace/comp/Commonlit_kaggle_1/data/mlm/output_for_preprocessed_train_data/with_HPO/models/roberta-large-2',
    '/mnt/1TB/workspace/comp/Commonlit_kaggle_1/data/mlm/output_for_preprocessed_train_data/with_HPO/models/roberta-large-3',
    '/mnt/1TB/workspace/comp/Commonlit_kaggle_1/data/mlm/output_for_preprocessed_train_data/with_HPO/models/roberta-large-4',
    '/mnt/1TB/workspace/comp/Commonlit_kaggle_1/data/mlm/output_for_preprocessed_train_data/with_HPO/models/roberta-large-5',
    '/mnt/1TB/workspace/comp/Commonlit_kaggle_1/data/mlm/output_for_preprocessed_train_data/with_HPO/models/roberta-large-6',
    '/mnt/1TB/workspace/comp/Commonlit_kaggle_1/data/mlm/output_for_preprocessed_train_data/with_HPO/models/roberta-large-7'
]

def compute_rmse(eval_pred):
    predictions, labels = eval_pred
    return {'rmse': mean_squared_error(labels, predictions.reshape(-1), squared=False)}

training_args = TrainingArguments(
    output_dir="../data/ignore",
    overwrite_output_dir = True,
    learning_rate=1e-5,                             # config
    do_train=True,
    do_eval=True,
    evaluation_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model = 'eval_rmse',
    greater_is_better = False,
    num_train_epochs=10,                            # config
    max_steps=-1,                                   # config
    per_device_train_batch_size=1,                  # config
    per_device_eval_batch_size=10,                  # config
    warmup_steps=0,                                 # config
    weight_decay=0.1,                               # config
    logging_dir="../data/ignore/",
    logging_steps=10,
    skip_memory_metrics=True,
    report_to="none",
    fp16=True,
    seed=912,                                       # config
)


for index, model_path in enumerate(tqdm(model_paths, total=len(model_paths))):
    msg1 = "Starting training on model "+str(index)+" at path "+model_path
    logging.info(msg1)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=1, problem_type="regression").to('cuda')

    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = ds_enc['train'],
        eval_dataset = ds_enc['valid'],
        compute_metrics = compute_rmse,
        tokenizer = tokenizer
    )

    eval_res = trainer.evaluate()['eval_rmse']
    msg2 = "RMSE before training "+str(eval_res)+" for model "+str(index)
    logging.info(msg2)

    trainer.train()

    eval_res = trainer.evaluate()['eval_rmse']
    msg3 = "RMSE after training "+str(eval_res)+" for model "+str(index)
    logging.info(msg3)

    test_res = trainer.predict(test_dataset=ds_enc['test'])[2]['test_rmse']
    msg4 = "RMSE on unseen data "+str(test_res)+" for model "+str(index)
    logging.info(msg4)

    msg5 = "Saving model "+str(index)+" started"
    logging.info(msg5)

    d = '/mnt/1TB/workspace/comp/Commonlit_kaggle_1/data/regression/output_for_preprocessed_train_data/without_HPO/models/roberta_large_'+str(index)
    if not os.path.exists(d):
        os.makedirs(d)
        msg6 = "Folder created at "+d
        logging.info(msg6)

    trainer.save_model(d)

    msg7 = "Saving model completed for model "+str(index)
    logging.info(msg7)

    torch.cuda.empty_cache()
    gc.collect()
logging.info("Training all models completed")
