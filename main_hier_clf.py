import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
import torch
import numpy as np
import pandas as pd
from torch import nn, optim
from data import HierDataModule
from model import HierClassifier
from transformers import AutoTokenizer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


model_type = HierClassifier

def main(args):
    checkpoint_callback = ModelCheckpoint(dirpath='checkpoints',save_top_k=1,save_weights_only=True,monitor='val_loss')
    model = model_type(**vars(args))
    if "bertweet" in args.model_type:
        tokenizer = AutoTokenizer.from_pretrained(args.model_type, normalization=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_type)
    data_module = HierDataModule(args.bs, args.input_dir, tokenizer, args.max_len)
    early_stop_callback = EarlyStopping(
        monitor='val_f1',
        patience=args.patience,
        mode="max"
    )
    
    trainer = pl.Trainer(gpus=1, callbacks=[early_stop_callback, checkpoint_callback], val_check_interval=1.0, max_epochs=100, min_epochs=1, accumulate_grad_batches=args.accumulation, gradient_clip_val=args.gradient_clip_val, deterministic=True, log_every_n_steps=100)

    if args.find_lr:
        # Run learning rate finder
        lr_finder = trainer.tuner.lr_find(model, datamodule=data_module)

        # # Results can be found in
        # print(lr_finder.results)

        # # Plot with
        # fig = lr_finder.plot(suggest=True)
        # fig.show()

        # Pick point based on plot, or get suggestion
        new_lr = lr_finder.suggestion()
        print(new_lr)

        # # update hparams of the model
        # model.hparams.lr = new_lr
    else:
        trainer.fit(model, data_module)

    return checkpoint_callback.best_model_path


def run_chunk(model, args):
    print('args',args)
    model.eval()
    if "bertweet" in args.model_type:
        tokenizer = AutoTokenizer.from_pretrained(args.model_type, normalization=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_type)

    data_module = HierDataModule(args.bs, args.input_dir, tokenizer, args.max_len)
    test_loader = data_module.val_dataloader()
    all_y_hat = []
    all_label = []
    for i, data in enumerate(test_loader):
        x, label = data
        y_hat, atten_score = model(x)
        all_y_hat += y_hat.sigmoid().tolist()
        all_label += label.tolist()

    return all_y_hat, all_label



if __name__ == "__main__":
    seed_everything(2021, workers=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="prajjwal1/bert-tiny")
    parser.add_argument("--hier", type=int, default=1)
    # parser.add_argument("--model_type", type=str, default="bert-base-uncased")
    temp_args, _ = parser.parse_known_args()
    model_type = HierClassifier
    parser = model_type.add_model_specific_args(parser)
    parser.add_argument("--bs", type=int, default=4)
    # parser.add_argument("--input_dir", type=str, default="./sst2")
    parser.add_argument("--input_dir", type=str, default="./processed/kmeans16")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--gradient_clip_val", type=float, default=0.1)
    parser.add_argument("--find_lr", action="store_true")
    args = parser.parse_args()
    
    main(args)