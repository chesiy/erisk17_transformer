from process_sentence_embedding import get_chunk_summary,get_sample_summary, get_sample_summary_v2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
import xml
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from model import HierClassifier
from transformers import AutoTokenizer
from pytorch_lightning.callbacks import ModelCheckpoint
from data import HierDataModule
from ERDE import ERDE_chunk,ERDE_sample
import numpy as np
import time
from get_extra_features import get_test_features_chunk
import torch


def main(args):
    print("training....")
    checkpoint_callback = ModelCheckpoint(dirpath='checkpoints', save_top_k=1, save_weights_only=True,
                                          monitor='val_loss')
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

    trainer = pl.Trainer(gpus=1, callbacks=[early_stop_callback, checkpoint_callback], val_check_interval=1.0,
                         max_epochs=100, min_epochs=1, accumulate_grad_batches=args.accumulation,
                         gradient_clip_val=args.gradient_clip_val, deterministic=True, log_every_n_steps=100)

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


def run_test(model, args, tokenizer, all_extra_feats):
    # print('args', args)
    model.eval()

    data_module = HierDataModule(args.bs, args.input_dir, tokenizer, args.max_len)
    test_loader = data_module.val_dataloader()
    all_y_hat = []
    all_label = []
    for i, data in enumerate(test_loader):
        x, label = data
        extra_feats = all_extra_feats[i * len(x):(i + 1) * len(x)]
        y_hat, atten_score = model(x, extra_feats)
        if len(y_hat.shape) == 0:
            y_hat = y_hat.unsqueeze(dim=0)
        # print(y_hat, label)
        all_y_hat += y_hat.sigmoid().tolist()
        all_label += label.tolist()

    return all_y_hat, all_label


def ERDE_in_chunk(model, chunk_num, args):
    postnum_recorder = {}
    for base_path in ["negative_examples_anonymous", "negative_examples_test", "positive_examples_anonymous",
                      "positive_examples_test"]:
        base_path = "dataset/" + base_path
        filenames = sorted(os.listdir(base_path))
        postnum_recorder[base_path] = []
        for fname in filenames:
            dom = xml.dom.minidom.parse(base_path + "/" + fname)
            collection = dom.documentElement
            title = collection.getElementsByTagName('TITLE')
            postnum_recorder[base_path].append(len(title))

    postnum_rec = postnum_recorder["dataset/negative_examples_test"]+postnum_recorder["dataset/positive_examples_test"]

    chunk_pred_probas=[]
    chunk_cum_posts=[]
    label=[]

    test_user_num=len(postnum_rec)

    for i in range(test_user_num):
        chunk_pred_probas.append([])
        chunk_cum_posts.append([])

    if "bertweet" in args.model_type:
        tokenizer = AutoTokenizer.from_pretrained(args.model_type, normalization=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_type)

    for chunk in range(chunk_num):
        get_chunk_summary(chunk, postnum_recorder, chunk_num)

        extra_feats = get_test_features_chunk(postnum_recorder["dataset/negative_examples_test"],postnum_recorder["dataset/positive_examples_test"], chunk_num, chunk)
        extra_feats = torch.from_numpy(extra_feats)

        y_hat, label = run_test(model, args, tokenizer, extra_feats)
        print('chunk', chunk, len(y_hat), len(label), len(postnum_rec))

        for i in range(len(label)):
            chunk_pred_probas[i].append(y_hat[i])
            chunk_cum_posts[i].append(postnum_rec[i]*(chunk+1)/chunk_num)

    chunk_pred_probas = np.array(chunk_pred_probas)
    chunk_cum_posts = np.array(chunk_cum_posts)

    print(chunk_pred_probas.shape, chunk_cum_posts.shape)

    erde5 = ERDE_chunk(chunk_pred_probas,chunk_cum_posts,label,0.5,o=5)
    erde50 = ERDE_chunk(chunk_pred_probas, chunk_cum_posts, label, 0.5, o=50)

    return erde5, erde50

def ERDE_in_sample(model, args):
    postnum_recorder = {}
    for base_path in ["negative_examples_anonymous", "negative_examples_test", "positive_examples_anonymous",
                      "positive_examples_test"]:
        base_path = "dataset/" + base_path
        filenames = sorted(os.listdir(base_path))
        postnum_recorder[base_path] = []
        for fname in filenames:
            dom = xml.dom.minidom.parse(base_path + "/" + fname)
            collection = dom.documentElement
            title = collection.getElementsByTagName('TITLE')
            postnum_recorder[base_path].append(len(title))

    postnum_rec = postnum_recorder["dataset/negative_examples_test"] + postnum_recorder[
        "dataset/positive_examples_test"]

    sample_pred_probas = []
    label = []

    test_user_num = len(postnum_rec)

    for i in range(test_user_num):
        sample_pred_probas.append([])

    if "bertweet" in args.model_type:
        tokenizer = AutoTokenizer.from_pretrained(args.model_type, normalization=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_type)

    for sam in range(1,151):
        get_sample_summary(sam)

        y_hat, label = run_test(model, args, tokenizer)
        print('sample', sam, len(y_hat), len(label), len(postnum_rec))

        for i in range(len(label)):
            if len(sample_pred_probas[i]) < postnum_rec[i]:
                sample_pred_probas[i].append(y_hat[i])

    erde5 = ERDE_sample(sample_pred_probas, label, o=5)
    erde50 = ERDE_sample(sample_pred_probas, label, o=50)

    return erde5, erde50


def ERDE_in_sample_v2(model, args):
    if "bertweet" in args.model_type:
        tokenizer = AutoTokenizer.from_pretrained(args.model_type, normalization=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_type)

    sample_pred_probas = []
    labels = []

    n = 0
    for base_path in ["negative_examples_test", "positive_examples_test"]:
        base_path = "dataset/" + base_path
        filenames = sorted(os.listdir(base_path))
        for file in filenames:
            get_sample_summary_v2(file,base_path)

            y_hat, label = run_test(model, args, tokenizer)
            sample_pred_probas.append(list(y_hat))
            labels.append(label[0])
            n += 1
            print('user:',n)
            # if n == 4:
            #     break

    print(sample_pred_probas)
    print(labels)
    erde5 = ERDE_sample(sample_pred_probas, labels, o=5)
    erde50 = ERDE_sample(sample_pred_probas, labels, o=50)

    return erde5, erde50


if __name__ == '__main__':
    seed_everything(2021, workers=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="prajjwal1/bert-tiny")
    parser.add_argument("--hier", type=int, default=1)

    parser.add_argument("--chunk",type=int, default=1)
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

    # best_checkpoint_path = main(args)
    best_checkpoint_path = './checkpoints/epoch=4-step=609.ckpt'

    start=time.time()

    model = HierClassifier.load_from_checkpoint(best_checkpoint_path)

    if args.chunk==1:
        print("test in chunk....")
        erde5, erde50 = ERDE_in_chunk(model, 10, args)
        print('erde5:',erde5,'erde50:',erde50)
    else:
        print("test in sample....")
        # erde5, erde50 = ERDE_in_sample(model, args)
        erde5, erde50 = ERDE_in_sample_v2(model, args)
        print('erde5:',erde5,'erde50:',erde50)

    end=time.time()

    print('time:',end-start)
