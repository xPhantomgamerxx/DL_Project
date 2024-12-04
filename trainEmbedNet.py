#!/usr/bin/python
#-*- coding: utf-8 -*-

import sys
import time
import os
import argparse
import pdb
import glob
import datetime
import numpy
import logging
import torch
from EmbedNet import *
from early_stopping import *
from DatasetLoader import get_data_loader
from sklearn import metrics
import torchvision.transforms as transforms

# ## ===== ===== ===== ===== ===== ===== ===== =====
# ## Parse arguments
# ## ===== ===== ===== ===== ===== ===== ===== =====

parser = argparse.ArgumentParser(description = "Face Recognition Training");

## Data loader
parser.add_argument('--batch_size',         type=int, default=100,	help='Batch size, defined as the number of classes per batch, EfficientNet: 85, MobileNet: default, DenseNet:32');
parser.add_argument('--max_img_per_cls',    type=int, default=500,	help='Maximum number of images per class per epoch');
parser.add_argument('--nDataLoaderThread',  type=int, default=5, 	help='Number of data loader threads');

## Training details
parser.add_argument('--test_interval',  type=int,   default=1,      help='Test and save every [test_interval] epochs');
parser.add_argument('--max_epoch',      type=int,   default=25,     help='Maximum number of epochs');
parser.add_argument('--trainfunc',      type=str,   default="softmax",  help='Loss function to use');

## Optimizer
parser.add_argument('--optimizer',      type=str,   default="adam", help='Optimizer');
parser.add_argument('--scheduler',      type=str,   default="steplr", help='Learning rate scheduler');
parser.add_argument('--lr',             type=float, default=0.001,  help='Initial learning rate');
parser.add_argument("--lr_decay",       type=float, default=0.90,   help='Learning rate decay every [test_interval] epochs');
parser.add_argument('--weight_decay',   type=float, default=0,      help='Weight decay in the optimizer');

## Loss functions
parser.add_argument('--margin',         type=float, default=0.1,    help='Loss margin, only for some loss functions');
parser.add_argument('--scale',          type=float, default=30,     help='Loss scale, only for some loss functions');
parser.add_argument('--nPerClass',      type=int,   default=1,      help='Number of images per class per batch, only for metric learning based losses');
parser.add_argument('--nClasses',       type=int,   default=9500,   help='Number of classes in the softmax layer, only for softmax-based losses');

## Load and save
parser.add_argument('--initial_model',  type=str,   default="",     help='Initial model weights, otherwise initialise with random weights');
parser.add_argument('--save_path',      type=str,   default="exps/exp1", help='Path for model and logs');

## Training and evaluation data
parser.add_argument('--train_path',     type=str,   default="/mnt/home/joonson/ee488_24_data/train1",   help='Absolute path to the train set');
parser.add_argument('--train_ext',      type=str,   default="jpg",  help='Training files extension');
parser.add_argument('--test_path',      type=str,   default="/mnt/home/joonson/ee488_24_data/val",     help='Absolute path to the test set');
parser.add_argument('--test_list',      type=str,   default="/mnt/home/joonson/ee488_24_data/val_pairs.csv",   help='Evaluation list');

## Model definition
parser.add_argument('--model',          type=str,   default="ResNet18", help='Name of model definition');
parser.add_argument('--nOut',           type=int,   default=512,    help='Embedding size in the last FC layer');

## For test only
parser.add_argument('--eval',           dest='eval', action='store_true',   help='Eval only')
parser.add_argument('--output',         type=str,   default="",     help='Save a log of output to this file name');

## Training
parser.add_argument('--gpu',            type=int,   default=3,      help='GPU index');

args = parser.parse_args();

## ===== ===== ===== ===== ===== ===== ===== =====
## Script to compute EER
## ===== ===== ===== ===== ===== ===== ===== =====

def compute_eer(all_labels,all_scores):

    # compute receiver operating characteristic (ROC) for binary classification
    fpr, tpr, thresholds = metrics.roc_curve(all_labels, all_scores)

    # calculate false negative rate (FNR)
    fnr = 1 - tpr

    # calculate equal error rate (EER). The EER is the error rate at which FNR is equal to FPR.
    eer_threshold_index = numpy.nanargmin(numpy.abs(fnr - fpr))
    EER = fpr[eer_threshold_index]

    return EER

## ===== ===== ===== ===== ===== ===== ===== =====
## Trainer script
## ===== ===== ===== ===== ===== ===== ===== =====

def main_worker(args):

    early_stopping = EarlyStopping(patience=5, min_delta=0.01, path=args.save_path + '/best_model_checkpoint.pt')
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(args.save_path+"/scores.txt", mode="a+"),
        ],
        level=logging.DEBUG,
        format='[%(levelname)s] :: %(asctime)s :: %(message)s', 
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ## Load models
    model = EmbedNet(**vars(args)).cuda();

    ep          = 1

    ## Input transformations for training (you can change if you like)
    train_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize(256),
         transforms.RandomCrop([224,224]),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    train2_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(p=0.5),  
        transforms.RandomRotation(15),          
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  
        transforms.RandomCrop([224, 224]),      
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
    ])

    ## Input transformations for evaluation (you can change if you like)
    test_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize(256),
         transforms.CenterCrop([224,224]),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    ## Initialise trainer and data loader
    trainLoader = get_data_loader(transform=train_transform, **vars(args)); # change the train transform between the train1 and train2 tranform
    trainer     = ModelTrainer(model, **vars(args))

    ## Load model weights
    modelfiles = glob.glob('{}/epoch0*.model'.format(args.save_path))
    modelfiles.sort()

    ## If the target directory already exists, start from the existing file
    if len(modelfiles) >= 1:
        trainer.loadParameters(modelfiles[-1]);
        print("Model {} loaded from previous state!".format(modelfiles[-1]));
        ep = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][5:]) + 1
    elif(args.initial_model != ""):
        trainer.loadParameters(args.initial_model);
        print("Model {} loaded!".format(args.initial_model));

    ## If the current iteration is not 1, update the scheduler
    for ii in range(1,ep):
        trainer.__scheduler__.step()

    ## Print total number of model parameters
    pytorch_total_params = sum(p.numel() for p in model.__E__.parameters())
    print('Total model parameters: {:,}'.format(pytorch_total_params))
    
    ## Evaluation code 
    if args.eval == True:
        sc, lab, trials = trainer.evaluateFromList(transform=test_transform, **vars(args))
        EER = compute_eer(lab, sc)
        print('EER {:.2f}%'.format(EER*100))
        if args.output != '':
            with open(args.output,'w') as f:
                for ii in range(len(sc)):
                    f.write('{:4f},{:d},{}\n'.format(sc[ii],lab[ii],trials[ii]))
        quit();

    ## Log arguments
    logger.info('{}'.format(args))

    ## Core training script
    for ep in range(ep,args.max_epoch+1):
        clr = [x['lr'] for x in trainer.__optimizer__.param_groups]
        logger.info("Epoch {:04d} started with LR {:.5f} ".format(ep,max(clr)));

        # train step
        loss = trainer.train_network(trainLoader);
        logger.info("Epoch {:04d} completed with TLOSS {:.5f}".format(ep,loss));

        if ep % args.test_interval == 0:
            # validation step
            sc, lab, trials = trainer.evaluateFromList(transform=test_transform, **vars(args))
            EER = compute_eer(lab, sc)
            logger.info("Epoch {:04d}, Val EER {:.2f}%".format(ep, EER*100));

            # save model params
            trainer.saveParameters(args.save_path+"/epoch{:04d}.model".format(ep));

            # Early Stopping Check
            val_loss = EER  
            early_stopping(val_loss, model)

            if early_stopping.early_stop:
                print("Early stopping triggered. Stopping training.")
                break

# ## ===== ===== ===== ===== ===== ===== ===== =====
# ## Main function
# ## ===== ===== ===== ===== ===== ===== ===== =====


def main():

    os.environ["CUDA_VISIBLE_DEVICES"]='{}'.format(args.gpu)
            
    if not(os.path.exists(args.save_path)):
        os.makedirs(args.save_path)

    main_worker(args)


if __name__ == '__main__':
    main()