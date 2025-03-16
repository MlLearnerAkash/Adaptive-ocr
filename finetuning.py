
# import os
# import pdb
# import pickle
# import logging

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import random_split
# from argparse import ArgumentParser

# from src.modules.trainer import OCRTrainer
# from src.utils.utils import EarlyStopping, gmkdir
# from src.models.crnn import CRNN
# from src.options.opts import base_opts
# from src.data.synth_dataset import SynthDataset, SynthCollator
# from src.data.pickle_dataset import PickleDataset
# from src.criterions.ctc import CustomCTCLoss 
# from src.utils.top_sampler import SamplingTop

# class Learner(object):
#     def __init__(self, model, optimizer, savepath=None, resume=False, pretrained=None):  # Modified
#         self.model = model
#         self.optimizer = optimizer
#         self.savepath = os.path.join(savepath, 'best.ckpt')
#         self.cuda = torch.cuda.is_available()
#         self.cuda_count = torch.cuda.device_count()
        
#         # Load pretrained weights
#         if pretrained:  # New block
#             print(f"Loading pretrained weights from {pretrained}")
#             checkpoint = torch.load(pretrained)
#             state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
#             # Handle DataParallel wrapping
#             state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
#             self.model.load_state_dict(state_dict, strict=False)
            
#         if self.cuda:
#             self.model = self.model.cuda()
#         self.epoch = 0
#         if self.cuda_count > 1:
#             print("Let's use", torch.cuda.device_count(), "GPUs!")
#             self.model = nn.DataParallel(self.model)
#         self.best_score = None
#         if resume and os.path.exists(self.savepath):
#             self.checkpoint = torch.load(self.savepath)
#             self.epoch = self.checkpoint['epoch']
#             self.best_score=self.checkpoint['best']
#             self.load()
#         else:
#             print('checkpoint does not exist')
#     def fit(self, opt):
#         opt.cuda = self.cuda
#         opt.model = self.model
#         opt.optimizer = self.optimizer
#         logging.basicConfig(filename="%s/%s.csv" %(opt.log_dir, opt.name), level=logging.INFO)
#         self.saver = EarlyStopping(self.savepath, patience=15, verbose=True, best_score=self.best_score)
#         opt.epoch = self.epoch
#         trainer = OCRTrainer(opt)
        
#         for epoch in range(opt.epoch, opt.epochs):
#             train_result = trainer.run_epoch()
#             val_result = trainer.run_epoch(validation=True)
#             trainer.count = epoch
#             info = '%d, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f'%(epoch, train_result['train_loss'], 
#                 val_result['val_loss'], train_result['train_ca'],  val_result['val_ca'],
#                 train_result['train_wa'], val_result['val_wa'])
#             logging.info(info)
#             self.val_loss = val_result['val_loss']
#             print(self.val_loss)
#             if self.savepath:
#                 self.save(epoch)
#             if self.saver.early_stop:
#                 print("Early stopping")
#                 break

#     def load(self):
#         print('Loading checkpoint at {} trained for {} epochs'.format(self.savepath, self.checkpoint['epoch']))
#         self.model.load_state_dict(self.checkpoint['state_dict'])
#         if 'opt_state_dict' in self.checkpoint.keys():
#             print('Loading optimizer')
#             self.optimizer.load_state_dict(self.checkpoint['opt_state_dict'])

#     def save(self, epoch):
#         self.saver(self.val_loss, epoch, self.model, self.optimizer)


# if __name__ == '__main__':
#     parser = ArgumentParser()
#     base_opts(parser)
#     # Add new argument for pretrained model
#     parser.add_argument('--pretrained', type=str, default=None, 
#                       help='Path to pretrained model checkpoint')
#     args = parser.parse_args()
    

#     data = PickleDataset(args)
#     args.collate_fn = SynthCollator()
#     train_split = int(0.001*len(data))
#     val_split = len(data) - train_split
#     args.data_train, args.data_val = random_split(data, (train_split, val_split))
#     print('Traininig Data Size:{}\nVal Data Size:{}'.format(
#         len(args.data_train), len(args.data_val)))
    
#     args.alphabet = """Only thewigsofrcvdampbkuq.$A-210xT5'MDL,RYHJ"ISPWENj&BC93VGFKz();#:!7U64Q8?+*ZX/%""" 
#     args.nClasses = len(args.alphabet)
#     model = CRNN(args)
#     args.criterion = CustomCTCLoss()
#     savepath = os.path.join(args.save_dir, args.name)
#     gmkdir(savepath)
#     gmkdir(args.log_dir)
#     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
#     learner = Learner(model, optimizer, savepath=savepath, resume=args.resume, pretrained= args.pretrained)
#     learner.fit(args)




#!/usr/bin/env python3
import os
import pdb
import pickle
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from argparse import ArgumentParser
import csv

from src.modules.trainer import OCRTrainer
from src.utils.utils import EarlyStopping, gmkdir
from src.models.crnn import CRNN
from src.options.opts import base_opts
from src.data.synth_dataset import SynthDataset, SynthCollator
from src.data.pickle_dataset import PickleDataset
from src.criterions.ctc import CustomCTCLoss 
from src.utils.top_sampler import SamplingTop

class Learner(object):
    def __init__(self, model, optimizer, savepath=None, resume=False, pretrained=None, freeze_layers=0):
        """
        Args:
            model: The neural network model.
            optimizer: The optimizer.
            savepath: Directory path to save checkpoint.
            resume: If True, resume training from existing checkpoint.
            pretrained: Path to pretrained model checkpoint (if any).
            freeze_layers: Number of layers (starting from layer 0) to freeze.
        """
        self.model = model
        self.optimizer = optimizer
        self.savepath = os.path.join(savepath, 'best.ckpt')
        self.cuda = torch.cuda.is_available()
        self.cuda_count = torch.cuda.device_count()
        
        # Load pretrained weights if provided
        if pretrained:
            print(f"Loading pretrained weights from {pretrained}")
            checkpoint = torch.load(pretrained)
            state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            # Remove potential DataParallel wrapper keys
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict, strict=False)
        
        # Freeze the first 'freeze_layers' child modules if requested
        if freeze_layers > 0:
            print(f"Freezing first {freeze_layers} layers of the model")
            children = list(self.model.children())
            for i, child in enumerate(children):
                if i > freeze_layers:
                    for param in child.parameters():
                        param.requires_grad = False
                    print(f"Frozen layer {i}: {child.__class__.__name__}")
                    
        if self.cuda:
            self.model = self.model.cuda()
        self.epoch = 0
        if self.cuda_count > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)
        self.best_score = None
        if resume and os.path.exists(self.savepath):
            self.checkpoint = torch.load(self.savepath)
            self.epoch = self.checkpoint['epoch']
            self.best_score = self.checkpoint['best']
            self.load()
        else:
            print('Checkpoint does not exist')
            


    def fit(self, opt):
        opt.cuda = self.cuda
        opt.model = self.model
        opt.optimizer = self.optimizer

        log_file_path = os.path.join(opt.save_dir, opt.name, "experiment_metrics.csv")
        logging.basicConfig(filename="%s/%s.csv" % (opt.log_dir, opt.name), level=logging.INFO)
        self.saver = EarlyStopping(self.savepath, patience=15, verbose=True, best_score=self.best_score)
        opt.epoch = self.epoch
        trainer = OCRTrainer(opt)

        # Create CSV file and write header if it doesn't exist
        if not os.path.exists(log_file_path):
            with open(log_file_path, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["epoch", "train_loss", "val_loss", "train_ca", "val_ca", "train_wa", "val_wa"])

        for epoch in range(opt.epoch, opt.epochs):
            train_result = trainer.run_epoch()
            val_result = trainer.run_epoch(validation=True)
            trainer.count = epoch

            # Prepare log entry
            info = [
                epoch, 
                train_result["train_loss"], val_result["val_loss"],
                train_result["train_ca"], val_result["val_ca"],
                train_result["train_wa"], val_result["val_wa"]
            ]

            # Append results to CSV file
            with open(log_file_path, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(info)

            # Log the same info in logging system
            logging.info(", ".join(map(str, info)))

            self.val_loss = val_result["val_loss"]
            print(f"Epoch {epoch} - Val Loss: {self.val_loss}")

            if self.savepath:
                self.save(epoch)
            if self.saver.early_stop:
                print("Early stopping")
                break

    def load(self):
        print('Loading checkpoint at {} trained for {} epochs'.format(self.savepath, self.checkpoint['epoch']))
        self.model.load_state_dict(self.checkpoint['state_dict'])
        if 'opt_state_dict' in self.checkpoint.keys():
            print('Loading optimizer')
            self.optimizer.load_state_dict(self.checkpoint['opt_state_dict'])

    def save(self, epoch):
        self.saver(self.val_loss, epoch, self.model, self.optimizer)


if __name__ == '__main__':
    parser = ArgumentParser()
    base_opts(parser)
    # Add new argument for pretrained model
    parser.add_argument('--pretrained', type=str, default=None, 
                        help='Path to pretrained model checkpoint')
    # Add new argument to control the number of layers to freeze (starting from layer 0)
    parser.add_argument('--freeze_layers', type=int, default=0, 
                        help='Number of layers to freeze starting from layer 0')
    parser.add_argument("--val_dir", type=str, default = "")
    args = parser.parse_args()
    
    train_data = PickleDataset(args, mode = "train")
    val_data = PickleDataset(args, mode = "val")
    args.collate_fn = SynthCollator()
    train_split = int(0.5 * len(train_data))
    val_split = len(train_data) - train_split
    train_data, _ = random_split(train_data, (train_split, val_split))
    args.data_train, args.data_val = train_data , val_data #random_split(data, (train_split, val_split))
    print('Training Data Size: {}\nVal Data Size: {}'.format(len(args.data_train), len(args.data_val)))
    
    args.alphabet = """Only thewigsofrcvdampbkuq.$A-210xT5'MDL,RYHJ"ISPWENj&BC93VGFKz();#:!7U64Q8?+*ZX/%""" 
    args.nClasses = len(args.alphabet)
    model = CRNN(args)
    args.criterion = CustomCTCLoss()
    savepath = os.path.join(args.save_dir, args.name)
    gmkdir(savepath)
    gmkdir(args.log_dir)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    learner = Learner(model, optimizer, savepath=savepath, resume=args.resume, 
                      pretrained=args.pretrained, freeze_layers=args.freeze_layers)
    learner.fit(args)
