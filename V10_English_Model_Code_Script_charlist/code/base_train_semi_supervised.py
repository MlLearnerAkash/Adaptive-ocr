from __future__ import print_function
from __future__ import division

import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from torch.nn import CTCLoss
import os, shutil
import pdb
import pickle
import cv2
import logging
import math
import pandas as pd
import time
import torchvision
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR, StepLR
from torchvision.transforms import Compose
from torch.utils.data import Dataset


import utils
import datasets.dataset as dataset
from config import *
from datasets.ae_transforms import *
from tools.logger import Writer
from models.model import ModelBuilder
from create_lmdb import LMDBCreator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def train_params(parameters, opt_class, sch_class, opt_args, sch_args):
    optimizers = []
    schedulers = []
    for i in range(len(parameters)):
        optimizers.append(opt_class[i](parameters[i], **opt_args[i]))
        if sch_class[i]:
            schedulers.append(sch_class[i](optimizers[i], **sch_args[i]))
    return optimizers, schedulers

def weights_init(m):
    classname = m.__class__.__name__
    # print(f'Random weight initialization in {classname}')
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def check_data(data_loader, name='sample'):
        data_iter = iter(data_loader)
        data = next(data_iter) #data_iter.next()
        cpu_images = data[0]
        cpu_texts = data[1]
        nim = min(16, cpu_images.size(0))
        out = torchvision.utils.make_grid(cpu_images[:nim], nrow=1)
        out = out.permute(1, 2, 0)
        # print(f'Pixel range {name}: ', cpu_images[0].max(), cpu_images[0].min())
        out = (out*128 + 128).cpu().numpy()
        #cv2.imwrite('/ssd_scratch/cvit/ajoy/printed/home/temp/{}.jpg'.format(name), out)
        return

def write_info(model, opt):
    with open('{}/model.info'.format(opt.node_dir), 'w') as f:
        # f.write(str(opt)+'\n')
        for arg, value in sorted(vars(opt).items()):
            f.write(f"Argument {arg}: {value}\n")

        f.write('\n\nModel Architecture\n\n')
        f.write(str(model)+'\n')
    return

class CustomImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        """
        Args:
            image_paths (list): List of image file paths.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Get the image file path
        img_path = self.image_paths[idx]
        # Open image using PIL
        image = Image.open(img_path).convert('L')
        
        # Apply transformation if provided
        if self.transform:
            image = self.transform(image)
        
        # Return a dictionary with image data and its file path
        sample = {
            'image': image,
            'image_path': img_path
        }
        dummy_label = img_path
        return image, dummy_label, img_path

class BaseHTR(object):
    def __init__(self, opt, dataset_name='iam', reset_log=False):
        self.opt = opt
        self.mode = self.opt.mode
        self.dataset_name = dataset_name
        self.stn_nc = self.opt.stn_nc
        self.cnn_nc = self.opt.cnn_nc
        self.nheads = self.opt.nheads
        self.criterion = CTCLoss(blank=0, reduction='sum', zero_infinity=True)
        self.label_transform = self.init_label_transform()
        self.test_transforms = self.init_test_transforms()
        self.train_transforms = self.init_train_transforms()
        self.val1_iter = self.opt.val1_iter # Number of train data batches that will be validated
        self.val2_iter = self.opt.val2_iter # Number of validation data batches that will be validated
        self.stn_attn = None
        self.val_metric = 'cer'
        self.use_loc_bn = False
        self.CNN = 'ResCRNN'
        self.loc_block = 'LocNet'
        self.identity_matrix = torch.tensor([1, 0, 0, 0, 1, 0],
                                       dtype=torch.float).cuda()
        
        ##################################################################
        #training set
        if self.mode == 'train':
            if len(self.opt.trainRoot) == 0:
                #self.train_root = "/ssd_scratch/cvit/santhoshini/{}-train-lmdb".format(self.dataset_name)
                print("empty training set")
            else:
                self.train_root = self.opt.trainRoot

        ##################################################################
        #validation set
        if len(self.opt.valRoot) == 0:
            #self.test_root = "/ssd_scratch/cvit/santhoshini/{}-test-lmdb".format(self.dataset_name)
            print("empty validation set")
        else:
            self.test_root = self.opt.valRoot

        if not os.path.exists(self.opt.node_dir):
            os.makedirs(self.opt.node_dir)
        elif reset_log:
            shutil.rmtree(self.opt.node_dir)
            os.makedirs(self.opt.node_dir)

        random.seed(self.opt.manualSeed)
        np.random.seed(self.opt.manualSeed)
        torch.manual_seed(self.opt.manualSeed)

        # cudnn.benchmark = True
        cudnn.deterministic = True
        cudnn.benchmark = False
        cudnn.enabled = True
        # print('CudNN enabled', cudnn.enabled)

        if torch.cuda.is_available() and not self.opt.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            self.opt.gpu_id = list(map(int, self.opt.gpu_id.split(',')))
            torch.cuda.set_device(self.opt.gpu_id[0])


    def run(self):
        if self.mode == "train":
            # print(self.train_root, self.test_root)
            self.train_data, self.train_loader = self.get_data_loader(self.train_root,
                                                                      self.train_transforms,
                                                                      self.label_transform)
            self.test_data, self.test_loader = self.get_data_loader(self.test_root,
                                                                      self.test_transforms,
                                                                      self.label_transform)
            self.converter = utils.strLabelConverter(self.test_data.id2char,
                                                     self.test_data.char2id,
                                                     self.test_data.ctc_blank)
            check_data(self.train_loader, '{}train'.format(self.dataset_name))
            check_data(self.test_loader, '{}val'.format(self.dataset_name))
            self.nclass = self.test_data.rec_num_classes
            self.model, self.parameters = self.get_model()
            self.init_variables()
            self.init_train_params()
            print('Classes: ', self.test_data.voc)
            print('#Train Samples: ', self.train_data.nSamples)
            print('#Val Samples: ', self.test_data.nSamples)
            self.train()
        elif self.mode == "test":
            self.test_data, self.test_loader = self.get_data_loader(self.test_root,
                                                                      self.test_transforms,
                                                                      self.label_transform)
            self.converter = utils.strLabelConverter(self.test_data.id2char,
                                                     self.test_data.char2id,
                                                     self.test_data.ctc_blank)
            check_data(self.test_loader, '{}test'.format(self.dataset_name))
            self.nclass = self.test_data.rec_num_classes
            self.model, self.parameters = self.get_model()
            self.init_variables()
            print('Classes: ', self.test_data.voc)
            print('#Test Samples: ', self.test_data.nSamples)
            self.eval(self.test_data)

    def init_train_transforms(self):
        T = Compose([Rescale((self.opt.imgH, self.opt.imgW)),ElasticTransformation(0.7),ToTensor()])
        return T

    def init_test_transforms(self):
        T = Compose([Rescale((self.opt.imgH, self.opt.imgW)),ToTensor()])
        return T

    def init_label_transform(self):
        T = None
        return T

    def init_variables(self):
        self.image = torch.FloatTensor(self.opt.batchSize, 3, self.opt.imgH, self.opt.imgH)
        self.text = torch.LongTensor(self.opt.batchSize * 5)
        self.length = torch.LongTensor(self.opt.batchSize)
        if self.opt.cuda:
            self.image = self.image.cuda()
            self.criterion = self.criterion.cuda()
            self.text = self.text.cuda()
            self.length = self.length.cuda()
        self.image = Variable(self.image)
        self.text = Variable(self.text)
        self.length = Variable(self.length)

    def init_train_params(self):
        if self.opt.adam:
            self.optimizer = optim.Adam(self.parameters, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        elif self.opt.adadelta:
            self.optimizer = optim.Adadelta(self.parameters, lr=self.opt.lr)#, weight_decay = 0.001
        elif self.opt.rmsprop:
            self.optimizer = optim.RMSprop(self.parameters, lr=self.opt.lr)
        else:
            self.optimizer = optim.SGD(self.parameters, lr=self.opt.lr, momentum=self.opt.momentum)

        if self.opt.StepLR:
            self.scheduler = StepLR(self.optimizer, step_size=20, gamma=0.5)
        else:
            self.scheduler = None
        # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.00001, max_lr=0.001,
        #                                             cycle_momentum=False)
        print(self.optimizer)
        return

    def get_model(self):
        crnn = ModelBuilder(self.opt.imgH, self.opt.imgW, self.opt.tps_inputsize,
                        self.opt.tps_outputsize, self.opt.num_control_points, self.opt.tps_margins, self.opt.stn_activation,
                        self.opt.nh, self.stn_nc, self.cnn_nc, self.nclass, STN_type=self.opt.STN_type,
                        nheads=self.nheads, stn_attn=self.stn_attn, use_loc_bn=self.use_loc_bn, loc_block = self.loc_block,
                        CNN=self.CNN)
        if self.opt.cuda:
            crnn.cuda()
            crnn = torch.nn.DataParallel(crnn, device_ids=self.opt.gpu_id, dim=1)
        else:
            crnn = torch.nn.DataParallel(crnn, device_ids=self.opt.gpu_id)
        if self.opt.pretrained != '':
            if self.opt.transfer:
                d_params = crnn.state_dict()
                s_params = torch.load(self.opt.pretrained)
                for name1 in s_params:
                    param1 = s_params[name1]
                    try:
                        d_params[name1].data.copy_(param1.data)
                    except:
                        print('Skipping weight ', name1)
                        continue
                crnn.load_state_dict(d_params)
            else:
                print('Using pretrained model', self.opt.pretrained)
                crnn.load_state_dict(torch.load(self.opt.pretrained))
                # # Freeze all model parameters
                # for param in crnn.parameters():
                #     param.requires_grad = False

                # # Unfreeze only the CNN module parameters
                # if hasattr(crnn, 'module'):
                #     # If crnn is wrapped in DataParallel, access the underlying model
                #     for param in crnn.module.crnn.cnn.parameters():
                #         param.requires_grad = True
                # else:
                #     for param in crnn.crnn.cnn.parameters():
                #         param.requires_grad = True
                # Get the CNN module whether the model is wrapped in DataParallel or not
                if hasattr(crnn, 'module'):
                    cnn = crnn.module.crnn.cnn
                else:
                    cnn = crnn.crnn.cnn

                # Iterate through the layers in the CNN module and freeze only the first 10 layers.
                # for i, layer in enumerate(cnn.children()):
                #     if i < 10:
                #         # Freeze parameters of the first 10 layers
                #         for param in layer.parameters():
                #             param.requires_grad = True
                #     else:
                #         # Ensure parameters in later layers remain trainable
                #         for param in layer.parameters():
                #             param.requires_grad = False
                
        else:
            crnn.apply(weights_init)
        return crnn, crnn.parameters()

    def get_data_loader(self, root, im_transforms, label_transforms, num_samples=np.inf):
        data = dataset.lmdbDataset(root=root, voc=self.opt.alphabet, num_samples=num_samples,
                                   transform=im_transforms, label_transform=label_transforms,
                                   voc_type=self.opt.alphabet_type, lowercase=self.opt.lowercase,
                                   alphanumeric=self.opt.alphanumeric, return_list=True)
        if not self.opt.random_sample:
            sampler = dataset.randomSequentialSampler(data, self.opt.batchSize)
        else:
            sampler = None
        data_loader = torch.utils.data.DataLoader(data, batch_size=self.opt.batchSize,
                                                shuffle=True, sampler=sampler,
                                                num_workers=int(self.opt.workers),
                                                collate_fn=dataset.collatedict(),drop_last=False)
        return data, data_loader
    


 

    #NOTE: min-max thresh
    def eval_confidence(self, image_dir, target_dir, max_iter=np.inf, max_conf=1.0, min_conf=0.7):
        import os, glob, shutil
        import torch
        from torch.autograd import Variable
        import torch.nn.functional as F
        import numpy as np
        import pandas as pd
        import random

        # Create a list of image file paths from the input directory (recursively)
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
        image_paths = []
        # for ext in image_extensions:
        #     image_paths.extend(glob.glob(os.path.join(image_dir, '**', ext), recursive=True))
        
        # if len(image_paths) == 0:
        #     print("No images found in directory:", image_dir)
        #     return None

        with open(image_dir, 'r') as f:
            image_paths = [line.strip() for line in f if line.strip().lower().endswith(image_extensions)]

        if len(image_paths) == 0:
            print("No images found in file:", image_dir)
            return None

        # Create a custom dataset
        dataset1 = CustomImageDataset(image_paths, transform=self.test_transforms)
        
        # Create DataLoader
        data_loader = torch.utils.data.DataLoader(dataset1, 
                                                batch_size=self.opt.batchSize,
                                                num_workers=int(self.opt.workers),
                                                pin_memory=True,
                                                collate_fn=dataset.collatedict(),
                                                drop_last=False)
        
        self.model.eval()
        image_confidences = []
        iter_loader = iter(data_loader)
        max_iter = min(max_iter, len(data_loader))
        
        with torch.no_grad():
            for i in range(max_iter):
                if self.opt.mode == 'test':
                    print('%d / %d' % (i, len(data_loader)), end='\r')
                
                batch = next(iter_loader)
                output_dict = self.forward_sample(batch)
                batch_size = output_dict['batch_size']
                
                # Get probabilities and decoded predictions
                log_probs = F.log_softmax(output_dict['probs'], 2)
                probs = torch.exp(log_probs)  # Convert to probabilities
                preds_size = torch.IntTensor([log_probs.size(0)] * batch_size)
                decoded_pred, preds2 = self.decoder(log_probs, preds_size)
                
                # Calculate sequence probabilities for each sample in batch
                seq_probs = []
                for j in range(batch_size):
                    # Get the predicted path for this sample
                    path = preds2[j].cpu().numpy()  # [time_steps,]
                    
                    # Collect probabilities along the decoded path (excluding blanks)
                    valid_probs = []
                    for t in range(log_probs.size(0)):  # Iterate through time steps
                        char_idx = path[t]
                        if char_idx != 0:  # Skip blank characters
                            prob = probs[t, j, char_idx].item()
                            valid_probs.append(prob)
                    
                    # Calculate average probability for non-blank characters
                    if valid_probs:
                        seq_prob = sum(valid_probs) / len(valid_probs)
                    else:
                        seq_prob = 0.0
                    seq_probs.append(seq_prob)
                
                # Store results with path-based confidence
                for j in range(batch_size):
                    image_path = batch[1][j]
                    image_confidences.append((
                        image_path,
                        torch.tensor(seq_probs[j]),  # Confidence stored as tensor
                        decoded_pred[j]
                    ))
        
        # Filter samples with confidence between 0.7 and 1.0
        filtered_confidences = [sample for sample in image_confidences if min_conf <= sample[1].item() <= max_conf]
        
        # Randomly sample 6000 samples from the filtered list if needed
        if len(filtered_confidences) > 10000:
            filtered_confidences = random.sample(filtered_confidences, 10000)
        
        # Create DataFrame with proper confidence values
        df = pd.DataFrame(
            [(img_path, pred, conf.item()) for (img_path, conf, pred) in filtered_confidences],
            columns=['image_path', 'predicted_label', 'confidence']
        )
        
        return df





    def train(self, max_iter=np.inf):
        loss_avg = utils.averager()
        prev_cer = 1000.0
        prev_wer = 1000.0
        write_info(self.model, self.opt)
        self.writer = Writer(self.opt.lr, self.opt.nepoch, self.opt.node_dir, use_tb=self.opt.use_tb)
        self.iterations = 0
        conf_thresh=0.99
        max_conf = 1.0
        min_conf = 0.8
        for epoch in range(self.opt.nepoch+1):

            if epoch%100 ==0:
                if self.opt.type=="semi":
                    accumulated_confident_samples = pd.DataFrame()
                    semi_images_dir = self.opt.source_imgs_dir 
                    label_txt_file = self.opt.source_labels_dir 
                    target_dir = self.opt.target_imgs_dir 
                    lmdb_dir = self.opt.lmdb_dir

                    
                    df = self.eval_confidence(image_dir=target_dir,
                                            target_dir=target_dir,
                                            max_conf=max_conf,
                                            min_conf=min_conf)

                    if accumulated_confident_samples.empty:
                        accumulated_confident_samples = df
                    else:
                        # Avoid duplicates by image_path
                        accumulated_confident_samples = pd.concat([accumulated_confident_samples, df]).drop_duplicates(subset=['image_path'])
                    
                    df.to_csv("target_images.csv")
                    conf_thresh *= 0.95 #math.floor(conf_thresh * 100) / 100.0
                    #removing old files
                    shutil.rmtree(lmdb_dir)
                    os.makedirs(lmdb_dir, exist_ok=True)
                    train_loader_updated = LMDBCreator(semi_images_dir, label_txt_file, accumulated_confident_samples, lmdb_dir)
                    train_loader_updated.create_dataset()
                    logging.warning(f"changing train root from {self.train_root} --> {lmdb_dir}")
                    self.train_root  =lmdb_dir
                    self.train_data, self.train_loader = self.get_data_loader(self.train_root,
                                                                      self.train_transforms,
                                                                      self.label_transform,
                                                                      )
            #May be: have to change mode : eval --> train
            self.writer.epoch = epoch
            self.writer.nbatches = len(self.train_loader)
            self.train_iter = iter(self.train_loader)
            logging.info(f"Training iteration is {len(self.train_loader)}")
            i = 0
            
            while i < len(self.train_loader):
                start_time1 = time.time()
                i+=1
                if i<23:
                    continue
                if (epoch) % self.opt.valInterval == 0:
                    valloss, val_CER, val_WER = self.eval(self.test_data, max_iter=self.val2_iter)
                    self.writer.update_valloss(valloss.val().item(), val_WER) 
                    if (epoch+1) %2==0:
                        torch.save(self.model.state_dict(), '{0}/{1}.pth'.format(self.opt.node_dir, str(epoch)))
                    if val_CER < prev_cer:
                        torch.save(self.model.state_dict(), '{0}/{1}.pth'.format(self.opt.node_dir,'best_cer'))
                        prev_cer = val_CER
                    if val_WER < prev_wer:
                        torch.save(self.model.state_dict(), '{0}/{1}.pth'.format(self.opt.node_dir,'best_wer'))
                        prev_wer = val_WER
                        self.writer.update_best_er(val_WER, self.iterations)
                print("Time taken for data_loader is>>>>>>>", {time.time()- start_time1})
                cost = self.trainBatch()
                loss_avg.add(cost)
                self.iterations += 1
                # i += 1
                self.writer.iterations = self.iterations
                self.writer.batch = i
                

                if self.iterations % self.opt.displayInterval == 0:
                    self.writer.update_trloss(loss_avg.val().item())
                    loss_avg.reset()
            #Itertive update
            # self.iterative_update()
                # if i==240:
                #     break
            
        self.writer.end()
        return

    def forward_sample(self, data):
        cpu_images, cpu_texts = data
        utils.loadData(self.image, cpu_images)
        t, l = self.converter.encode(cpu_texts)
        utils.loadData(self.text, t)
        utils.loadData(self.length, l)
        output_dict = self.model(self.image)
        batch_size = cpu_images.size(0)
        output_dict['batch_size'] = batch_size
        output_dict['gt'] = cpu_texts
        return output_dict

    def get_loss(self, data):
        preds = data['preds']
        batch_size = data['batch_size']
        preds_size = data['preds_size']
        torch.backends.cudnn.enabled = False
        cost = self.criterion(preds, self.text, preds_size, self.length) / batch_size
        torch.backends.cudnn.enabled = True
        return cost

    def decoder(self, preds, preds_size):
        if self.opt.beamdecoder:
            sim_preds = []
            for j in range(preds.size()[1]):
                probs = preds[:, j, :]
                probs = torch.cat([probs[:, 1:], probs[:, 0].unsqueeze(1)], dim=1).cpu().detach().numpy()
                sim_preds.append(ctc_bs.ctcBeamSearch(probs, self.test_data.voc, None))
        else:
            _, preds = preds.max(2)
            ####################################################################
            #need to write probability of each word
            preds1 = torch.transpose(preds, 0, 1)
            #print("preds ==>", preds)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            sim_preds = self.converter.decode(preds.data, preds_size.data, raw=False)
            #print("sim_preds==",sim_preds)
        return sim_preds, preds1

    def eval(self, data, max_iter=np.inf):
        data_loader = torch.utils.data.DataLoader(data, batch_size=self.opt.batchSize,
                                                num_workers=int(self.opt.workers),
                                                pin_memory=True,
                                                collate_fn=dataset.collatedict(),drop_last=False,
                                                persistent_workers=True)
        self.model.eval()
        gts = []
        decoded_preds = []
        val_preds = []
        max_probs = []
        val_iter = iter(data_loader)

        tc = 0
        wc = 0
        ww = 0
        tw = 0
        cw = 0
        
        loss_avg = utils.averager()
        #NOTE: change to 10 die to lack of time//CAREFUL
        max_iter = min(max_iter, len(data_loader))
        with torch.no_grad():
            for i in range(max_iter):
                if self.opt.mode == 'test':
                    print('%d / %d' % (i, len(data_loader)), end='\r')
                output_dict = self.forward_sample(next(val_iter)) #val_iter.next())
                batch_size = output_dict['batch_size']
                preds = F.log_softmax(output_dict['probs'], 2)
                ##########################################################
                # print("preds==>", preds.shape)# probability of each charcter
                preds1 = torch.transpose(preds, 0, 1)
                preds1 = torch.exp(preds1)
                preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
                cost = self.get_loss({'preds': preds, 'batch_size': batch_size,
                                      'preds_size': preds_size, 'params':output_dict['params']})
                loss_avg.add(cost)
                decoded_pred, preds2 = self.decoder(preds, preds_size)
                gts += list(output_dict['gt'])
                decoded_preds += list(decoded_pred)
                val_preds += list(preds1)
                max_probs += list(preds2) 

        if self.mode == "train":
            pcounter = 0
            for target, pred in zip(gts, decoded_preds):
                if pcounter < 20:
                    print('Gt:   ', target)
                    print('Pred: ', pred)
                    pcounter += 1
                if target!=pred:
                    ww += 1 

                tw += 1
                wc += utils.levenshtein(target, pred)
                tc += len(target)
            if (tw > 0):
                wer = (ww/float(tw))*100.0
            else:
                wer = 100.0
                
            if (tc > 0):
                cer = (wc/float(tc))*100.0
            else:
                cer = 100.0
                
            return loss_avg, cer, wer
        else:
            directory = self.opt.out
            dataset_name = self.opt.mode
            writepath = directory + '/' + dataset_name + "_accuracy" + ".txt"
            writepath1 = directory + '/' + dataset_name + "_gt_and_predicted_text" + ".txt" 
            mode = 'a' if os.path.exists(writepath) else 'w'
            f1 = open(writepath1, mode) 
            ajoy_index = 1
            for target, pred, val, prob_index in zip(gts, decoded_preds, val_preds, max_probs):         
                if target!=pred:
                    ww += 1
                else:
                    cw += 1
                tw += 1
                wc += utils.levenshtein(target, pred)
                tc += len(target)
                val1 = val.cpu().detach().numpy()
                prob_index1 = prob_index.cpu().detach().numpy()
                if (self.opt.store_probability=="yes"):
                    ajoy_path = directory + "/" + "data_prob/"    #probability of all characters in the character list 
                    ajoy_path1 = directory + "/" + "max_prob/"    #indix of character having max probability
                    if not os.path.exists(ajoy_path):
                        os.makedirs(ajoy_path)
                    if not os.path.exists(ajoy_path1):
                        os.makedirs(ajoy_path1)
                    ajoy_name = ajoy_path + str(ajoy_index) + ".csv"
                    ajoy_name1 = ajoy_path1 + str(ajoy_index) + ".csv"
                    np.savetxt(ajoy_name, val1, delimiter=",")
                    np.savetxt(ajoy_name1, prob_index1, delimiter=",")
                #########################################################
                # find for predicted word probability value
                sum1 = 0.0
                total = 0
                for ar1 in range(len(prob_index1)):
                    if (prob_index[ar1]!=0.0):
                        value = max(val1[ar1])
                        sum1 = sum1 + value
                        total = total + 1
                if (total>0):
                    word_prob = sum1/total
                else:
                    word_prob = sum1
                
                f1.write(str(target))
                f1.write("\t")
                f1.write(str(pred))
                f1.write("\t")
                f1.write(str(word_prob))
                f1.write("\n") 

                ajoy_index = ajoy_index + 1

            if (tw > 0):
                wrr = (cw/float(tw))*100.0
            else:
                wrr = 0.0

            if (tc > 0):
                crr = 100.0-((wc/float(tc))*100.0)
            else:
                crr = 0.0
            f1.close()
            print("FINAL RESULTS")
            print("Word Recognition Rate :",str(wrr))
            print("Character Recognition Rate :",str(crr))

            with open(writepath, mode) as f:
                f.write("Word Recognition Accuracy == ")
                f.write(str(wrr)) 
                f.write("\n")
                f.write("Character Recognition Accuracy == ")
                f.write(str(crr))
            f.close()    

            return

    def trainBatch(self):
        self.model.train()
        output_dict = self.forward_sample(next(self.train_iter))  
        batch_size = output_dict['batch_size']
        preds = F.log_softmax(output_dict['probs'], 2)
        preds_size = torch.LongTensor([preds.size(0)] * batch_size)
        cost = self.get_loss({
            'preds': preds, 
            'batch_size': batch_size, 
            'preds_size': preds_size, 
            'params': output_dict['params']
        })
        
        if torch.isnan(cost):
            pdb.set_trace()

        self.model.zero_grad()

        # Time measurement starts here
        start_time = time.time()
        
        cost.backward()
        # Optional: gradient clipping could go here if needed
        self.optimizer.step()

        if self.scheduler:
            self.scheduler.step()
        
        # Ensure all GPU operations have completed before stopping the timer
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        elapsed_time = time.time() - start_time
        print("Time taken for backpropagation and optimizer step: {:.4f} seconds".format(elapsed_time))
        
        return cost

if __name__ == "__main__":
    opt = parser.parse_args()
    obj = BaseHTR(opt)
    obj.run()
