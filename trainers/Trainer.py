import time
import os
import copy
import cv2
import shutil
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import random
import pandas as pd
import ttach as tta
import torchvision
import torchvision.transforms as transforms
from PIL import Image

from pathlib import Path
from tqdm import tqdm

from models import RESNET18
from sklearn.model_selection import StratifiedKFold
from Dataloader import CIFAR10_Dataset, WrapperDataset, Inference_Dataset
import sys
sys.path.append("../")

import utils.utils as utils
from utils.Logger import Logger
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from utils.utils import cal_acc_recal_pre_f1, Cutout
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from AutoAugment.autoaugment import CIFAR10Policy

class Trainer():
    def __init__(self, args, data_set, test_dataset):
        super(Trainer, self).__init__()
        self.args = args 
        self.epoch = 1 
        self.folds = args.fold
        self.kfold = StratifiedKFold(n_splits=self.folds, random_state=self.args.seed, shuffle=True)
        self.data_set = data_set
        self.test_dataset = test_dataset
        self.save_path = "./results/" + args.output_dir

        if os.path.exists("./results/") is False:
            os.makedirs("./results/")
        if os.path.exists(self.save_path) is False:
            os.makedirs(self.save_path)
        self.torch_device = f"cuda:{args.gpus}"

        self.logger = Logger(args, self.save_path)
        
        self.test_save_path = './results/' + args.output_dir + '/test'
        if os.path.exists(self.test_save_path) is False:
            os.makedirs(self.test_save_path)

        self.train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            CIFAR10Policy(),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            Cutout(n_holes=1, length=16),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.valid_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    def save(self, epoch, metric, filename="recent"):
        save_path = str(self.save_path)
        if os.path.exists(save_path) is False:
            os.mkdir(save_path)

        torch.save({
                    "epoch": epoch,
                    "param": self.model.state_dict(), #self.model.state_dict(),
                    "optimizer": self.optim.state_dict(),
                    "score": metric,
                    }, str(save_path + "/" + "{}_{}fold.pth.tar".format(filename, self.fold)))

        if metric > self.best_metric:
            print('{} -> {}'.format(self.best_metric, metric), end=', ')
            self.best_metric = metric
            shutil.copyfile(str(save_path + "/" + "{}_{}fold.pth.tar".format(filename, self.fold)), str(save_path + "/" + "best_{}fold.pth.tar".format(self.fold)))
            print("Model saved %d epoch" % (epoch))
    
    def load(self, fold, method="train", file_name="somemodel.pth.tar"):
        save_path = self.save_path
        file_name = "best_{}fold.pth.tar".format(fold)
        if os.path.exists(save_path + "/" + file_name): 
            print("Load %s File" % (save_path + "/" + file_name))
            ckpoint = torch.load(str(save_path + "/" + file_name), map_location=lambda storage, loc: storage)
            self.model.load_state_dict(ckpoint['param'])
            if method == "train":
              self.optim.load_state_dict(ckpoint['optimizer'])
              self.epoch = ckpoint['epoch']
              self.best_metric = ckpoint["score"]
              print("Load Complete, epoch : %d" % (self.epoch))
            else:
                epoch = ckpoint['epoch']
                best_metric = ckpoint["score"]
                print("Load Complete, epoch : %d, best_metric: %d" % (epoch, best_metric))
        else:
            print("Load Failed, not exists file")
    
    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def train(self):
        print("\nStart Train")
        start_time = time.time()
        count_label = list()
        
        for idx_dataset in range(len(self.data_set)):
            count_label.append(np.argmax(self.data_set[idx_dataset][1], axis=0))

        for folder, (train_idx, validate_idx) in enumerate(self.kfold.split(np.arange(len(self.data_set)), count_label)):
            self.fold = folder
            self.epoch = 1 
            self.model = RESNET18(out_channels=10)
            self.model = self.model.to(self.torch_device)

            self.criterion_ce = nn.CrossEntropyLoss().to(self.torch_device)
            self.optim = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=5e-4)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, mode='max', min_lr=0.000001, factor=0.8, patience=10)#patience=100)

            self.best_metric = 0
            self.load(folder) 
            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(validate_idx)
            
            self.train_loader = DataLoader(WrapperDataset(dataset=self.data_set, transform=self.train_transform), 
                                            batch_size=self.args.batch_size, sampler=train_sampler, num_workers=64)
            self.val_loader = DataLoader(WrapperDataset(dataset=self.data_set, transform=self.valid_transform),
                                            batch_size=self.args.batch_size, sampler=val_sampler, num_workers=64)

            for epoch in range(self.epoch, self.args.num_epoch+1):
                loader = tqdm(self.train_loader,
                              #total=len(self.data_loaders['train']),
                              desc="[Train fold {} epoch {}]".format(self.fold, epoch))

                losses = []
                acc_list = []
                for i, (input_image, target) in enumerate(loader):
                    self.model.train()

                    input_image, target = input_image.to(self.torch_device), target.to(self.torch_device)
                    
                    if self.args.cutmix == 1 and np.random.random() > 0.50: # cutmix 작동될 확률      
                        lam = np.random.beta(1.0, 1.0)
                        rand_index = torch.randperm(input_image.size()[0]).to(self.torch_device)
                        target_a = target
                        target_b = target[rand_index]            
                        bbx1, bby1, bbx2, bby2 = self.rand_bbox(input_image.size(), lam)
                        input_image[:, :, bbx1:bbx2, bby1:bby2] = input_image[rand_index, :, bbx1:bbx2, bby1:bby2]
                        im = Image.fromarray(np.uint8(input_image[0].permute(1, 2, 0).cpu().numpy() * 255))
                        im.save("sample_cutmix_image_{}.png".format(torch.argmax(target, dim=1)[0]))
                        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input_image.size()[-1] * input_image.size()[-2]))
                        output = self.model(input_image)
                        loss = self.criterion_ce(output, torch.argmax(target_a, axis=1)) * lam + self.criterion_ce(output, torch.argmax(target_b, axis=1)) * (1. - lam)
                    else:
                        im = Image.fromarray(np.uint8(input_image[0].permute(1, 2, 0).cpu().numpy() * 255))
                        im.save("sample_aug_image_{}_cutout.png".format(torch.argmax(target, dim=1)[0]))
                        output = self.model(input_image)
                        loss = self.criterion_ce(output, torch.argmax(target, axis=1))

                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()
                    
                    loss_np = loss.cpu().detach().numpy()
                    losses += [loss_np]
                    acc, recall, precision, F1_score = cal_acc_recal_pre_f1(torch.argmax(output, axis=1).to("cpu"), torch.argmax(target, axis=1).to("cpu"))
                    acc_list += [acc]
                    loader.set_postfix(loss='{:f} / {:f}'.format(loss_np, np.average(losses)))

                    del input_image, target, output, loss

                self.logger.will_write("[Train] epoch:%d loss:%f acc:%f" % (epoch, np.average(losses), np.average(acc_list)), is_print=False)
                val_acc = self.valid(epoch)
                self.scheduler.step(val_acc)

        print("\nEnd Train")
        total_time = time.time() - start_time
        print("\nTraining Time:", total_time, 'sec')
        self.logger.will_write("Total training time: %f sec" %total_time)

    def valid(self, epoch):
        self.model.eval()
        num_workers = self.args.batch_size if self.args.batch_size < 11 else 10
        
        with torch.no_grad():
            losses = []
            acc_list = []
            for i, (input_image, target) in enumerate(self.val_loader):
                input_image = input_image.to(self.torch_device)
                target = target.to(self.torch_device)
                output = self.model(input_image)

                loss = self.criterion_ce(output, torch.argmax(target, axis=1))
                
                loss_np = loss.cpu().detach().numpy()
                losses += [loss_np]
                acc, recall, precision, F1_score = cal_acc_recal_pre_f1(torch.argmax(output, axis=1).to("cpu"), torch.argmax(target, axis=1).to("cpu"))
                acc_list += [acc]
            metric = np.mean(acc_list)

            self.save(epoch, metric)
            self.logger.write("[Val] fold: %d epoch:%d loss:%f acc:%f" % (self.fold, epoch, np.mean(losses), np.mean(acc_list)))

            del input_image, output, target, loss
            return np.mean(acc_list)

    def mode(self, num_list):
        count = 0
        mode = 0
        for x in num_list: 
            if num_list.count(x) > count:
                count = num_list.count(x)
                mode = x
            elif num_list.count(x) == count:
                if x > mode:
                    count = num_list.count(x)
                    mode = x
        return mode

    def test(self):
        outputs_list = list()
        answers_list = list()
        num_workers = self.args.batch_size if self.args.batch_size < 11 else 10

        test_transforms = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        tta_transforms = tta.Compose([
                tta.HorizontalFlip(),
                tta.Rotate90(angles=[0, 30])      
        ])
        for folder in range(self.folds):
            self.criterion_ce = nn.CrossEntropyLoss().to(self.torch_device)
            self.model = RESNET18(out_channels=10).to(self.torch_device)
            self.load(folder, method="test", file_name='best.pth.tar')
            with torch.no_grad():
                losses = []
                acc_list = []
                output_list = []
                answer_list = []
                data_loader = DataLoader(WrapperDataset(dataset=self.test_dataset, transform=test_transforms), 
                                                batch_size=num_workers, shuffle=False)
                loader = tqdm(data_loader)
                for i, (input_image, target) in enumerate(loader):
                    self.model.eval()
                    input_image = input_image.to(self.torch_device)
                    target = target.to(self.torch_device)
  
                    #TTA
                    output_sum = 0
                    for transformer in tta_transforms:
                        augmented_image = transformer.augment_image(input_image)
                        with torch.no_grad():
                            output = self.model(augmented_image)
                            output_sum += output

                    output_list.append(output_sum.to("cpu"))

                    if folder == 0:
                      answer_list.append(target.to("cpu"))
                    loss = self.criterion_ce(output, torch.argmax(target, axis=1))
                    acc, recall, precision, F1_score = cal_acc_recal_pre_f1(torch.argmax(output, axis=1).to("cpu"), torch.argmax(target, axis=1).to("cpu"))
                    acc_list += [acc]
                    loss_np = loss.cpu().detach().numpy()
                    losses += [loss_np]
                metric = np.mean(losses)
                self.logger.write(("[Summary] loss:%f acc:%f" % (metric, np.mean(acc_list))))
                if folder == 0:
                  answers_list.append(torch.cat(answer_list, dim=0).numpy())
                outputs_list.append(torch.cat(output_list, dim=0).numpy())
        answers_list = np.array(answers_list)[0]
        outputs_list = np.array(outputs_list)
        print("soft voting")
        print(outputs_list.shape)
        mean_outputs_list = np.mean(outputs_list, axis=0)
        print(mean_outputs_list.shape)
        acc, recall, precision, F1_score = cal_acc_recal_pre_f1(np.argmax(mean_outputs_list, axis=1), np.argmax(answers_list, axis=1))
        print("acc:", acc)
        argmax_y = np.argmax(mean_outputs_list, axis=1)
        argmax_target = np.argmax(answers_list, axis=1)
        wrong_dict = dict()
        for idx in range(len(argmax_y)):
            if argmax_y[idx] != argmax_target[idx]:
                if argmax_target[idx] not in wrong_dict.keys():
                    wrong_dict[argmax_target[idx]] = dict()
                if argmax_y[idx] not in wrong_dict[argmax_target[idx]].keys():
                    wrong_dict[argmax_target[idx]][argmax_y[idx]] = 0
                wrong_dict[argmax_target[idx]][argmax_y[idx]] += 1
        for idx in wrong_dict.keys():
            print("answer: ", idx)
            print(wrong_dict[idx])
        print("hard voting")
        argmax_outputs_list = np.argmax(outputs_list, axis=2)
        hard_outputs_list = []
        for idx in range(argmax_outputs_list.shape[1]):
          final_ans = self.mode(argmax_outputs_list[:, idx].tolist())
          hard_outputs_list.append(final_ans)
        acc, recall, precision, F1_score = cal_acc_recal_pre_f1(np.array(hard_outputs_list), np.argmax(answers_list, axis=1))
        print("acc:", acc)

    def inference(self):
        label_dict = {
            "airplane" : 0,
            "automobile" : 1,
            "bird" : 2,
            "cat" : 3,
            "deer" : 4,
            "dog" : 5,
            "frog" : 6,
            "horse" : 7,
            "ship" : 8,
            "truck" : 9
        }
        label_encoder = {key:idx for idx, key in enumerate(label_dict)}
        label_decoder = {val:key for key, val in label_encoder.items()}
        pd_csv = pd.read_csv(self.args.csv_path)
        outputs_list = list()

        test_transforms = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        num_workers = self.args.batch_size if self.args.batch_size < 11 else 10

        tta_transforms = tta.Compose([
                tta.HorizontalFlip(),
                tta.Rotate90(angles=[0, 30])      
        ])

        for folder in range(self.folds):
            self.model = RESNET18(out_channels=10).to(self.torch_device)
            self.load(folder, method="test", file_name='best.pth.tar')
            inference_dataset = Inference_Dataset(self.args.test_path, self.args.csv_path, test_transforms)
            data_loader = DataLoader(dataset=inference_dataset, batch_size=num_workers)
            with torch.no_grad():
                losses = []
                acc_list = []
                output_list = []
                count_batch = 0
                image = None
                
                loader = tqdm(data_loader)
                for i, input_image in enumerate(loader):
                    input_image = input_image.to(self.torch_device)

                    self.model.eval()
                    output_sum = 0
                    for transformer in tta_transforms:
                        augmented_image = transformer.augment_image(input_image)
                        with torch.no_grad():
                            output = self.model(augmented_image)
                            output_sum += output

                    output_list.append(output_sum.to("cpu"))

                outputs_list.append(torch.cat(output_list, dim=0).numpy())
        outputs_list = np.array(outputs_list)
        print("soft voting")
        mean_outputs_list = np.mean(outputs_list, axis=0)
        argmax_outputs = np.argmax(mean_outputs_list, axis=1)
        for cnt in range(mean_outputs_list.shape[0]):
          pd_csv["target"][cnt] = label_decoder[argmax_outputs[cnt]]
        pd_csv.to_csv(os.path.join(self.save_path, "final_output.csv"), index=False)