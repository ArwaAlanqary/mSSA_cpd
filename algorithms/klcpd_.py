#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function
import argparse
import pickle
import math
import numpy as np
import os
import random
import sklearn.metrics
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

import klcpd.mmd_util as mmd_util
from klcpd.data_loader import DataLoader
from klcpd.optim import Optim
import utils

class NetG(nn.Module):
    def __init__(self, wnd_dim,RNN_hid_dim,  data):
        super(NetG, self).__init__()
        self.wnd_dim = wnd_dim
        self.var_dim = data.var_dim
        self.D = data.D
        self.RNN_hid_dim = RNN_hid_dim

        self.rnn_enc_layer = nn.GRU(self.var_dim, self.RNN_hid_dim, num_layers=1, batch_first=True)
        self.rnn_dec_layer = nn.GRU(self.var_dim, self.RNN_hid_dim, num_layers=1, batch_first=True)
        self.fc_layer = nn.Linear(self.RNN_hid_dim, self.var_dim)

    # X_p:   batch_size x wnd_dim x var_dim (Encoder input)
    # X_f:   batch_size x wnd_dim x var_dim (Decoder input)
    # h_t:   1 x batch_size x RNN_hid_dim
    # noise: 1 x batch_size x RNN_hid_dim
    def forward(self, X_p, X_f, noise):
        X_p_enc, h_t = self.rnn_enc_layer(X_p)
        X_f_shft = self.shft_right_one(X_f)
        hidden = h_t + noise
        Y_f, _ = self.rnn_dec_layer(X_f_shft, hidden)
        output = self.fc_layer(Y_f)
        return output

    def shft_right_one(self, X):
        X_shft = X.clone()
        X_shft[:, 0, :].data.fill_(0)
        X_shft[:, 1:, :] = X[:, :-1, :]
        return X_shft


class NetD(nn.Module):
    def __init__(self, wnd_dim, RNN_hid_dim,data):
        super(NetD, self).__init__()

        self.wnd_dim = wnd_dim
        self.var_dim = data.var_dim
        self.D = data.D
        self.RNN_hid_dim = RNN_hid_dim

        self.rnn_enc_layer = nn.GRU(self.var_dim, self.RNN_hid_dim, batch_first=True)
        self.rnn_dec_layer = nn.GRU(self.RNN_hid_dim, self.var_dim, batch_first=True)

    def forward(self, X):
        X_enc, _ = self.rnn_enc_layer(X)
        X_dec, _ = self.rnn_dec_layer(X_enc)
        return X_enc, X_dec




class KLCPD(nn.Module):
    def __init__(self,lambda_real = 0.1,CRITIC_ITERS=5, weight_clip =0.1,  lambda_ae = 0.001, lr = 3e-4, eval_freq = 50, grad_clip = 10., weight_decay =0., momentum =0,   RNN_hid_dim =10,max_iter =2000, optim = 'adam', batch_size =64, wnd_dim=25, sub_dim = 1,gpu =  0,trn_ratio = 1.0, val_ratio = 1.0, random_seed =1126, data_name = 'data_name', k = 3):
        super(KLCPD, self).__init__()

        self.trn_ratio = trn_ratio
        self.val_ratio = val_ratio
        self.gpu = gpu
        self.cuda = True
        self.random_seed = random_seed
        self.wnd_dim = wnd_dim
        self.sub_dim = sub_dim
        self.k = k 
        # RNN hyperparemters
        self.RNN_hid_dim = RNN_hid_dim 
        
        # optimization
        self.batch_size = batch_size 
        self.max_iter = max_iter 
        self.optim =optim
        self.lr = lr
        self.weight_decay = weight_decay 
        self.momentum = momentum 

        self.grad_clip = grad_clip 
        self.eval_freq = eval_freq
        
        # GAN
        self.CRITIC_ITERS =CRITIC_ITERS
        self.weight_clip = weight_clip
        self.lambda_ae = lambda_ae 
        self.lambda_real= lambda_real


        # save models
        self.save_path = 'klcpd/exp_simulate/' + '%s.wnd-%d.lambda_ae-%f.lambda_real-%f.clip-%f' % (data_name, wnd_dim, lambda_ae, lambda_real, weight_clip)
        
        
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        assert(os.path.isdir(self.save_path))
        # assert(args.sub_dim == 1)

        # #XXX For Yahoo dataset, trn_ratio=0.50, val_ratio=0.75
        # if 'yahoo' in args.data_path:
        #     args.trn_ratio = 0.50
        #     args.val_ratio = 0.75



        # ========= Setup GPU device and fix random seed=========#
        if torch.cuda.is_available():
            self.cuda = True
            torch.cuda.set_device(self.gpu)
            print('Using GPU device', torch.cuda.current_device())
        else:
            raise EnvironmentError("GPU device not available!")
        np.random.seed(seed=self.random_seed)
        random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed(self.random_seed)
        # [INFO] cudnn.benckmark=True enable cudnn auto-tuner to find the best algorithm to use for your hardware
        # [INFO] benchmark mode is good whenever input sizes of network do not vary much!!!
        # [INFO] https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
        # [INFO] https://discuss.pytorch.org/t/pytorch-performance/3079/2
        cudnn.benchmark == True

        # [INFO} For reproducibility and debugging, set cudnn.enabled=False
        # [INFO] Some operations are non-deterministic when cudnn.enabled=True
        # [INFO] https://discuss.pytorch.org/t/non-determinisic-results/459
        # [INFO] https://discuss.pytorch.org/t/non-reproducible-result-with-gpu/1831
        cudnn.enabled = True

    def train(self, ts):

        # ========= Load Dataset and initialize model=========#
        self.Data = DataLoader(data_path,self.wnd_dim, trn_ratio=self.trn_ratio, val_ratio= self.val_ratio)
        netG = NetG(self.wnd_dim, self.RNN_hid_dim, self.Data)
        netD = NetD(self.wnd_dim, self.RNN_hid_dim, self.Data)
        self.netG = netG
        self.netD = netD
        if self.cuda:
            self.netG.cuda()
            self.netD.cuda()
        netG_params_count = sum([p.nelement() for p in self.netG.parameters()])
        netD_params_count = sum([p.nelement() for p in self.netD.parameters()])
        print(self.netG)
        print(self.netD)
        print('netG has number of parameters: %d' % (netG_params_count))
        print('netD has number of parameters: %d' % (netD_params_count))
        one = torch.tensor(1,dtype=torch.float).to(torch.cuda.current_device())
        mone = one * -1


        # ========= Setup loss function and optimizer  =========#
        optimizerG = Optim(self.netG.parameters(),
                           self.optim,
                           lr=self.lr,
                           grad_clip=self.grad_clip,
                           weight_decay=self.weight_decay,
                           momentum=self.momentum)

        optimizerD = Optim(self.netD.parameters(),
                           self.optim,
                           lr=self.lr,
                           grad_clip=self.grad_clip,
                           weight_decay=self.weight_decay,
                           momentum=self.momentum)


        # sigma for mixture of RBF kernel in MMD
        #sigma_list = [1.0]
        #sigma_list = mmd_util.median_heuristic(Data.Y_subspace, beta=1.)
        sigma_list = mmd_util.median_heuristic(self.Data.Y_subspace, beta=.5)
        self.sigma_var = torch.FloatTensor(sigma_list).cuda()
        print('sigma_list:', self.sigma_var)


        # ========= Main loop for adversarial training kernel with negative samples X_f + noise =========#
        # Y_val = self.Data.val_set['Y'].numpy()
        # L_val = self.Data.val_set['L'].numpy()
        # Y_tst = self.Data.tst_set['Y'].numpy()
        # L_tst = self.Data.tst_set['L'].numpy()

        n_batchs = int(math.ceil(len(self.Data.trn_set['Y']) / float(self.batch_size)))
        print('n_batchs', n_batchs, 'batch_size', self.batch_size)

        lambda_ae = self.lambda_ae
        lambda_real = self.lambda_real
        gen_iterations = 0
        total_time = 0.
        best_epoch = -1
        best_val_mae = 1e+6
        best_val_auc = -1
        best_tst_auc = -1
        best_mmd_real = 1e+6
        start_time = time.time()
        print('start training: lambda_ae', lambda_ae, 'lambda_real', lambda_real, 'weight_clip', self.weight_clip)
        for epoch in range(1, self.max_iter + 1):
            trn_loader = self.Data.get_batches(self.Data.trn_set, batch_size=self.batch_size, shuffle=True)
            bidx = 0
            self.netD.train()
            while bidx < n_batchs:
                self.netD.train()
                ############################
                # (1) Update D network
                ############################
                for p in self.netD.parameters():
                    p.requires_grad = True

                for diters in range(self.CRITIC_ITERS):
                    # clamp parameters of NetD encoder to a cube
                    for p in self.netD.rnn_enc_layer.parameters():
                        p.data.clamp_(-self.weight_clip, self.weight_clip)
                    if bidx == n_batchs:
                        break

                    inputs = next(trn_loader)
                    X_p, X_f, Y_true = inputs[0], inputs[1], inputs[2]
                    batch_size = X_p.size(0)
                    bidx += 1

                    # real data
                    X_p_enc, X_p_dec = self.netD(X_p)
                    X_f_enc, X_f_dec = self.netD(X_f)

                    # fake data
                    noise = torch.cuda.FloatTensor(1, batch_size, self.RNN_hid_dim).normal_(0, 1)
                    noise = Variable(noise, requires_grad=True) # total freeze netG
                    Y_f = Variable(self.netG(X_p, X_f, noise).data)
                    Y_f_enc, Y_f_dec = self.netD(Y_f)

                    # batchwise MMD2 loss between X_f and Y_f
                    D_mmd2 = mmd_util.batch_mmd2_loss(X_f_enc, Y_f_enc, self.sigma_var)

                    # batchwise MMD loss between X_p and X_f
                    mmd2_real = mmd_util.batch_mmd2_loss(X_p_enc, X_f_enc, self.sigma_var)

                    # reconstruction loss
                    real_L2_loss = torch.mean((X_f - X_f_dec)**2)
                    #real_L2_loss = torch.mean((X_p - X_p_dec)**2)
                    fake_L2_loss = torch.mean((Y_f - Y_f_dec)**2)
                    #fake_L2_loss = torch.mean((Y_f - Y_f_dec)**2) * 0.0

                    # update netD
                    self.netD.zero_grad()
                    lossD = D_mmd2.mean() - lambda_ae * (real_L2_loss + fake_L2_loss) - lambda_real * mmd2_real.mean()
                    #lossD = 0.0 * D_mmd2.mean() - lambda_ae * (real_L2_loss + fake_L2_loss) - lambda_real * mmd2_real.mean()
                    #lossD = -real_L2_loss
                    lossD.backward(mone)
                    optimizerD.step()

                ############################
                # (2) Update G network
                ############################
                for p in self.netD.parameters():
                    p.requires_grad = False  # to avoid computation

                if bidx == n_batchs:
                    break

                inputs = next(trn_loader)
                X_p, X_f = inputs[0], inputs[1]
                batch_size = X_p.size(0)
                bidx += 1

                # real data
                X_f_enc, X_f_dec = self.netD(X_f)

                # fake data
                noise = torch.cuda.FloatTensor(1, batch_size, self.RNN_hid_dim).normal_(0, 1)
                noise = Variable(noise)
                Y_f = self.netG(X_p, X_f, noise)
                Y_f_enc, Y_f_dec = self.netD(Y_f)

                # batchwise MMD2 loss between X_f and Y_f
                G_mmd2 = mmd_util.batch_mmd2_loss(X_f_enc, Y_f_enc, self.sigma_var)
                print(G_mmd2.mean(), D_mmd2.mean().data.item())
                # update netG
                self.netG.zero_grad()
                lossG = G_mmd2.mean()
                #lossG = 0.0 * G_mmd2.mean()
                lossG.backward(one)
                optimizerG.step()

                #G_mmd2 = Variable(torch.FloatTensor(batch_size).zero_())
                gen_iterations += 1

                print('[%5d/%5d] [%5d/%5d] [%6d] D_mmd2 %.4e G_mmd2 %.4e mmd2_real %.4e real_L2 %.6f fake_L2 %.6f'
                      % (epoch, self.max_iter, bidx, n_batchs, gen_iterations,
                         D_mmd2.mean().data.item(), G_mmd2.mean().data.item(), mmd2_real.mean().item(),
                         real_L2_loss.data.item(), fake_L2_loss.data.item()))

                if gen_iterations % self.eval_freq == 0:
                    # ========= Main block for evaluate MMD(X_p_enc, X_f_enc) on RNN codespace  =========#
                    # val_dict = self.detect_evaluate( self.Data.val_set, Y_val, L_val)
                    # tst_dict = self.detect_evaluate( self.Data.tst_set, Y_tst, L_tst)
                    total_time = time.time() - start_time
                    # print('iter %4d tm %4.2fm val_mse %.1f val_mae %.1f val_auc %.6f'
                    #         % (epoch, total_time / 60.0, val_dict['mse'], val_dict['mae'], val_dict['auc']), end='')

                    # print (" tst_mse %.1f tst_mae %.1f tst_auc %.6f" % (tst_dict['mse'], tst_dict['mae'], tst_dict['auc']), end='')

                    assert(np.isnan(val_dict['auc']) != True)
                    #if val_dict['auc'] > best_val_auc:
                    #if val_dict['auc'] > best_val_auc and mmd2_real.mean().data[0] < best_mmd_real:
                    if mmd2_real.mean().data.item() < best_mmd_real:
                        # best_mmd_real = mmd2_real.mean().data.item()
                        # best_val_mae = val_dict['mae']
                        # best_val_auc = val_dict['auc']
                        # best_tst_auc = tst_dict['auc']
                        # best_epoch = epoch
                        # self.threshold = val_dict['threshold']
                        save_pred_name = '%s/pred.pkl' % (self.save_path)
                        with open(save_pred_name, 'wb') as f:
                            pickle.dump(tst_dict, f)
                        torch.save(self.netG.state_dict(), '%s/netG.pkl' % (self.save_path))
                        torch.save(self.netD.state_dict(), '%s/netD.pkl' % (self.save_path))
                    # print(" [best_val_auc %.6f best_tst_auc %.6f best_epoch %3d]" % (best_val_auc, best_tst_auc, best_epoch))

                # stopping condition
                #if best_mmd_real < 1e-4:
                # if mmd2_real.mean().data.item() < 1e-5:
                #     exit(0)
    
    def detect(self, ts):
        # Y, L should be numpy array

        data = DataLoader(ts, self.wnd_dim, trn_ratio=self.trn_ratio, val_ratio= self.val_ratio)
        

        self.netD.eval()
        loader = data
        Y_pred = []
        for inputs in loader.get_batches(self.Data.trn_set, self.batch_size, shuffle=False):
            X_p, X_f = inputs[0], inputs[1]
            self.batch_size = X_p.size(0)

            X_p_enc, _ = self.netD(X_p)
            X_f_enc, _ = self.netD(X_f)
            Y_pred_batch = mmd_util.batch_mmd2_loss(X_p_enc, X_f_enc, self.sigma_var)
            Y_pred.append(Y_pred_batch.data.cpu().numpy())
        Y_pred = np.concatenate(Y_pred, axis=0)

        L_pred = Y_pred
        self.score = L_pred
        # get the best threshold somehow and calculate cps ..
        threshold = np.mean(self.score) + self.k * np.std(threshold)
        
        binary = self.score > threshold
        ### return binaries based on threshold 

        self.cp = utils.convert_binary_to_intervals(binary)
        
    # def detect_evaluate(self, data, Y_true, L_true):
    #     # Y, L should be numpy array
    #     self.netD.eval()
    #     loader = self.Data
    #     Y_pred = []
    #     for inputs in loader.get_batches(data, self.batch_size, shuffle=False):
    #         X_p, X_f = inputs[0], inputs[1]
    #         self.batch_size = X_p.size(0)
    #         X_p_enc, _ = self.netD(X_p)
    #         X_f_enc, _ = self.netD(X_f)
    #         Y_pred_batch = mmd_util.batch_mmd2_loss(X_p_enc, X_f_enc, self.sigma_var)
    #         Y_pred.append(Y_pred_batch.data.cpu().numpy())
    #     Y_pred = np.concatenate(Y_pred, axis=0)

    #     L_pred = Y_pred
    #     fp_list, tp_list, thresholds = sklearn.metrics.roc_curve(L_true, L_pred)
    #     f1 = -1
    #     threshold = L_pred.mean()
    #     for thre in thresholds:
    #         f1_temp = sklearn.metrics.f1_score(L_true, L_pred>thre)
    #         if f1_temp>=f1:
    #             f1 = f1_temp
    #             threshold = thre
    #             print('f1:', f1)
    #     auc = sklearn.metrics.auc(fp_list, tp_list)
    #     eval_dict = {'Y_pred': Y_pred,
    #                  'L_pred': L_pred,
    #                  'Y_true': Y_true,
    #                  'L_true': L_true,
    #                  'mse': -1, 'mae': -1, 'auc': auc, 'threshold': threshold}
    #     return eval_dict

