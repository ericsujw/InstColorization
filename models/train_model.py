import os

import torch
from collections import OrderedDict
from util.image_pool import ImagePool
from util import util
from .base_model import BaseModel
from . import networks
import numpy as np
from skimage import io
from skimage import img_as_ubyte

import matplotlib.pyplot as plt
import math
from matplotlib import colors


class TrainModel(BaseModel):
    def name(self):
        return 'TrainModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.loss_names = ['G', 'L1']
        # load/define networks
        num_in = opt.input_nc + opt.output_nc + 1
        self.optimizers = []
        if opt.stage == 'full' or opt.stage == 'instance':
            self.model_names = ['G']
            self.netG = networks.define_G(num_in, opt.output_nc, opt.ngf,
                                        'siggraph', opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids,
                                        use_tanh=True, classification=opt.classification)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
        elif opt.stage == 'fusion':
            self.model_names = ['G', 'GF', 'GComp']
            self.netG = networks.define_G(num_in, opt.output_nc, opt.ngf,
                                        'instance', opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids,
                                        use_tanh=True, classification=False)
            self.netG.eval()
            
            self.netGF = networks.define_G(num_in, opt.output_nc, opt.ngf,
                                        'fusion', opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids,
                                        use_tanh=True, classification=False)
            self.netGF.eval()

            self.netGComp = networks.define_G(num_in, opt.output_nc, opt.ngf,
                                        'siggraph', opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids,
                                        use_tanh=True, classification=opt.classification)
            self.netGComp.eval()
            self.optimizer_G = torch.optim.Adam(list(self.netGF.module.weight_layer.parameters()) +
                                                list(self.netGF.module.weight_layer2.parameters()) +
                                                list(self.netGF.module.weight_layer3.parameters()) +
                                                list(self.netGF.module.weight_layer4.parameters()) +
                                                list(self.netGF.module.weight_layer5.parameters()) +
                                                list(self.netGF.module.weight_layer6.parameters()) +
                                                list(self.netGF.module.weight_layer7.parameters()) +
                                                list(self.netGF.module.weight_layer8_1.parameters()) +
                                                list(self.netGF.module.weight_layer8_2.parameters()) +
                                                list(self.netGF.module.weight_layer9_1.parameters()) +
                                                list(self.netGF.module.weight_layer9_2.parameters()) +
                                                list(self.netGF.module.weight_layer10_1.parameters()) +
                                                list(self.netGF.module.weight_layer10_2.parameters()) +
                                                list(self.netGF.module.model10.parameters()) +
                                                list(self.netGF.module.model_out.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
        else:
            print('Error Stage!')
            exit()
        self.criterionL1 = networks.HuberLoss(delta=1. / opt.ab_norm)
        # self.criterionL1 = networks.L1Loss()

        # initialize average loss values
        self.avg_losses = OrderedDict()
        self.avg_loss_alpha = opt.avg_loss_alpha
        self.error_cnt = 0
        for loss_name in self.loss_names:
            self.avg_losses[loss_name] = 0
        
    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.hint_B = input['hint_B'].to(self.device)
        
        self.mask_B = input['mask_B'].to(self.device)
        self.mask_B_nc = self.mask_B + self.opt.mask_cent

        self.real_B_enc = util.encode_ab_ind(self.real_B[:, :, ::4, ::4], self.opt)
    
    def set_fusion_input(self, input, box_info):
        AtoB = self.opt.which_direction == 'AtoB'
        self.full_real_A = input['A' if AtoB else 'B'].to(self.device)
        self.full_real_B = input['B' if AtoB else 'A'].to(self.device)

        self.full_hint_B = input['hint_B'].to(self.device)
        self.full_mask_B = input['mask_B'].to(self.device)

        self.full_mask_B_nc = self.full_mask_B + self.opt.mask_cent
        self.full_real_B_enc = util.encode_ab_ind(self.full_real_B[:, :, ::4, ::4], self.opt)
        self.box_info_list = box_info

    def forward(self):
        if self.opt.stage == 'full' or self.opt.stage == 'instance':
            (_, self.fake_B_reg) = self.netG(self.real_A, self.hint_B, self.mask_B)
        elif self.opt.stage == 'fusion':
            (_, self.comp_B_reg) = self.netGComp(self.full_real_A, self.full_hint_B, self.full_mask_B)
            (_, feature_map) = self.netG(self.real_A, self.hint_B, self.mask_B)
            self.fake_B_reg = self.netGF(self.full_real_A, self.full_hint_B, self.full_mask_B, feature_map, self.box_info_list)
        else:
            print('Error! Wrong stage selection!')
            exit()

    def optimize_parameters(self):
        self.forward()
        self.optimizer_G.zero_grad()
        if self.opt.stage == 'full' or self.opt.stage == 'instance':
            self.loss_L1 = torch.mean(self.criterionL1(self.fake_B_reg.type(torch.cuda.FloatTensor),
                                                        self.real_B.type(torch.cuda.FloatTensor)))
            self.loss_G = 10 * torch.mean(self.criterionL1(self.fake_B_reg.type(torch.cuda.FloatTensor),
                                                        self.real_B.type(torch.cuda.FloatTensor)))
        elif self.opt.stage == 'fusion':
            self.loss_L1 = torch.mean(self.criterionL1(self.fake_B_reg.type(torch.cuda.FloatTensor),
                                                        self.full_real_B.type(torch.cuda.FloatTensor)))
            self.loss_G = 10 * torch.mean(self.criterionL1(self.fake_B_reg.type(torch.cuda.FloatTensor),
                                                        self.full_real_B.type(torch.cuda.FloatTensor)))
        else:
            print('Error! Wrong stage selection!')
            exit()
        self.loss_G.backward()
        self.optimizer_G.step()

    def get_current_visuals(self):
        from collections import OrderedDict
        visual_ret = OrderedDict()
        if self.opt.stage == 'full' or self.opt.stage == 'instance':
            visual_ret['gray'] = util.lab2rgb(torch.cat((self.real_A.type(torch.cuda.FloatTensor), torch.zeros_like(self.real_B).type(torch.cuda.FloatTensor)), dim=1), self.opt)
            visual_ret['real'] = util.lab2rgb(torch.cat((self.real_A.type(torch.cuda.FloatTensor), self.real_B.type(torch.cuda.FloatTensor)), dim=1), self.opt)
            visual_ret['fake_reg'] = util.lab2rgb(torch.cat((self.real_A.type(torch.cuda.FloatTensor), self.fake_B_reg.type(torch.cuda.FloatTensor)), dim=1), self.opt)

            visual_ret['hint'] = util.lab2rgb(torch.cat((self.real_A.type(torch.cuda.FloatTensor), self.hint_B.type(torch.cuda.FloatTensor)), dim=1), self.opt)
            visual_ret['real_ab'] = util.lab2rgb(torch.cat((torch.zeros_like(self.real_A.type(torch.cuda.FloatTensor)), self.real_B.type(torch.cuda.FloatTensor)), dim=1), self.opt)
            visual_ret['fake_ab_reg'] = util.lab2rgb(torch.cat((torch.zeros_like(self.real_A.type(torch.cuda.FloatTensor)), self.fake_B_reg.type(torch.cuda.FloatTensor)), dim=1), self.opt)
            
        elif self.opt.stage == 'fusion':
            visual_ret['gray'] = util.lab2rgb(torch.cat((self.full_real_A.type(torch.cuda.FloatTensor), torch.zeros_like(self.full_real_B).type(torch.cuda.FloatTensor)), dim=1), self.opt)
            visual_ret['real'] = util.lab2rgb(torch.cat((self.full_real_A.type(torch.cuda.FloatTensor), self.full_real_B.type(torch.cuda.FloatTensor)), dim=1), self.opt)
            visual_ret['comp_reg'] = util.lab2rgb(torch.cat((self.full_real_A.type(torch.cuda.FloatTensor), self.comp_B_reg.type(torch.cuda.FloatTensor)), dim=1), self.opt)
            visual_ret['fake_reg'] = util.lab2rgb(torch.cat((self.full_real_A.type(torch.cuda.FloatTensor), self.fake_B_reg.type(torch.cuda.FloatTensor)), dim=1), self.opt)

            self.instance_mask = torch.nn.functional.interpolate(torch.zeros([1, 1, 176, 176]), size=visual_ret['gray'].shape[2:], mode='bilinear').type(torch.cuda.FloatTensor)
            visual_ret['box_mask'] = torch.cat((self.instance_mask, self.instance_mask, self.instance_mask), 1)
            visual_ret['real_ab'] = util.lab2rgb(torch.cat((torch.zeros_like(self.full_real_A.type(torch.cuda.FloatTensor)), self.full_real_B.type(torch.cuda.FloatTensor)), dim=1), self.opt)
            visual_ret['comp_ab_reg'] = util.lab2rgb(torch.cat((torch.zeros_like(self.full_real_A.type(torch.cuda.FloatTensor)), self.comp_B_reg.type(torch.cuda.FloatTensor)), dim=1), self.opt)
            visual_ret['fake_ab_reg'] = util.lab2rgb(torch.cat((torch.zeros_like(self.full_real_A.type(torch.cuda.FloatTensor)), self.fake_B_reg.type(torch.cuda.FloatTensor)), dim=1), self.opt)
        else:
            print('Error! Wrong stage selection!')
            exit()
        return visual_ret

    # return training losses/errors. train.py will print out these errors as debugging information
    def get_current_losses(self):
        self.error_cnt += 1
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                # float(...) works for both scalar tensor and float number
                self.avg_losses[name] = float(getattr(self, 'loss_' + name)) + self.avg_loss_alpha * self.avg_losses[name]
                errors_ret[name] = (1 - self.avg_loss_alpha) / (1 - self.avg_loss_alpha**self.error_cnt) * self.avg_losses[name]
        return errors_ret

    def save_fusion_epoch(self, epoch):
        path = '{0}/{1}_net_GF.pth'.format(os.path.join(self.opt.checkpoints_dir, self.opt.name), epoch)
        latest_path = '{0}/latest_net_GF.pth'.format(os.path.join(self.opt.checkpoints_dir, self.opt.name))
        torch.save(self.netGF.state_dict(), path)
        torch.save(self.netGF.state_dict(), latest_path)