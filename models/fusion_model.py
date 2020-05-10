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


class FusionModel(BaseModel):
    def name(self):
        return 'FusionModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.model_names = ['G', 'GF']

        # load/define networks
        num_in = opt.input_nc + opt.output_nc + 1
        
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

    def set_forward_without_box(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        self.full_real_A = input['A' if AtoB else 'B'].to(self.device)
        self.full_real_B = input['B' if AtoB else 'A'].to(self.device)
        # self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.full_hint_B = input['hint_B'].to(self.device)
        self.full_mask_B = input['mask_B'].to(self.device)
        self.full_mask_B_nc = self.full_mask_B + self.opt.mask_cent
        self.full_real_B_enc = util.encode_ab_ind(self.full_real_B[:, :, ::4, ::4], self.opt)

        (_, self.comp_B_reg) = self.netGComp(self.full_real_A, self.full_hint_B, self.full_mask_B)
        self.fake_B_reg = self.comp_B_reg

    def forward(self):
        (_, feature_map) = self.netG(self.real_A, self.hint_B, self.mask_B)
        self.fake_B_reg = self.netGF(self.full_real_A, self.full_hint_B, self.full_mask_B, feature_map, self.box_info_list)
        
    def save_current_imgs(self, path):
        out_img = torch.clamp(util.lab2rgb(torch.cat((self.full_real_A.type(torch.cuda.FloatTensor), self.fake_B_reg.type(torch.cuda.FloatTensor)), dim=1), self.opt), 0.0, 1.0)
        out_img = np.transpose(out_img.cpu().data.numpy()[0], (1, 2, 0))
        io.imsave(path, img_as_ubyte(out_img))

    def setup_to_test(self, fusion_weight_path):
        GF_path = 'checkpoints/{0}/latest_net_GF.pth'.format(fusion_weight_path)
        print('load Fusion model from %s' % GF_path)
        GF_state_dict = torch.load(GF_path)
        
        # G_path = 'checkpoints/coco_finetuned_mask_256/latest_net_G.pth' # fine tuned on cocostuff
        G_path = 'checkpoints/{0}/latest_net_G.pth'.format(fusion_weight_path)
        G_state_dict = torch.load(G_path)

        # GComp_path = 'checkpoints/siggraph_retrained/latest_net_G.pth' # original net
        # GComp_path = 'checkpoints/coco_finetuned_mask_256/latest_net_GComp.pth' # fine tuned on cocostuff
        GComp_path = 'checkpoints/{0}/latest_net_GComp.pth'.format(fusion_weight_path)
        GComp_state_dict = torch.load(GComp_path)

        self.netGF.load_state_dict(GF_state_dict, strict=False)
        self.netG.module.load_state_dict(G_state_dict, strict=False)
        self.netGComp.module.load_state_dict(GComp_state_dict, strict=False)
        self.netGF.eval()
        self.netG.eval()
        self.netGComp.eval()