from os import listdir
from os.path import isfile, join
from random import sample

import numpy as np
import torch
import torch.utils.data as Data
import torchvision.transforms as transforms

from image_util import *


class Fusion_Testing_Dataset(Data.Dataset):
    def __init__(self, opt, box_num=8):
        self.PRED_BBOX_DIR = '{0}_bbox'.format(opt.test_img_dir)
        self.IMAGE_DIR = opt.test_img_dir
        self.IMAGE_ID_LIST = [f for f in listdir(self.IMAGE_DIR) if isfile(join(self.IMAGE_DIR, f))]

        self.transforms = transforms.Compose([transforms.Resize((opt.fineSize, opt.fineSize), interpolation=2),
                                              transforms.ToTensor()])
        self.final_size = opt.fineSize
        self.box_num = box_num

    def __getitem__(self, index):
        pred_info_path = join(self.PRED_BBOX_DIR, self.IMAGE_ID_LIST[index].split('.')[0] + '.npz')
        output_image_path = join(self.IMAGE_DIR, self.IMAGE_ID_LIST[index])
        pred_bbox = gen_maskrcnn_bbox_fromPred(pred_info_path, self.box_num)

        img_list = []
        pil_img = read_to_pil(output_image_path)
        img_list.append(self.transforms(pil_img))
        
        cropped_img_list = []
        index_list = range(len(pred_bbox))
        box_info, box_info_2x, box_info_4x, box_info_8x = np.zeros((4, len(index_list), 6))
        for i in index_list:
            startx, starty, endx, endy = pred_bbox[i]
            box_info[i] = np.array(get_box_info(pred_bbox[i], pil_img.size, self.final_size))
            box_info_2x[i] = np.array(get_box_info(pred_bbox[i], pil_img.size, self.final_size // 2))
            box_info_4x[i] = np.array(get_box_info(pred_bbox[i], pil_img.size, self.final_size // 4))
            box_info_8x[i] = np.array(get_box_info(pred_bbox[i], pil_img.size, self.final_size // 8))
            cropped_img = self.transforms(pil_img.crop((startx, starty, endx, endy)))
            cropped_img_list.append(cropped_img)
        output = {}
        output['full_img'] = torch.stack(img_list)
        output['file_id'] = self.IMAGE_ID_LIST[index].split('.')[0]
        if len(pred_bbox) > 0:
            output['cropped_img'] = torch.stack(cropped_img_list)
            output['box_info'] = torch.from_numpy(box_info).type(torch.long)
            output['box_info_2x'] = torch.from_numpy(box_info_2x).type(torch.long)
            output['box_info_4x'] = torch.from_numpy(box_info_4x).type(torch.long)
            output['box_info_8x'] = torch.from_numpy(box_info_8x).type(torch.long)
            output['empty_box'] = False
        else:
            output['empty_box'] = True
        return output

    def __len__(self):
        return len(self.IMAGE_ID_LIST)


class Training_Full_Dataset(Data.Dataset):
    '''
    Training on COCOStuff dataset. [train2017.zip]
    
    Download the training set from https://github.com/nightrome/cocostuff
    '''
    def __init__(self, opt):
        self.IMAGE_DIR = opt.train_img_dir
        self.transforms = transforms.Compose([transforms.Resize((opt.fineSize, opt.fineSize), interpolation=2),
                                              transforms.ToTensor()])
        self.IMAGE_ID_LIST = [f for f in listdir(self.IMAGE_DIR) if isfile(join(self.IMAGE_DIR, f))]

    def __getitem__(self, index):
        output_image_path = join(self.IMAGE_DIR, self.IMAGE_ID_LIST[index])
        rgb_img, gray_img = gen_gray_color_pil(output_image_path)
        output = {}
        output['rgb_img'] = self.transforms(rgb_img)
        output['gray_img'] = self.transforms(gray_img)
        return output

    def __len__(self):
        return len(self.IMAGE_ID_LIST)


class Training_Instance_Dataset(Data.Dataset):
    '''
    Training on COCOStuff dataset. [train2017.zip]
    
    Download the training set from https://github.com/nightrome/cocostuff

    Make sure you've predicted all the images' bounding boxes using inference_bbox.py

    It would be better if you can filter out the images which don't have any box.
    '''
    def __init__(self, opt):
        self.PRED_BBOX_DIR = '{0}_bbox'.format(opt.train_img_dir)
        self.IMAGE_DIR = opt.train_img_dir
        self.IMAGE_ID_LIST = [f for f in listdir(self.IMAGE_DIR) if isfile(join(self.IMAGE_DIR, f))]
        self.transforms = transforms.Compose([
            transforms.Resize((opt.fineSize, opt.fineSize), interpolation=2),
            transforms.ToTensor()
        ])
    
    def __getitem__(self, index):
        pred_info_path = join(self.PRED_BBOX_DIR, self.IMAGE_ID_LIST[index].split('.')[0] + '.npz')
        output_image_path = join(self.IMAGE_DIR, self.IMAGE_ID_LIST[index])
        pred_bbox = gen_maskrcnn_bbox_fromPred(pred_info_path)

        rgb_img, gray_img = gen_gray_color_pil(output_image_path)

        index_list = range(len(pred_bbox))
        index_list = sample(index_list, 1)
        startx, starty, endx, endy = pred_bbox[index_list[0]]
        output = {}
        output['rgb_img'] = self.transforms(rgb_img.crop((startx, starty, endx, endy)))
        output['gray_img'] = self.transforms(gray_img.crop((startx, starty, endx, endy)))
        return output

    def __len__(self):
        return len(self.IMAGE_ID_LIST)


class Training_Fusion_Dataset(Data.Dataset):
    '''
    Training on COCOStuff dataset. [train2017.zip]
    
    Download the training set from https://github.com/nightrome/cocostuff

    Make sure you've predicted all the images' bounding boxes using inference_bbox.py

    It would be better if you can filter out the images which don't have any box.
    '''
    def __init__(self, opt, box_num=8):
        self.PRED_BBOX_DIR = '{0}_bbox'.format(opt.train_img_dir)
        self.IMAGE_DIR = opt.train_img_dir
        self.IMAGE_ID_LIST = [f for f in listdir(self.IMAGE_DIR) if isfile(join(self.IMAGE_DIR, f))]

        self.transforms = transforms.Compose([transforms.Resize((opt.fineSize, opt.fineSize), interpolation=2),
                                              transforms.ToTensor()])
        self.final_size = opt.fineSize
        self.box_num = box_num

    def __getitem__(self, index):
        pred_info_path = join(self.PRED_BBOX_DIR, self.IMAGE_ID_LIST[index].split('.')[0] + '.npz')
        output_image_path = join(self.IMAGE_DIR, self.IMAGE_ID_LIST[index])
        pred_bbox = gen_maskrcnn_bbox_fromPred(pred_info_path, self.box_num)

        full_rgb_list = []
        full_gray_list = []
        rgb_img, gray_image = gen_gray_color_pil(output_image_path)
        full_rgb_list.append(self.transforms(rgb_img))
        full_gray_list.append(self.transforms(gray_image))
        
        cropped_rgb_list = []
        cropped_gray_list = []
        index_list = range(len(pred_bbox))
        box_info, box_info_2x, box_info_4x, box_info_8x = np.zeros((4, len(index_list), 6))
        for i in range(len(index_list)):
            startx, starty, endx, endy = pred_bbox[i]
            box_info[i] = np.array(get_box_info(pred_bbox[i], rgb_img.size, self.final_size))
            box_info_2x[i] = np.array(get_box_info(pred_bbox[i], rgb_img.size, self.final_size // 2))
            box_info_4x[i] = np.array(get_box_info(pred_bbox[i], rgb_img.size, self.final_size // 4))
            box_info_8x[i] = np.array(get_box_info(pred_bbox[i], rgb_img.size, self.final_size // 8))
            cropped_rgb_list.append(self.transforms(rgb_img.crop((startx, starty, endx, endy))))
            cropped_gray_list.append(self.transforms(gray_image.crop((startx, starty, endx, endy))))
        output = {}
        output['cropped_rgb'] = torch.stack(cropped_rgb_list)
        output['cropped_gray'] = torch.stack(cropped_gray_list)
        output['full_rgb'] = torch.stack(full_rgb_list)
        output['full_gray'] = torch.stack(full_gray_list)
        output['box_info'] = torch.from_numpy(box_info).type(torch.long)
        output['box_info_2x'] = torch.from_numpy(box_info_2x).type(torch.long)
        output['box_info_4x'] = torch.from_numpy(box_info_4x).type(torch.long)
        output['box_info_8x'] = torch.from_numpy(box_info_8x).type(torch.long)
        output['file_id'] = self.IMAGE_ID_LIST[index]
        return output

    def __len__(self):
        return len(self.IMAGE_ID_LIST)