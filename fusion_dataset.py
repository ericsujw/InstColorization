from os import listdir
from os.path import isfile, join

import numpy as np
import torch
import torch.utils.data as Data
import torchvision.transforms as transforms

from options.train_options import TrainOptions
from image_util import *


class Fusion_Testing_Dataset(Data.Dataset):
    def __init__(self, opt):
        self.PRED_BBOX_DIR = 'example_bbox'
        self.IMAGE_DIR = 'example'
        self.IMAGE_ID_LIST = [f.split('.')[0] for f in listdir(self.IMAGE_DIR) if isfile(join(self.IMAGE_DIR, f))]

        self.transforms = transforms.Compose([transforms.Resize((opt.fineSize, opt.fineSize), interpolation=2),
                                              transforms.ToTensor()])
        self.final_size = opt.fineSize

    def __getitem__(self, index):
        pred_info_path = join(self.PRED_BBOX_DIR, self.IMAGE_ID_LIST[index] + '.npz')
        output_image_path = join(self.IMAGE_DIR, self.IMAGE_ID_LIST[index] + '.jpg')
        pred_bbox = gen_maskrcnn_bbox_fromPred(pred_info_path, 8)

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
        output['file_id'] = self.IMAGE_ID_LIST[index]
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