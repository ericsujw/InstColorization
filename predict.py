import argparse
import multiprocessing
import os
import shutil
import tempfile
from glob import glob
from os import listdir
from os.path import isfile, join
from pathlib import Path

os.environ["FVCORE_CACHE"] = "checkpoints/fvcore_cache"

import cog
import cv2
import numpy as np
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger

from fusion_dataset import Fusion_Testing_Dataset
from models import create_model
from options.train_options import TestOptions
from util import util


class InstColorizationPredictor(cog.Predictor):
    def setup(self):
        self.has_gpu = torch.cuda.is_available()

        setup_logger()

        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")

        if self.has_gpu:
            self.device = torch.device("cuda")
        else:
            cfg.MODEL.DEVICE = "cpu"
            self.device = torch.device("cpu")

        self.predictor = DefaultPredictor(cfg)

        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        multiprocessing.set_start_method('spawn', True)

        torch.backends.cudnn.benchmark = True

        parser = argparse.ArgumentParser()
        TestOptions().initialize(parser)
        opt = parser.parse_args(["--name", "test_fusion", "--sample_p", "1.0", "--model", "fusion", "--fineSize", "256", "--test_img_dir", "inputs", "--results_img_dir", "results"])
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)


        if not self.has_gpu:
            opt.gpu_ids = []

        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])
        opt.A = 2 * opt.ab_max / opt.ab_quant + 1
        opt.B = opt.A
        opt.isTrain = False

        os.makedirs(opt.test_img_dir, exist_ok=True)

        self.save_img_dir = opt.results_img_dir
        if os.path.isdir(self.save_img_dir) is False:
            print('Create path: {0}'.format(self.save_img_dir))
            os.makedirs(self.save_img_dir)
        self.output_npz_dir = "{0}_bbox".format(opt.test_img_dir)
        if os.path.isdir(self.output_npz_dir) is False:
            os.makedirs(self.output_npz_dir)

        opt.batch_size = 1
        #dataset = Fusion_Testing_Dataset(opt)
        #dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=2)

        #dataset_size = len(dataset)
        #print('#Testing images = %d' % dataset_size)

        self.model = create_model(opt)
        # model.setup_to_test('coco_finetuned_mask_256')
        self.model.setup_to_test('coco_finetuned_mask_256_ffs', map_location=self.device)

        self.opt = opt

    @cog.input("input", type=Path, help="grayscale input image")
    def predict(self, input):
        output_dir = tempfile.mkdtemp()
        color_output_path = Path(os.path.join(output_dir, str(input).split(".")[0] + ".png"))

        try:
            input_dir = self.opt.test_img_dir
            input_path = os.path.join(input_dir, os.path.basename(input))
            shutil.copy(str(input), input_path)
            image_list = [f for f in listdir(input_dir) if isfile(join(input_dir, f))]

            self.get_bounding_boxes(input_dir, image_list)

            dataset = Fusion_Testing_Dataset(self.opt)
            dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0)

            count_empty = 0

            # Colorize
            with torch.no_grad():
                output_paths = []
                input_paths = []
                for data_raw in dataset_loader:
                    # if os.path.isfile(join(save_img_dir, data_raw['file_id'][0] + '.png')) is True:
                    #     continue
                    data_raw['full_img'][0] = data_raw['full_img'][0].to(self.device)
                    if data_raw['empty_box'][0] == 0:
                        data_raw['cropped_img'][0] = data_raw['cropped_img'][0].to(self.device)
                        box_info = data_raw['box_info'][0]
                        box_info_2x = data_raw['box_info_2x'][0]
                        box_info_4x = data_raw['box_info_4x'][0]
                        box_info_8x = data_raw['box_info_8x'][0]
                        cropped_data = util.get_colorization_data(data_raw['cropped_img'], self.opt, ab_thresh=0, p=self.opt.sample_p)
                        full_img_data = util.get_colorization_data(data_raw['full_img'], self.opt, ab_thresh=0, p=self.opt.sample_p)
                        self.model.set_input(cropped_data)
                        self.model.set_fusion_input(full_img_data, [box_info, box_info_2x, box_info_4x, box_info_8x])
                        self.model.forward()
                    else:
                        count_empty += 1
                        full_img_data = util.get_colorization_data(data_raw['full_img'], self.opt, ab_thresh=0, p=self.opt.sample_p)
                        self.model.set_forward_without_box(full_img_data)
                    output_path = join(self.save_img_dir, data_raw['file_id'][0] + '.png')
                    self.model.save_current_imgs(output_path, is_cuda=self.has_gpu)
                    output_paths.append(output_path)

                    input_path = glob(input_dir + "/" + data_raw["file_id"][0] + ".*")[0]
                    input_paths.append(input_path)

            # Resize

            if len(input_paths) > 1 or len(output_paths) > 1:
                print("WARNING: len(input_paths): {len(input_paths)}, len(output_paths): {len(output_paths)}")

            for input_path, output_path in zip(input_paths, output_paths):
                input_img = cv2.imread(input_path)
                height, width, _ = input_img.shape
                output_img = cv2.imread(output_path)
                output_img = cv2.resize(output_img, (width, height))
                input_hls = cv2.cvtColor(input_img, cv2.COLOR_BGR2HLS)
                output_hls = cv2.cvtColor(output_img, cv2.COLOR_BGR2HLS)
                output_hls[:, :, 1] = input_hls[:, :, 1]
                output_bgr = cv2.cvtColor(output_hls, cv2.COLOR_HLS2BGR)

                cv2.imwrite(str(color_output_path), output_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        finally:
            self.cleanup()

        return color_output_path

    def get_bounding_boxes(self, input_dir, image_list):
        for image_path in image_list:
            img = cv2.imread(join(input_dir, image_path))
            lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab_image)
            l_stack = np.stack([l_channel, l_channel, l_channel], axis=2)
            outputs = self.predictor(l_stack)
            save_path = join(self.output_npz_dir, image_path.split('.')[0])
            pred_bbox = outputs["instances"].pred_boxes.to(torch.device('cpu')).tensor.numpy()
            pred_scores = outputs["instances"].scores.cpu().data.numpy()
            np.savez(save_path, bbox = pred_bbox, scores = pred_scores)

    def cleanup(self):
        clean_folder(self.opt.test_img_dir)
        clean_folder(self.output_npz_dir)
        clean_folder(self.save_img_dir)

def clean_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
