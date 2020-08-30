import numpy as np
from PIL import Image
from skimage import color
import torch

def gen_gray_color_pil(color_img_path):
    '''
    return: RGB and GRAY pillow image object
    '''
    rgb_img = Image.open(color_img_path)
    if len(np.asarray(rgb_img).shape) == 2:
        rgb_img = np.stack([np.asarray(rgb_img), np.asarray(rgb_img), np.asarray(rgb_img)], 2)
        rgb_img = Image.fromarray(rgb_img)
    gray_img = np.round(color.rgb2gray(np.asarray(rgb_img)) * 255.0).astype(np.uint8)
    gray_img = np.stack([gray_img, gray_img, gray_img], -1)
    gray_img = Image.fromarray(gray_img)
    return rgb_img, gray_img

def read_to_pil(img_path):
    '''
    return: pillow image object HxWx3
    '''
    out_img = Image.open(img_path)
    if len(np.asarray(out_img).shape) == 2:
        out_img = np.stack([np.asarray(out_img), np.asarray(out_img), np.asarray(out_img)], 2)
        out_img = Image.fromarray(out_img)
    return out_img

def gen_maskrcnn_bbox_fromPred(pred_data_path, box_num_upbound=-1):
    '''
    ## Arguments:
    - pred_data_path: Detectron2 predict results
    - box_num_upbound: object bounding boxes number. Default: -1 means use all the instances.
    '''
    pred_data = np.load(pred_data_path)
    assert 'bbox' in pred_data
    assert 'scores' in pred_data
    pred_bbox = pred_data['bbox'].astype(np.int32)
    if box_num_upbound > 0 and pred_bbox.shape[0] > box_num_upbound:
        pred_scores = pred_data['scores']
        index_mask = np.argsort(pred_scores, axis=0)[pred_scores.shape[0] - box_num_upbound: pred_scores.shape[0]]
        pred_bbox = pred_bbox[index_mask]
    # pred_scores = pred_data['scores']
    # index_mask = pred_scores > 0.9
    # pred_bbox = pred_bbox[index_mask].astype(np.int32)
    return pred_bbox

def get_box_info(pred_bbox, original_shape, final_size):
    assert len(pred_bbox) == 4
    resize_startx = int(pred_bbox[0] / original_shape[0] * final_size)
    resize_starty = int(pred_bbox[1] / original_shape[1] * final_size)
    resize_endx = int(pred_bbox[2] / original_shape[0] * final_size)
    resize_endy = int(pred_bbox[3] / original_shape[1] * final_size)
    rh = resize_endx - resize_startx
    rw = resize_endy - resize_starty
    if rh < 1:
        if final_size - resize_endx > 1:
            resize_endx += 1
        else:
            resize_startx -= 1
        rh = 1
    if rw < 1:
        if final_size - resize_endy > 1:
            resize_endy += 1
        else:
            resize_starty -= 1
        rw = 1
    L_pad = resize_startx
    R_pad = final_size - resize_endx
    T_pad = resize_starty
    B_pad = final_size - resize_endy
    return [L_pad, R_pad, T_pad, B_pad, rh, rw]