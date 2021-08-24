#taken from this StackOverflow answer: https://stackoverflow.com/a/39225039
import requests
from os.path import join, isdir
import os
from argparse import ArgumentParser


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


parser = ArgumentParser()
parser.add_argument("--mode", type=str, default='pretrained-weight', help='pretrained-weight / cocostuff')
parser.add_argument("--dataset_dir", type=str, default='data', help='training dataset path')
args = parser.parse_args()

if args.mode == 'pretrained-weight':

    file_id = '1Xb-DKAA9ibCVLqm8teKd1MWk6imjwTBh'
    destination = 'checkpoints.zip'
    download_file_from_google_drive(file_id, destination)


elif args.mode == 'cocostuff':
    print('download cocostuff training dataset')
    url = "http://images.cocodataset.org/zips/train2017.zip"
    response = requests.get(url, stream = True)
    if isdir(join(args.dataset_dir, "cocostuff")) is False:
        os.makedirs(join(args.dataset_dir, "cocostuff"))
    save_response_content(response, join(args.dataset_dir, "cocostuff", "train.zip"))

elif args.mode == 'coco-weights':
    os.environ["FVCORE_CACHE"] = "checkpoints/fvcore_cache"

    from detectron2 import model_zoo
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor

    # Download coco weights
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
    cfg.MODEL.DEVICE = "cpu"
    predictor = DefaultPredictor(cfg)

else:
    print('Error Mode!')
