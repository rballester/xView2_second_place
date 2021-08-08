import argparse
import glob
import os
import re
import time
import imageio

from collections import namedtuple

import numpy as np
import torch
from albumentations.pytorch.transforms import img_to_tensor
from xView2_second_place import models
from xView2_second_place.tools.config import load_config
import matplotlib
from osgeo import gdal


class DamagePredictor():

    """
    Take pre/post disaster imagery and create a segmentation based on the degree of building damage.

    Code adapted from the second-place solution of the xView2 challenge (https://www.ibm.com/cloud/blog/the-xview2-ai-challenge), available here: https://github.com/DIUx-xView/xView2_second_place
    """

    def __init__(self):
        """
        Load and prepare model weights
        """

        ModelConfig = namedtuple("ModelConfig", "config_path weight_path type weight")
        config = ModelConfig("configs/d92_softmax.json", "softmax_dpn_seamese_unet_shared_dpn92_0_best_xview", "damage", 1)  # Note there are other models, and the best results are achieved by a running an ensemble of all models. However, this is almost as good and much faster
        conf = load_config(config.config_path)
        model = models.__dict__[conf['network']](seg_classes=5, backbone_arch=conf['encoder'])
        checkpoint_path = os.path.join(weight_path, config.weight_path)
        print("=> loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict({k[7:]: v for k, v in checkpoint['state_dict'].items()})
        model.eval()

        self.model = model.cuda()#.cpu()
        self.conf = load_config(config.config_path)

    def predict(self, pre: str, post: str):
        """
        Model inference for one pre/post image pair.

        :param pre: a path containing the image before the disaster
        :param post: a path containing the image after the disaster
        :return:
            - `predim`: a 5 x N x M ndarray, where N and M are the number of row and column pixels of the input images (1024 and 1024). The 5 channels at each pixel indicate the prediction: a (1, 0, 0, 0, 0) means no building detected, (0, 1, 0, 0, 0) means unharmed building, (0, 0, 1, 0, 0) means lightly damaged building, (0, 0, 0, 1, 0) means seriously damaged building, and (0, 0, 0, 0, 1) means building destroyed.
            - `overlay`: an N x M x 3 RGB image colorcoding the prediction (0 -> post disaster image, 1 -> white, 2 -> yellow, 3 -> orange, 4 -> red). It is useful for visualization purposes
        """

        assert isinstance(pre, str)
        assert isinstance(post, str)

        print("Predicting pair ({}, {})".format(os.path.basename(pre), os.path.basename(post)))

        image_pre = np.array(imageio.imread(pre)).astype(np.uint8)
        image_post = np.array(imageio.imread(post)).astype(np.uint8)
        image = np.concatenate([image_pre, image_post], axis=-1)

        with torch.no_grad():
            image = img_to_tensor(image, self.conf["input"]["normalize"]).cpu().numpy()
            image = torch.from_numpy(image).cpu().float()
            start = time.time()
            logits = self.model(image[None, ...].to('cuda'))
            pred = torch.softmax(logits, dim=1).cpu().numpy()
            pred = pred[0, ...]
            print('Inference time was:', time.time() - start)
            print()

        # Process output
        overlay = image_post.astype(np.float64)
        predim = pred.astype(np.float64)
        predim = predim / np.sum(predim, axis=0, keepdims=True)
        colors = [overlay / 255] + [
            np.array(matplotlib.colors.to_rgba(c)[:3])[None, None, :] * np.ones([overlay.shape[0], overlay.shape[1], 1]) for
            c in ['white', 'gold', 'darkorange', 'red']]
        colors = np.concatenate([c[None, ...] for c in colors], axis=0)
        overlay = np.einsum('rij,rijc->ijc', predim, colors)
        return predim, overlay


os.environ["MKL_NUM_THREADS"] = "16"
os.environ["NUMEXPR_NUM_THREADS"] = "16"
os.environ["OMP_NUM_THREADS"] = "16"
weight_path = "weights"

# Adjust as needed...
input_folder = '/mnt/data/xview2/all_tiffs'
output_folder = '/mnt/data/xview2/results'

# Select what disasters to run
disasters = [
    'hurricane-harvey',
    # 'socal-fire',
    # 'midwest-flooding',
    # 'lower-puna-volcano',
    # 'woolsey-fire',
    # 'hurricane-florence',
    # 'nepal-flooding',
    # 'tuscaloosa-tornado',
    # 'mexico-earthquake',
    # 'hurricane-matthew',
    # 'hurricane-michael',
    # 'joplin-tornado',
    # 'moore-tornado',
    # 'palu-tsunami',
    # 'pinery-bushfire',
    # 'portugal-wildfire',
    # 'santa-rosa-wildfire',
    # 'sunda-tsunami'
]

# Batch processing of all disaster pre/post pairs
for disaster in disasters:
    files = glob.glob(os.path.join(input_folder, disaster, '*'))
    disaster_files = dict()

    for file in files:
        m = re.match('{}_(\d+)_(pre|post)_disaster.*\.tif'.format(disaster), os.path.basename(file))
        if m is not None:
            number, when = m.groups()
            if number not in disaster_files:
                disaster_files[number] = dict()
            disaster_files[number][when] = file

    dp = DamagePredictor()
    for disaster in disasters:
        output_path = os.path.join(output_folder, disaster)
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        count = 1
        for number in sorted(disaster_files.keys()):

            # Inference
            predim, overlay = dp.predict(pre=disaster_files[number]['pre'], post=disaster_files[number]['post'])

            # Save tiff band (i.e. the 5 x N x M prediction) as an N x M .tif file. Each pixel contains a value (0, 1, 2, 3, or 4) containing the most likely class
            output_tif = os.path.join(output_path, '{}_{}_damage_band.tif'.format(disaster, number))
            if not os.path.exists(output_tif):
                predim2 = np.argmax(predim, axis=0)
                predim2 = (predim2).astype(np.uint8)
                ds = gdal.Open(disaster_files[number]['post'])
                driver = gdal.GetDriverByName("GTiff")
                outdata = driver.Create(output_tif, 1024, 1024, 1, gdal.GDT_UInt16)
                outdata.SetGeoTransform(ds.GetGeoTransform())  ##sets same geotransform as input
                outdata.SetProjection(ds.GetProjection())  ##sets same projection as input
                outdata.GetRasterBand(1).WriteArray(predim2)
                outdata.FlushCache()

            # Save combined overlay (i.e. the N x M x 3 RGB)
            output_tif = os.path.join(output_path, '{}_{}_post_damage.tif'.format(disaster, number))
            if not os.path.exists(output_tif):
                overlay2 = (overlay * 255).astype(np.uint8)
                ds = gdal.Open(disaster_files[number]['post'])
                driver = gdal.GetDriverByName("GTiff")
                outdata = driver.Create(output_tif, 1024, 1024, 3, gdal.GDT_UInt16)
                outdata.SetGeoTransform(ds.GetGeoTransform())  ##sets same geotransform as input
                outdata.SetProjection(ds.GetProjection())  ##sets same projection as input
                outdata.GetRasterBand(1).WriteArray(overlay2[..., 0])
                outdata.GetRasterBand(2).WriteArray(overlay2[..., 1])
                outdata.GetRasterBand(3).WriteArray(overlay2[..., 2])
                outdata.FlushCache()
                count += 1
