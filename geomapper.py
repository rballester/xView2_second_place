"""
Script to combine all post-disaster .tifs into one large .tif (using GDAL) and visualize it as an image
"""

# import json
#
# with open('xview_geotransforms.json', 'r') as f:
#     data = json.load(f)
# print(data)
# # print(data.keys())
#
# with open('tmp.json', 'w') as outfile:
#     json.dump(data, outfile, indent=8)

# https://www.neonscience.org/resources/learning-hub/tutorials/merge-lidar-geotiff-py

import numpy as np
import matplotlib.pyplot as plt
import subprocess, glob
import os
import osgeo
import cv2
import imageio
import matplotlib.pyplot as plt
# from osgeo import gdal
# import gdal


# from osgeo import gdal
# ds = gdal.Open('/mnt/data/xview2/tiffs/geotiffs/all_images/hurricane-harvey_00000427_post_disaster.tif')
# print(ds.GetProjection())
# ds = gdal.Open('/mnt/data/xview2/results/hurricane-harvey/hurricane-harvey_00000427_post_damage.tif')
# print(ds.GetProjection())
# # image = imageio.imread('/mnt/data/xview2/results/hurricane-harvey/hurricane-harvey_00000427_post_damage.tif')
# # plt.figure()
# # plt.imshow(image)
# # plt.show()
# assert 0


# files_to_mosaic = [
#     'socal-fire_00000001_pre_disaster.tif',
#     'socal-fire_00000002_pre_disaster.tif'
# ]
# disaster = 'hurricane-harvey'
# disaster = 'socal-fire'
# disaster = 'midwest-flooding'
# disaster = 'lower-puna-volcano'
# disaster = 'woolsey-fire'
# disaster = 'hurricane-florence'
# disaster = 'nepal-flooding'
# disaster = 'tuscaloosa-tornado'
disaster = 'mexico-earthquake'
# disaster = 'hurricane-matthew'
# disaster = 'hurricane-michael'
# disaster = 'joplin-tornado'
# disaster = 'moore-tornado'
# disaster = 'palu-tsunami'
# disaster = 'pinery-bushfire'
# disaster = 'portugal-wildfire'
# disaster = 'santa-rosa-wildfire'
# disaster = 'sunda-tsunami'
# resolution = 0.00002
resolution = '0.00005'

# kind = 'pre_disaster'
# kind = 'post_disaster'
kind = 'post_damage'
# kind = 'damage_band'

# input_path = os.path.join('/mnt/data/xview2/all_tiffs/', disaster)
input_path = os.path.join('/mnt/data/xview2/results/', disaster)
output_tif_path = os.path.join('/mnt/data/xview2/results/', disaster, disaster+'_{}.tif'.format(kind))
output_jpg_path = os.path.join('/mnt/data/xview2/results/', disaster, disaster+'_{}.jpg'.format(kind))
# input_path = '/mnt/data/xview2/tiffs/geotiffs/all_images/'
# output_tif_path = os.path.join(os.path.join('/mnt/data/xview2/results/', disaster), disaster+'post.tif')
# output_jpg_path = os.path.join(os.path.join('/mnt/data/xview2/results/', disaster), disaster+'post.jpg')

# Create TIF via GDAL
if os.path.exists(output_tif_path):  # GDAL skips tif creation if it already exists
    os.remove(output_tif_path)
files_to_mosaic = sorted(glob.glob(os.path.join(input_path, '{}*_{}.tif'.format(disaster, kind))))#[:10]
files_string = " ".join(files_to_mosaic)
files_string = os.path.join(input_path, '{}_0*_{}.tif'.format(disaster, kind))
command = "/home/rballester/miniconda3/envs/xview2/bin/gdal_merge.py -o {} -ps {} {} -of gtiff ".format(output_tif_path, resolution, resolution) + files_string
print(command)
os.system(command)

# Convert TIF -> JPG
image = imageio.imread(output_tif_path)
image = np.array(image).astype(np.uint8)
imageio.imwrite(output_jpg_path, image)
os.system('geeqie {}'.format(output_jpg_path))
