# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 08:24:46 2023

@author: Redha Ali
"""

import os
import numpy as np
from scipy import ndimage
import cv2
import matplotlib.pyplot as plt
import timm
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image
from pytorch_grad_cam import GradCAM
import pydicom
from swintransformer import SwinTransformer
import tensorflow as tf
from PIL import Image
from matplotlib.animation import FuncAnimation
from matplotlib import animation
# print("Numpy version:", np.__version__)
# print("Scipy version:", scipy.__version__)
# print("Pydicom version:", pydicom.__version__)
# print("TensorFlow version:", tf.__version__)

dir_path = "Y:\\CCHMC\\MRI1\\QC.T2\\PT-804-1022-804-1022\\SE-8-Ax_T2_FrFSE_RTr_FAT_SAT\\"

def inference(dir_path): 

    def list_dcm_files(folder):
        dcm_files = []
        for file_name in os.listdir(folder):
            if file_name.endswith(".dcm"):
                dcm_files.append(file_name)
        dcm_files = sorted(dcm_files,key=len)
        return dcm_files
     
    file_list = list_dcm_files(dir_path)
    first_file = pydicom.read_file(os.path.join(dir_path, file_list[0]))
    nx, ny = first_file.Columns, first_file.Rows
    nz = len(file_list)
    scaled_vol = np.zeros((nx, ny, nz))
    
    def range_d(data, a, b):
        min_value = min(data)
        max_value = max(data)
        rescaled_data = []
        for value in data:
            scaled_value = (value - min_value) * (b - a) / (max_value - min_value) + a
            rescaled_data.append(scaled_value)
        return rescaled_data
        
    def rescaled_dim(raw_vol):
        sx, sy, sz = raw_vol.shape
        volum_vec_nor = raw_vol.flatten()
        mu = 99.40
        st = 39.39
        volum_vec_nor = (volum_vec_nor - mu) / st
        a = -17
        b = 201
        scaled_vec = range_d(volum_vec_nor, a, b)
        scaled_vol = np.reshape(scaled_vec, (sx, sy, sz))
        return scaled_vol
    
    for i, file_name in enumerate(file_list):
        file_path = os.path.join(dir_path, file_name)
        dicom_file = pydicom.read_file(file_path)
        image_2d = dicom_file.pixel_array
        scaled_vol[:, :, i] = image_2d
    
    def resize_volume(img):
        # Set the desired depth
        img = np.dstack((img, img, img))
        desired_depth = 3
        desired_width = 224
        desired_height = 224
        # Get current depth
        current_depth = img.shape[-1]
        current_width = img.shape[0]
        current_height = img.shape[1]
        # Compute depth factor
        depth = current_depth / desired_depth
        width = current_width / desired_width
        height = current_height / desired_height
        depth_factor = 1 / depth
        width_factor = 1 / width
        height_factor = 1 / height
        img = ndimage.zoom(img, (width_factor, height_factor, depth_factor))
        return img
    
    def reshape_transform(tensor, height=14, width=14):
        result = tensor.reshape(tensor.size(0),
            height, width, tensor.size(2))
        result = result.transpose(2, 3).transpose(1, 2)
        return result
    
    fea_ex_model = tf.keras.Sequential([
      tf.keras.layers.Lambda(lambda data: tf.keras.applications.imagenet_utils.preprocess_input(tf.cast(data, tf.float32), mode="torch"), input_shape=[224,224, 3]),
      SwinTransformer('swin_base_224', include_top=False, pretrained=True),
    ])
    
    
    sx, sy, sz = scaled_vol.shape
    mid = sz // 2
    index = [ mid - 5, mid - 4, mid - 3, mid - 2, mid - 1, mid, mid + 1, mid + 2, mid + 3, mid + 4, mid + 6]
    scaled_vol_V1 = rescaled_dim(scaled_vol)
    features = []
    for i, ind in enumerate(index):
        volume_image = scaled_vol_V1[:, :, ind]
        volume_res = resize_volume(volume_image)
        image_batch = tf.expand_dims(volume_res, axis=0)
        features_batch = fea_ex_model.predict(image_batch)
        features.append(features_batch)
        
    features = np.concatenate(features, axis=1)   
    model = tf.keras.models.load_model('best_model\\best_model')
    y_pred = model.predict(features)
    if np.rint(y_pred) == 0:
        score = 1- y_pred
        hard_label = 'High Risk'
    elif np.rint(y_pred) == 1:
        score = y_pred
        hard_label = 'Low Risk'
        
    model_GradCam = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
    model_GradCam.eval()
    target_layers =[model_GradCam.layers[2].blocks[2].norm2]
    height, width, channels = volume_res.shape
    stacked_images = np.empty((height, width, channels, len(index)), dtype=np.uint8)
    stacked_cam_image = np.empty((height, width, channels, len(index)), dtype=np.uint8)
    stacked_both_images = np.empty((224, 449, channels, len(index)), dtype=np.uint8)
    for i, ind in enumerate(index):
        GradCam_slice = scaled_vol_V1[:, :, ind]
        GradCam_slice = resize_volume(GradCam_slice)
        GradCam_slice = GradCam_slice.astype(np.float32)
        GradCam_slice_tensor = preprocess_image(GradCam_slice)
        cam = GradCAM(model=model_GradCam, target_layers=target_layers, reshape_transform=reshape_transform)
        grayscale_cam = cam(input_tensor=GradCam_slice_tensor,
                          targets=None,
                          eigen_smooth='--eigen_smooth',
                          aug_smooth='--aug_smooth')
        Norm_slice = cv2.normalize(GradCam_slice, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        grayscale_cam = grayscale_cam[0, :]
        cam_image = show_cam_on_image(Norm_slice, grayscale_cam, use_rgb=True, colormap = cv2.COLORMAP_JET, image_weight = 0.5)
        cam = np.uint8(255*grayscale_cam[0, :])
        cam = cv2.merge([cam, cam, cam])
        images = np.hstack((np.uint8(255*Norm_slice), cam , cam_image))
        Image.fromarray(images)
        stacked_cam_image[:, :, :, i] = cam_image
        stacked_images[:, :, :, i] = Norm_slice
        stacked_both_images[:, :, :, i] = images
    return hard_label, score, stacked_cam_image, stacked_images, stacked_both_images

hard_label, score, stacked_cam_image, stacked_images, stacked_both_images = inference(dir_path)

print(hard_label) 
print(score) 

fig, ax = plt.subplots()
confidence_score = []
# Function to update the plot for each frame
def animate(frame):
    ax.imshow(stacked_both_images[:, :, :, frame])
    ax.axis('off')
    score1 = score[0]
    score2 = score1[0]
    formatted_accuracy = '{:.5%}'.format(score2)
    title = f'Classified as {hard_label} with confidence score ={formatted_accuracy} \n Slice# {frame+1}'
    ax.set_title(title)
    
# Create the animation
anim  = FuncAnimation(fig, animate, frames=stacked_both_images.shape[3], interval=1000)
 
# Save the animation as a video file
anim.save('output/video_11.gif')


