# Run features extraction from MRI 2D slices using Swin Transormer
# Author: Redha Ali, PhD
# Data Modified: 03/10/2023

import os
from scipy import io
import numpy as np
from scipy import ndimage
import tensorflow as tf
from swintransformer import SwinTransformer
import numpy, scipy.io
import matplotlib.pyplot as plt

r=0

# Read and load slices function
def read_mat_file(filepath):
    """Read and load slices"""
    # load .mat Slice
    scan = io.loadmat(filepath)
    # Get raw data
    scan = scan['out'].astype(np.float32)
    # load .npy Slice
    # scan = np.load(filepath)
    # scan = np.dstack((scan, scan, scan))
    return scan

# Resize function
def resize_volume(img,r):
    """Resize across x & y-axis"""
    # Set the desired depth
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
    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor))
    return img

# Read and resize slices
def process_scan(path, r):
    """Read and resize slices"""
    # Read scan
    volume = read_mat_file(path)
    # Resize width, height 
    volume = resize_volume(volume,r)
    return volume
# In[extract features function]

# loop over the dataset to get batches of images
def extract_features(dataset):

    features = []
    labels = []

    total_batches = tf.data.experimental.cardinality(dataset)
    current_batch = 1

    for images_batch, labels_batch in dataset:
        print("[INFO] processing batch {}/{}".format(current_batch, total_batches))
        # extract the features using the SWIN Model
        features_batch = model.predict(images_batch)

        # store the current batch of features in a list
        features.append(features_batch)
        labels.append(labels_batch)
        current_batch += 1

    features = np.vstack(features) 
    labels = np.hstack(labels)     
    return features, labels

# In[Model]

model = tf.keras.Sequential([
  tf.keras.layers.Lambda(lambda data: tf.keras.applications.imagenet_utils.preprocess_input(tf.cast(data, tf.float32), mode="torch"), input_shape=[224,224, 3]),
  SwinTransformer('swin_base_224', include_top=False, pretrained=True),
])

# In[Med Slice Features]


# Path for Med slices 
normal_train_paths =sorted( [
    os.path.join(os.getcwd(), "Dataset/Med/", x)
    for x in os.listdir("Dataset/Med/")
])

# Save the slices name to help compare them with their stiffness score (double check and it is optional)
image_name_Med=[]
for i, name in enumerate(normal_train_paths):
    basename = os.path.basename(normal_train_paths[i])
    image_name_Med.append(basename)

scipy.io.savemat('Swin features/image_name_Med%s_%d.mat'% ("Rot", r), mdict={'image_name_Med': image_name_Med})
print("Total number of slices : " + str(len(normal_train_paths)))


normal_scans_train= np.array([process_scan(path,r) for path in normal_train_paths])
normal_labels_train= np.array([1 for _ in range(len(normal_scans_train))])
x_train = normal_scans_train
y_train = normal_labels_train
train_loader = tf.data.Dataset.from_tensor_slices((x_train,y_train))



# Plot one slice for visualization
data = train_loader.take(1)
images, labels = list(data)[0]
images = images.numpy()
image = images[0]
print("Dimension of the slice scan is:", image.shape)
plt.imshow(np.squeeze(images[:, :,1]), cmap="gray")

batch_size = 64
train_dataset = (
    train_loader.batch(batch_size)
    .prefetch(2)
)

train_features, train_labels = extract_features(train_dataset)

import numpy, scipy.io
scipy.io.savemat('Swin features/swin_base_224_MAT_T2_fea_Med_train_fea_%s_%d.mat'% ("Rot", r), mdict={'train_features': train_features})

del normal_train_paths
del x_train
del normal_scans_train
del normal_labels_train
del train_loader
del train_features
del train_dataset
# In[Med-1 Slice Features]

normal_train_paths =sorted( [
    os.path.join(os.getcwd(), "Dataset/Med__1/", x)
    for x in os.listdir("Dataset/Med__1/")
])

image_name_Med__1=[]
for i, name in enumerate(normal_train_paths):
    basename = os.path.basename(normal_train_paths[i])
    image_name_Med__1.append(basename)
    
scipy.io.savemat('Swin features/image_name_Med__1%s_%d.mat'% ("Rot", r), mdict={'image_name_Med__1': image_name_Med__1})
print("Total number of slices : " + str(len(normal_train_paths)))

normal_scans_train= np.array([process_scan(path,r) for path in normal_train_paths])
normal_labels_train= np.array([1 for _ in range(len(normal_scans_train))])
x_train = normal_scans_train
y_train = normal_labels_train

train_loader = tf.data.Dataset.from_tensor_slices((x_train,y_train))
train_dataset = (
    train_loader.batch(batch_size)
    .prefetch(2)
)

train_features, train_labels = extract_features(train_dataset)


import numpy, scipy.io 
scipy.io.savemat('Swin features/swin_base_224_MAT_T2_fea_Med__1_train_fea_%s_%d.mat'% ("Rot", r), mdict={'train_features': train_features})


del normal_train_paths
del x_train
del normal_scans_train
del normal_labels_train
del train_loader
del train_features
del train_dataset
# In[Med+1 Slice Features]

normal_train_paths =sorted( [
    os.path.join(os.getcwd(), "Dataset/Med_1/", x)
    for x in os.listdir("Dataset/Med_1/")
])

image_name_Med_1=[]
for i, name in enumerate(normal_train_paths):
    basename = os.path.basename(normal_train_paths[i])
    image_name_Med_1.append(basename)


scipy.io.savemat('Swin features/image_name_Med_1%s_%d.mat'% ("Rot", r), mdict={'image_name_Med_1': image_name_Med_1})

print("Total number of slices : " + str(len(normal_train_paths)))

normal_scans_train= np.array([process_scan(path,r) for path in normal_train_paths])

normal_labels_train= np.array([1 for _ in range(len(normal_scans_train))])
x_train = normal_scans_train
y_train = normal_labels_train

train_loader = tf.data.Dataset.from_tensor_slices((x_train,y_train))

train_dataset = (
    train_loader.batch(batch_size)
    .prefetch(2)
)

train_features, train_labels = extract_features(train_dataset)



import numpy, scipy.io
scipy.io.savemat('Swin features/swin_base_224_MAT_T2_fea_Med_1_train_fea_%s_%d.mat'% ("Rot", r), mdict={'train_features': train_features})

del normal_train_paths
del x_train
del normal_scans_train
del normal_labels_train
del train_loader
del train_features
del train_dataset

# In[Med+2 Slice Features]

normal_train_paths =sorted( [
    os.path.join(os.getcwd(), "Dataset/Med_2/", x)
    for x in os.listdir("Dataset/Med_2/")
])

image_name_Med_2=[]
for i, name in enumerate(normal_train_paths):
    basename = os.path.basename(normal_train_paths[i])
    image_name_Med_2.append(basename)


scipy.io.savemat('Swin features/image_name_Med_2%s_%d.mat'% ("Rot", r), mdict={'image_name_Med_2': image_name_Med_2})

print("Total number of slices : " + str(len(normal_train_paths)))

normal_scans_train= np.array([process_scan(path,r) for path in normal_train_paths])

normal_labels_train= np.array([1 for _ in range(len(normal_scans_train))])
x_train = normal_scans_train
y_train = normal_labels_train

train_loader = tf.data.Dataset.from_tensor_slices((x_train,y_train))

train_dataset = (
    train_loader.batch(batch_size)
    .prefetch(2)
)

train_features, train_labels = extract_features(train_dataset)


import numpy, scipy.io
scipy.io.savemat('Swin features/swin_base_224_MAT_T2_fea_Med_2_train_fea_%s_%d.mat'% ("Rot", r), mdict={'train_features': train_features})

del normal_train_paths
del x_train
del normal_scans_train
del normal_labels_train
del train_loader
del train_features
del train_dataset

# In[Med-2 Slice Features]
normal_train_paths =sorted( [
    os.path.join(os.getcwd(), "Dataset/Med__2/", x)
    for x in os.listdir("Dataset/Med__2/")
])

image_name_Med__2 =[]
for i, name in enumerate(normal_train_paths):
    basename = os.path.basename(normal_train_paths[i])
    image_name_Med__2.append(basename)


scipy.io.savemat('Swin features/image_name_Med__2%s_%d.mat'% ("Rot", r), mdict={'image_name_Med__2': image_name_Med__2})

print("Total number of slices : " + str(len(normal_train_paths)))

normal_scans_train= np.array([process_scan(path,r) for path in normal_train_paths])

normal_labels_train= np.array([1 for _ in range(len(normal_scans_train))])
x_train = normal_scans_train
y_train = normal_labels_train

train_loader = tf.data.Dataset.from_tensor_slices((x_train,y_train))

train_dataset = (
    train_loader.batch(batch_size)
    .prefetch(2)
)

train_features, train_labels = extract_features(train_dataset)

import numpy, scipy.io
scipy.io.savemat('Swin features/swin_base_224_MAT_T2_fea_Med__2_train_fea_%s_%d.mat'% ("Rot", r), mdict={'train_features': train_features})

del normal_train_paths
del x_train
del normal_scans_train
del normal_labels_train
del train_loader
del train_features
del train_dataset
