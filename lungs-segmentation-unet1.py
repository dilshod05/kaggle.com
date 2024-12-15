import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models,layers,metrics
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import ModelCheckpoint

import cv2 as cv
from cv2 import imread,resize
from scipy.ndimage import label, find_objects

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt


import os
import warnings
warnings.filterwarnings('ignore')



img_path='/kaggle/input/chest-x-ray-lungs-segmentation/Chest-X-Ray/Chest-X-Ray/image'
mask_path='/kaggle/input/chest-x-ray-lungs-segmentation/Chest-X-Ray/Chest-X-Ray/mask'
metadata_path='/kaggle/input/chest-x-ray-lungs-segmentation/MetaData.csv'

data=pd.read_csv(metadata_path)
data.head()

data.info()

sns.countplot(data=data,x='ptb',hue='gender')

sns.countplot(data=data,x='ptb',hue='county')

data['gender'].value_counts()

def clean_gender(value):
    value = str(value).strip().lower()  
    if value in ['m', 'male', 'male,', 'male35yrs', 'male35yrs']:
        return 'male'
    elif value in ['f', 'female', 'female,', 'femal', 'female24yrs']:
        return 'female'
    else:
        return 'other'  


data['gender'] = data['gender'].apply(clean_gender)


print(data['gender'].value_counts())


sns.countplot(data=data,x='ptb',hue='gender')

data.describe(include='all')

data['age'] = pd.to_numeric(data['age'], errors='coerce')

plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x='gender', y='age', palette='coolwarm')
plt.title('Age Distribution by Gender')
plt.show()


plt.figure(figsize=(10, 6))
sns.histplot(data['age'], bins=20, kde=True, color='blue')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data['ptb'], bins=10, kde=True, color='green')
plt.title('PTB Score Distribution')
plt.xlabel('PTB')
plt.ylabel('Frequency')
plt.show()

sns.pairplot(data, vars=['age', 'ptb'], hue='gender', palette='husl')
plt.show()

data['remarks'] = data['remarks'].fillna('No Remarks')

plt.figure(figsize=(12, 6))
sns.boxplot(data=data, x='county', y='ptb', palette='Set3')
plt.title('PTB Distribution by County')
plt.xticks(rotation=45)
plt.show()

#os.listdir(os.path.join(img_path))

image_files=[file for file in os.listdir(img_path) if file.endswith('.png')]
image_to_visualize=image_files[:9]

plt.figure(figsize=(15,15))

for i,img in enumerate(image_to_visualize,1):
    img=cv.imread(os.path.join(img_path,img))
    plt.subplot(3,3,i)
    plt.imshow(img)
    plt.axis('off')

plt.tight_layout()

image_files=[file for file in os.listdir(img_path) if file.endswith('.png')]
image_to_visualize=image_files[:10]
mask_files=[file for file in os.listdir(mask_path) if file.endswith('.png')]

plt.figure(figsize=(6,30))
for i,img_file in enumerate(image_to_visualize):
    img = cv.imread(os.path.join(img_path, img_file))  
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  #
    
    mask = cv.imread(os.path.join(mask_path, img_file), cv.IMREAD_GRAYSCALE) 
    
    
    # Display the image
    plt.subplot(len(image_to_visualize), 2, i * 2 + 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Image: {img_file}")

    # Display the corresponding mask
    plt.subplot(len(image_to_visualize), 2, i * 2 + 2)
    plt.imshow(mask, cmap='gray')  # Use grayscale for masks
    plt.axis('off')
    plt.title(f"Mask: {img_file}")

plt.tight_layout()

#overlaying mask on images

image_files = [file for file in os.listdir(img_path) if file.endswith('.png')]


subset_files = image_files[:6]  

# Plot the images with overlays
plt.figure(figsize=(6, 15))

for i, img_file in enumerate(subset_files):
    # Load the image
    img = cv.imread(os.path.join(img_path, img_file))
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  
    mask = cv.imread(os.path.join(mask_path, img_file), cv.IMREAD_GRAYSCALE)
    mask_normalized = cv.normalize(mask, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
    mask_rgb = cv.applyColorMap(mask_normalized, cv.COLORMAP_JET)  # Apply colormap for better visualization
    overlay = cv.addWeighted(img, 0.7, mask_rgb, 0.3, 0)  # Adjust the weights as needed

    # Display 
    plt.subplot(len(subset_files), 2, i + 1)
    plt.imshow(overlay)
    plt.axis('off')
    plt.title(f"Overlay: {img_file}")

plt.tight_layout()
plt.show()

train_imgs,val_imgs,train_masks,val_masks=train_test_split(image_files,mask_files,test_size=0.2,random_state=42)
len(train_imgs),len(val_imgs)

class DataGenerator(Sequence):
     def __init__(self,img_list,mask_list,img_dir,mask_dir,batch_size=32,target_size=(128,128)):
         self.img_list=img_list
         self.mask_list=mask_list
         self.img_dir=img_dir
         self.mask_dir=mask_dir
         self.batch_size=batch_size
         self.target_size=target_size

     def __len__(self):
        return int(np.ceil(len(self.img_list)/self.batch_size))
     def __getitem__(self, index):
        start = index * self.batch_size
        end = (index + 1) * self.batch_size
        batch_imgs = self.img_list[start:end]
        batch_masks = self.mask_list[start:end]
        images=[]
        masks=[]

        for img_file, mask_file in zip(batch_imgs, batch_masks):
            img = cv.imread(os.path.join(self.img_dir, img_file))
            img = cv.resize(img, self.target_size)
            img = img / 255.0  # Normalize to [0, 1]
            images.append(img)
            
            mask = cv.imread(os.path.join(self.mask_dir, mask_file), cv.IMREAD_GRAYSCALE)
            mask = cv.resize(mask, self.target_size)
            mask = np.expand_dims(mask, axis=-1)  
            mask = mask / 255.0 
            mask=(mask>0.5).astype('float32')
            masks.append(mask)
        
        return np.array(images), np.array(masks)
    
    

batch_size = 32
target_size = (128, 128)

train_gen = DataGenerator(train_imgs, train_masks, img_path, mask_path, batch_size, target_size)
val_gen = DataGenerator(val_imgs, val_masks, img_path, mask_path, batch_size, target_size)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Conv2D,MaxPooling2D,Conv2DTranspose,concatenate

def unet_model(input_size=(128,128,3)):
    inputs=Input(input_size)

    #encoder
    conv1=Conv2D(64,(3,3),activation='relu',padding='same')(inputs)
    conv1=Conv2D(64,(3,3),activation='relu',padding='same')(conv1)
    pool1=MaxPooling2D(pool_size=(2,2))(conv1)

    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Bottleneck
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)

    # Decoder
    up4 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv3)
    merge4 = concatenate([up4, conv2])
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(merge4)
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4)

    up5 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv4)
    merge5 = concatenate([up5, conv1])
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(merge5)
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv5)

    # Output layer
    output = Conv2D(1, (1, 1), activation='sigmoid')(conv5)

    model = Model(inputs, output)
    return model

    


#defining the custom metrics
import tensorflow.keras.backend as K 

def dice_coefficient(y_true, y_pred):
    """Calculate the Dice Coefficient."""
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1e-7) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1e-7)

def iou(y_true, y_pred):
    """Calculate the Intersection over Union (IoU)."""
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + 1e-7) / (union + 1e-7)

model=unet_model()
model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy', dice_coefficient, iou])
model.summary()

# Train the model
validation_steps = len(val_gen) if len(val_gen) % batch_size == 0 else len(val_gen) // batch_size + 1
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20,
)


val_loss, val_accuracy, val_dice, val_iou = model.evaluate(val_gen)
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Validation Dice Coefficient: {val_dice:.4f}")
print(f"Validation IoU: {val_iou:.4f}")

for i in range(5):  # Visualize 5 samples
    img, true_mask = val_gen[i]
    pred_mask = model.predict(img)
    pred_mask = (pred_mask > 0.5).astype(np.float32)  # Apply thresholding

    # Calculate metrics for this sample
    dice = dice_coefficient(true_mask, pred_mask).numpy()
    intersection = np.sum(true_mask * pred_mask)
    union = np.sum(true_mask) + np.sum(pred_mask) - intersection
    iou_value = intersection / (union + 1e-7)

    # Plot the results
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(img[0])
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(true_mask[0].squeeze(), cmap='gray')
    plt.title("True Mask")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(pred_mask[0].squeeze(), cmap='gray')
    plt.title(f"Predicted Mask\nDice: {dice:.4f}, IoU: {iou_value:.4f}")
    plt.axis("off")

    plt.show()


