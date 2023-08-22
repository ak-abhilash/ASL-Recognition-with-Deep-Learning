#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import random
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image

# Rest of the code remains unchanged

def load_data(container_path='C:/Users/91910/Desktop/dataset', folders=['A', 'B', 'C'],
              size=2000, test_split=0.2, seed=0):
    filenames, labels = [], []

    for label, folder in enumerate(folders):
        folder_path = os.path.join(container_path, folder)
        images = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path)]
        labels.extend(len(images) * [label])
        filenames.extend(images)
    
    random.seed(seed)
    data = list(zip(filenames, labels))
    random.shuffle(data)
    data = data[:size]
    filenames, labels = zip(*data)

    x = paths_to_tensor(filenames).astype('float32') / 255
    y = np.array(labels)

    x_train = np.array(x[:int(len(x) * (1 - test_split))])
    y_train = np.array(y[:int(len(x) * (1 - test_split))])
    x_test = np.array(x[int(len(x) * (1 - test_split)):])
    y_test = np.array(y[int(len(x) * (1 - test_split)):])

    return (x_train, y_train), (x_test, y_test)

def path_to_tensor(img_path, size):
    img = image.load_img(img_path, target_size=(size, size))
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths, size=50):
    list_of_tensors = [path_to_tensor(img_path, size) for img_path in img_paths]
    return np.vstack(list_of_tensors)

