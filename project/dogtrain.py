import matplotlib.pyplot as plt

import time
from datetime import timedelta

import scipy.misc
from scipy.stats import itemfreq
from random import sample
import pickle

import os
import numpy as np
import pandas as pd
import cv2
from glob import glob

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split

##archive_train = ZipFile("dog-breed-identification\\train.zip", 'r')
##archive_test = ZipFile("dog-breed-identification\\test.zip", 'r')

##archive_train.namelist()[0:5]

##len(archive_train.namelist()[:]) - 1

def build_model(size, num_classes):
    inputs = Input((size, size, 3))
    backbone = MobileNetV2(input_tensor=inputs, include_top=False, weights="imagenet")
    backbone.trainable = True
    x = backbone.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    x = Dense(1024, activation="relu")(x)
    x = Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, x)
    return model

def parse_data(x, y):
    x = x.decode()

    num_class = 120
    size = 224

    src = cv2.imread(x, cv2.IMREAD_COLOR)
    src = cv2.resize(src, (size, size))
    src = src / 255.0
    src = src.astype(np.float32)
    label = [0] * num_class
    label[y] = 1
    label = np.array(label)
    label = label.astype(np.int32)

    return src, label

def tf_parse(x, y):
    x, y = tf.numpy_function(parse_data, [x, y], [tf.float32, tf.int32])
    x.set_shape((224, 224, 3))
    y.set_shape((120))
    return x, y

def tf_dataset(x, y, batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch)
    dataset = dataset.repeat()
    return dataset


if __name__ == "__main__":
    #Cute way of not having to write the whole path
    path = "D:\\facultate\\IS\\dataset\\"
    train_path = os.path.join(path, "train\\*")
    test_path = os.path.join(path, "test\\*")
    labels_path = os.path.join(path, "labels.csv")

    labels_raw = pd.read_csv("D:\\facultate\\IS\\dataset\\labels.csv", compression=None, header=0, sep=',', quotechar='"')
    labels_raw.sample(5) #checking the labels

    #we save the dataset, using new label management
    #breed = labels_raw["breed"].unique()

    labels_df = pd.read_csv(labels_path)
    breed = labels_df["breed"].unique()
    print("Number of Breeds: ", len(breed))
    i = 0
    for b in breed:
        i=i+1
        print("Breed ",i,": ", b)

    breed2id = {name: i for i, name in enumerate(breed)}
    #id2breed = {name: i for i, name in enumarate(labels_df)} #unused

    ids = glob(train_path)

    labels = []
    
    for image_id in ids:
        image_id = image_id.split("\\")[-1].split(".")[0] #we get a certain ouput that we split to remain only with the image
        breed_name = list(labels_df[labels_df.id == image_id]["breed"])[0]
        breed_idx = breed2id[breed_name]
        print(image_id  )
        labels.append(breed_idx)

    ids = ids[:1000]
    labels = labels[:1000]

    # Spliting the dataset
    train_x, valid_x = train_test_split(ids, test_size=0.2, random_state=42)
    train_y, valid_y = train_test_split(labels, test_size=0.2, random_state=42)

    # Parameters for training the model
    # TODO: make it dynamic? (for size, num_classes)
    # TODO: random batch?
    size = 224
    num_classes = 120
    lr = 1e-4
    batch = 16
    epochs = 20


    # De aici in jos doamne ajuta, traiasca tensorflow
    # Model to be trained 
    model = build_model(size, num_classes)
    model.compile(loss="categorical_crossentropy", optimizer=Adam(lr), metrics=["acc"])
    model.summary()

    # Dataset
    train_dataset = tf_dataset(train_x, train_y, batch=batch)
    valid_dataset = tf_dataset(valid_x, valid_y, batch=batch)

    #Actual training
    callbacks = [
        ModelCheckpoint("model.h5", verbose=0, save_best_only=True),
        ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-6)
    ]
    train_steps = (len(train_x)//batch) + 1
    valid_steps = (len(valid_x)//batch) + 1
    history = model.fit(train_dataset,
        steps_per_epoch=train_steps,
        validation_steps=valid_steps,
        validation_data=valid_dataset,
        epochs=epochs,
        callbacks=callbacks) 
    #here training is done with the parameters chose by us.
    #We selected these parameteres based on another online project using this dataset

    
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()