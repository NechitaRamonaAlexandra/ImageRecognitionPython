import os
import numpy as np
import pandas as pd
import cv2
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy

def read_image(path, size):
    src = cv2.imread(path, cv2.IMREAD_COLOR)
    src = cv2.resize(src, (size, size))
    src = src / 255.0
    src = src.astype(np.float32)
    return src


if __name__ == "__main__":
    path = "D:\\facultate\\IS\\dataset\\"
    train_path = os.path.join(path, "train\\*")
    test_path = os.path.join(path, "test\\*")
    labels_path = os.path.join(path, "labels.csv")

    labels_df = pd.read_csv(labels_path)
    breed = labels_df["breed"].unique()
    print("Total of Breeds: ", len(breed))

    breedToId = {name: i for i, name in enumerate(breed)}
    idToBreed = {i: name for i, name in enumerate(breed)}

    ids = glob(train_path)
    labels = []

    for image_id in ids:
        image_id = image_id.split("\\")[-1].split(".")[0]
        breed_name = list(labels_df[labels_df.id == image_id]["breed"])[0]
        breed_idx = breedToId[breed_name]
        labels.append(breed_idx)

    ids = ids[:1000]
    labels = labels[:1000]

    train_x, valid_x = train_test_split(ids, test_size=0.2, random_state=42)
    train_y, valid_y = train_test_split(labels, test_size=0.2, random_state=42)

    model = tf.keras.models.load_model("model.h5")

    for i, path in tqdm(enumerate(valid_x[:10])):
        image = read_image(path, 224)
        image = np.expand_dims(image, axis=0)
        pred = model.predict(image)[0]
        label_idx = np.argmax(pred)
        breed_name = idToBreed[label_idx]

        breed = idToBreed[valid_y[i]]
        currentImage = cv2.imread(path, cv2.IMREAD_COLOR)

        currentImage = cv2.putText(currentImage, breed_name, (0, 10), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 0.5, (255, 0, 0), 1)
        currentImage = cv2.putText(currentImage, breed, (0, 30), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 0.5, (0, 0, 0), 1)
        print(breed)

        cv2.imwrite(f"D:\\facultate\\IS\\dataset\\save\\valid_{i}.png", currentImage)

    