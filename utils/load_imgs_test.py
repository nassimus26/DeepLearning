import math
import pathlib
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from utils.PageSlider import PageSlider

'''
 * @author Nassim MOUALEK
 * @since 06/12/2020
'''
def loadPage(page, size):
    it = iter(train_ds)
    for i in range(page):
        images, labels = next(it)
    for i in range(len(images)):
        plt.subplot(sbatch_count, sbatch_count, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(train_ds.class_names[labels[i]], pad=10)

if __name__ == '__main__':
    print(tf.__version__)
    data_dir = pathlib.Path("C:/work/xray_dataset_covid19/train/")
    batch_size = 20
    img_height = 120
    img_width = 120
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        labels= 'inferred',
        image_size=(img_height, img_width),
        batch_size=batch_size)

    print(train_ds.class_names)
    print("ceil ", math.ceil(4.2))
    data = np.concatenate([i for x, i in train_ds], axis=0)
    size = len(data)
    rows = int(sqrt(size))
    batch_count = math.ceil(size/batch_size)
    print("ceil ", size, " " ,batch_count)
    sbatch_count = math.ceil(sqrt(batch_size))
    figs = plt.figure(figsize=(batch_size, batch_size))
    ax_depth = plt.axes([0.23, 0.02, 0.56, 0.04])
    slider_depth = PageSlider(ax_depth, 'pages', numpages=batch_count)
    # update the figure with a change on the slider
    loadPage(1, batch_size)
    def update_depth(page):
        loadPage(math.ceil(page), batch_size)
    slider_depth.on_changed(update_depth)
            #plt.axis("off")
    #imgs = list(data_dir.glob('*.jpeg'))
    #print(imgs)

    #for image in imgs:
    #    print(str(image))
    #    PIL.Image.open(str(image))
'''
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(36, activation='tanh'),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(12, activation='tanh'),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(features, targets, epochs=300, batch_size=50)

    features2, targets2 = ds.generateDataset(rows) #generate test data
    features2 = ds.extendDataset(features2)

    model.evaluate(features2, targets2, verbose=2)

    ds.show(features, targets)
'''
