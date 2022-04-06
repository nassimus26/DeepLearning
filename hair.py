from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from os.path import exists
from pyunpack import Archive
import wget
import matplotlib.pyplot as plt
import numpy as np

hairFileName = "./hair.rar"
hairFolderName = "./hair"

#if exists(hairFileName) == False:
  #wget.download('https://uc78d62c37ee7a5df25266790bc1.dl.dropboxusercontent.com/cd/0/get/BIbPf8mglfgrrOIbppPrgfghuoutbieNT9tGerfpiDJHo2pvHSrI9jGK6qPbw8QwtafzKQyuWuR1pFTercvzvTvAy7GzU92XjOju3dsnfBsWCHC2skMEJYvBND6UPGU2Q04/file?_download_id=81578402605985324332729306446566510236790409392142015622153729515&_notify_domain=www.dropbox.com&dl=1')

#vgg16FileName = "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
#if exists(vgg16FileName) == False:
#  wget.download('https://github.com/fchollet/deep-learning-models/releases/download/v0.1/'+vgg16FileName)

#if exists(hairFolderName) == False:
#  Archive(hairFileName).extractall(".")

TRAIN_DIR = hairFolderName+'/train'
TEST_DIR = hairFolderName+'/test'

IMG_SIZE = (160, 160)
RANDOM_SEED = 123
batch_size = 30
epochs = 25
train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    brightness_range=[0, 0.6],
    horizontal_flip=True,
    vertical_flip=False,
    preprocessing_function=preprocess_input
)

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    color_mode='rgb',
    target_size=IMG_SIZE,
    batch_size=batch_size,
    class_mode='binary',
    seed=RANDOM_SEED
)

validation_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    color_mode='rgb',
    target_size=IMG_SIZE,
    batch_size=batch_size,
    class_mode='binary',
    seed=RANDOM_SEED
)

# load base model
vgg16_weight_path = './vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
base_model = VGG16(
    weights=vgg16_weight_path,
    include_top=False,
    input_shape=IMG_SIZE + (3,)
)

model = Sequential()
model.add(base_model)
model.add(layers.Flatten())
model.add(layers.Dense(4000, activation=tf.nn.tanh))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(256, activation=tf.nn.selu))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(64, activation=tf.nn.sigmoid))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(3))
model.layers[0].trainable = False

metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy', dtype=tf.float32)]
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(
    loss=loss,
    optimizer= RMSprop(lr=1e-3),
    metrics=metrics
)

model.summary()
if (False) :
    model = tf.keras.models.load_model('hair-pretrained.h5')
else :
    history = model.fit(
        train_generator,
        steps_per_epoch=30,
        epochs=epochs,
        validation_data=validation_generator
    )
    model.save('hair-pretrained.h5')
#Confusion matrix method
validation_generator.reset()
y = np.concatenate([validation_generator.next()[1] for i in range(validation_generator.__len__())])
prediction = model.predict(validation_generator)
result = tf.math.softmax(prediction)
y_pred = tf.argmax(input=result.numpy(), axis=1)
cm = confusion_matrix(y.ravel(), y_pred.numpy())
print("Confusion matrix : ")
print(cm)

test_datagen2 = ImageDataGenerator(

)

test_generator2 = test_datagen2.flow_from_directory(
    TEST_DIR,
    color_mode='rgb',
    target_size=IMG_SIZE,
    batch_size=batch_size,
    class_mode='binary',
    seed=RANDOM_SEED
)


def rdict(i):
    return dict(zip(i.values(), i.keys()))

imgs = np.concatenate([test_generator2.next()[0] for i in range(test_generator2.__len__())])
classes = rdict(train_generator.class_indices)
n=0
plt.figure(figsize=(30,30), dpi= 100)

for i in range(test_generator2.__len__()):
    expected = y[i]
    predicted = y_pred.numpy()[i]
    if (expected!=predicted):
        n = n + 1
        plt.subplot(6, 6, n)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(imgs[i].astype('uint8'))
        plt.xlabel(test_generator2.filenames[i] + "="+classes.get(expected)+" !"+ classes.get(predicted))
plt.show()