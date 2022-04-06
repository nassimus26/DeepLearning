#%%

import numpy as np
from sklearn.metrics import confusion_matrix
from tensorflow.keras import layers
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

TRAIN_DIR = './xray_dataset_covid19/train'
TEST_DIR = './xray_dataset_covid19/test'

IMG_SIZE = (160,160)
RANDOM_SEED = 123
batch_size = 12
epochs = 10
train_datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    brightness_range=[0.4, 1.2],
    horizontal_flip=False,
    vertical_flip=False,
    preprocessing_function=preprocess_input
)

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
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
model.add(layers.Dense(96, activation='tanh'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(16, activation='tanh'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(1, activation='sigmoid'))
model.layers[0].trainable = False

model.compile(
    loss='binary_crossentropy',
    optimizer=RMSprop(lr=1e-3),
    metrics=['accuracy']
)

model.summary()

#%%
history = model.fit(
    train_generator,
    steps_per_epoch=10,
    epochs=epochs,
    validation_data=validation_generator
)
#%%
#Confusion matrix method
validation_generator.reset()
y= np.concatenate([validation_generator.next()[1] for i in range(validation_generator.__len__())])
y_pred = np.where(model.predict(validation_generator)>0.5, 1, 0)
cm = confusion_matrix(y.ravel(), y_pred.ravel())
print("Confusion matrix : ")
print(cm)
#%%
#model.save('vgg_crossed_v2.h5')
#%%


