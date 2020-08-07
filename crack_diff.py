# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K


# %%
K.set_learning_phase(0)  ##

from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt


img_width, img_height = 150, 150
batch_size = 16

train_path = r"C:\Users\L7927301\Image_Analytics\Crack Classification\Train"


train_datagen = ImageDataGenerator(validation_split=0.2,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(img_height, img_width),
    shuffle=True, seed=13,
    batch_size=batch_size,
    class_mode='categorical',
    subset="training"
    )

validation_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(img_height, img_width),
    shuffle=True, seed=13,
    batch_size=batch_size,
    class_mode='categorical',
    subset="validation")


# %%
samples_per_epoch = 1000
validation_steps = 300
nb_filters1 = 32
nb_filters2 = 64
nb_filters3 = 128
conv1_size = 3
conv2_size = 2
pool_size = 2
classes_num = 2
lr = 1e-3
epochs=1


# %%
import keras
from keras import applications
from keras.models import Model, load_model
from keras.layers import Input, Flatten, Dense
from keras.layers.core import Dropout, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D, SeparableConv2D
from keras.callbacks import EarlyStopping,  ReduceLROnPlateau
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions


# %%
#Get back the convolutional part of a VGG network trained on ImageNet
vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# Freeze the layers except the last 4 layers
for layer in vgg_conv.layers[:-4]:
    layer.trainable = False
 


# %%
# Create the model
model = Sequential()

# Add the vgg convolutional base model
model.add(vgg_conv)

# Add new layers
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(classes_num, activation='softmax'))

# Show a summary of the model. Check the number of trainable parameters
model.summary()


# %%
lr=0.0001

model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.RMSprop(lr=lr),
              metrics=['accuracy'])


# %%
model.fit_generator(
    train_generator,
    samples_per_epoch=samples_per_epoch,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=[
        EarlyStopping(patience=10, restore_best_weights=True),
        ReduceLROnPlateau(patience=10)])

# Save the model
model.save('crack_diff.h5')


# %%
sess = K.get_session()
print(model.input, model.outputs)


# %%
import cv2 as cv
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

MODEL_PATH = 'crack_out'
MODEL_NAME = 'file'
input_node_name = 'vgg16_input_6'
output_node_name = 'dense_14/Softmax' 
get_ipython().system('rm -rf {MODEL_PATH}/')

tf.train.write_graph(sess.graph_def, MODEL_PATH, f'{MODEL_NAME}_graph.pb', as_text=False)
tf.train.write_graph(sess.graph_def, MODEL_PATH, f'{MODEL_NAME}_graph.pbtxt')
tf.train.Saver().save(sess, f'{MODEL_PATH}/{MODEL_NAME}.chkp')

freeze_graph.freeze_graph(f'{MODEL_PATH}/{MODEL_NAME}_graph.pbtxt',
                          None, False,
                          f'{MODEL_PATH}/{MODEL_NAME}.chkp',
                          output_node_name,
                          "save/restore_all",
                          "save/Const:0",
                          f'{MODEL_PATH}/frozen_{MODEL_NAME}.pb',
                          True, "")


# %%
cvNet = cv.dnn.readNetFromTensorflow(f'{MODEL_PATH}/frozen_{MODEL_NAME}.pb')


# %%
import cv2
import glob

image_size=150
num_channels=3
images=[]
def process(filename, key):
    image = cv2.imread(filename)
    image = cv2.resize(image, (image_size, image_size), cv2.INTER_LINEAR)
    image = np.array(image).reshape([image_size, image_size, num_channels]) 
    print(image.shape)
    images.append(image)


for (i,image_file) in enumerate(glob.iglob('C:/Users/L7927301/Image_Analytics/Crack Classification/Test/*.jpg')):
        process(image_file, i)

images = np.array(images, dtype=np.uint8)
images = images.astype('float32')
x_batch = np.multiply(images, 1.0/255.0)


# %%
frozen_graph=f'{MODEL_PATH}/frozen_{MODEL_NAME}.pb'
with tf.gfile.GFile(frozen_graph, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())


with tf.Graph().as_default() as graph:
      tf.import_graph_def(graph_def,
                          input_map=None,
                          return_elements=None,
                          name=""
      )
## NOW the complete graph with values has been restored
y_pred = graph.get_tensor_by_name('dense_14/Softmax:0')
## Let's feed the images to the input placeholders
x= graph.get_tensor_by_name("vgg16_input_6:0")
y_test_images = np.zeros((1, 2))
sess= tf.Session(graph=graph)
### Creating the feed_dict that is required to be fed to calculate y_pred 
feed_dict_testing = {x: x_batch}
result=sess.run(y_pred, feed_dict=feed_dict_testing)
print(result)


# %%
import pandas as pd
import numpy as np

result = pd.DataFrame(result)
# get the column name of max values in every row
max_prob = result.idxmax(axis=1)
df=max_prob.to_frame(name="Class")


# %%
a_dict = {}  # To store the values in
with open('C:/Users/L7927301/Image_Analytics/Crack Classification/crack_out/labels.txt', 'r') as input_file:
    for line in input_file:
        entry = line.split(":")  # split for key, value
        # store into dict, need to strip ' from the key and \n from value
        a_dict[entry[0].strip("' ")] = entry[1].strip()

print(a_dict)


# %%
df = df.applymap(str)
df.Class = [a_dict[item] for item in df.Class] 
print(df) 


# %%



