# %%
# # Defines and logger

import os
import sys
import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
from tensorflow import keras
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import logging


# create logger
logger = logging.getLogger('createCNN')
logging.basicConfig(filename='createCNN.log', filemode='w', encoding='utf-8', 
                    level=logging.DEBUG, 
                    format='[%(asctime)s][%(name)s][%(levelname)s]%(message)s')

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter('[%(asctime)s][%(levelname)s]%(message)s')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)

# 'application' code
# logger.debug('debug message')
# logger.info('info message')
# logger.warning('warn message')
# logger.error('error message')
# logger.critical('critical message')

def log_info(type : int, *message : str):
    assert type >= 0, f"type !>= 0: {type=}"
    message_types = ('Step', 'Status', 'Info')
    logger.info(f"[{message_types[type]}]: {' '.join(message)}")

log_info(2, "Tensoflow version:", tf.version.VERSION)

# %% 
# # Either load or create model

log_info(0, "Settings for loading or creating model")

LOAD_EXISTING_MODEL = True
FIT_MODEL = False
EPOCHS = 10
SAVE_MODEL = False

log_info(1, f"Set to: {LOAD_EXISTING_MODEL=}, {FIT_MODEL=}, {SAVE_MODEL=}")

# %% 
# # Defining and loading the dataset
log_info(0, "Defining and loading the dataset")

log_info(1, "Finding images")
args = sys.argv[1:]
USE_DEFAULT_IMAGEPATH = True
if USE_DEFAULT_IMAGEPATH:
    dataset_dir = "../get_images/imgs/"
else:
    if len(args) == 2 and args[0] == '-image_path':
        dataset_dir = str(args[1])	
    else:
        dataset_dir = input("Write path to images:")

data_dir = pathlib.Path(dataset_dir).with_suffix('')
log_info(2, f"data_dir set to: {data_dir}")
image_count = len(list(data_dir.glob('*/*.png')))
log_info(2, f"Number of images: {image_count}")
image_types = os.listdir(data_dir)
log_info(2, f"Image classes based on foldernames {image_types}")


log_info(1, "Loading images")
image_files = {}
image_dims = []
for type in image_types:
    image_files[type] = list(data_dir.glob(type + '/*'))
for type in image_types:
    image = PIL.Image.open(str(image_files[type][0]))
    if image_dims != []:
        #height, width, num channels
        assert [image.height, image.width, len(image.getbands())] == image_dims, f"Image types does not have the same dimensions!\n {type} has {[image.height, image.width, len(image.getbands())]}, others have {image_dims}"
    else:
        image_dims = [image.height, image.width, len(image.getbands())] 
    # image.show()

log_info(2, f"Images have dims: {image_dims}")


log_info(0, "Creating dataset from images")

batch_size = 128
train_ds = tf.keras.utils.image_dataset_from_directory(
data_dir,
validation_split=0.2,
subset="training",
seed=123,
image_size=image_dims[0:2],
color_mode="grayscale",
batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
data_dir,
validation_split=0.2,
subset="validation",
seed=123,
image_size=image_dims[0:2],
color_mode="grayscale",
batch_size=batch_size)

class_names = train_ds.class_names
log_info(2, f"Class names defined: {class_names}")

log_info(1, "Visualizing dataset")

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    used = []
    subplot_dims = int(np.ceil(np.sqrt(len(class_names))))
    for i, label in enumerate(labels[:-1]):
        if labels[i].numpy().astype("uint8") not in used:
            ax = plt.subplot(subplot_dims, subplot_dims, labels[i].numpy().astype("uint8") + 1)
            plt.imshow(images[i].numpy().astype("uint8"), cmap="gray")
            plt.title(class_names[labels[i]])
            plt.axis("off")
            used.append(labels[i].numpy().astype("uint8"))
plt.show()


for image_batch, labels_batch in train_ds:
    log_info(2, f"One traning data batch has image shape {image_batch.shape}")
    log_info(2, f"One traning data batch has label shape {labels_batch.shape}")
    break


# %%
# # Optimization of dataset

log_info(0, "Optimization of dataset")

log_info(1, "Enabling caching and prefetching of dataset for traning and validation")

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# %% 
# ## Load model

if LOAD_EXISTING_MODEL:
    log_info(0, "Loading existing model")
    new_model = tf.keras.models.load_model('my_model.tf')
    model = new_model

    # Show the model architecture
    log_info(1, "Showing model achitecture")
    new_model.summary()

# %%
# Evaluate the loaded model
if LOAD_EXISTING_MODEL:
    log_info(0, "Evaluating loaded model")
    loss, acc = new_model.evaluate(val_ds, verbose=2)
    log_info(2, 'Loaded model, accuracy: {:5.2f}%'.format(100 * acc))

# %% 
# ## Create model
if not LOAD_EXISTING_MODEL: 
    log_info(0, "Creating model")

    # ### Add data augmentation to ramdomly rotate, flip and zoom images

    log_info(1, "Add data augmentation to ramdomly rotate, flip and zoom images")
    data_augmentation = keras.Sequential(
        [
        keras.layers.RandomFlip(mode="horizontal_and_vertical", input_shape=image_dims),
        keras.layers.RandomRotation(0.1),
        keras.layers.RandomZoom(0.1),
        ]
    )

    # #### Show example

    log_info(1, "Show example of augmented image")

    plt.figure(figsize=(10, 10))
    for images, _ in train_ds.take(1):
        for i in range(9):
            augmented_images = data_augmentation(images)
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(augmented_images[0].numpy().astype("uint8"), cmap="gray")
            plt.axis("off")

    # ### Model definition
            
    log_info(1, "Defining model")

    num_classes = len(class_names)

    model = tf.keras.Sequential([
    data_augmentation,
    tf.keras.layers.Rescaling(1./255),

    tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(1,1), activation='relu', input_shape=image_dims),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),
    
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(1,1), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),
    
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(1,1), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),
    
    # tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=(1,1), activation='relu'),
    # tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu', use_bias=True),
    tf.keras.layers.Dense(32, activation='relu', use_bias=True),
    tf.keras.layers.Dense(16, activation='relu', use_bias=True),
    tf.keras.layers.Dense(num_classes)
    ])

    log_info(1, "Configuring model")

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])
    
    log_info(1, "Showing model summary")
    model.summary()

# %%
# # Model fitting

if FIT_MODEL:
    log_info(0, "Fit model")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )

    log_info(2, f"Epochs: {EPOCHS}")
else:
    history = None

# %%
# ### Summarise model

log_info(0, "Summarise relevant model layers")

for layer in model.layers:
    weights_n_bias = layer.get_weights()
    if weights_n_bias != []:
        weights = weights_n_bias[0]
        
        layer_dims = weights.shape
        
        bias = weights_n_bias[1]
        
        bias_dims = bias.shape
        print(layer.name, "weights dims:", layer_dims, "bias dims:", bias_dims)

# %% 
# ### Show training history

if history:
    log_info(0, "Show training history")

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(EPOCHS)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

# %% 
# # Evaluating model

log_info(0, "Evaluating model")
train_loss, train_acc = model.evaluate(train_ds, verbose=2)

log_info(2, f"Accuracy on training data: {train_acc}")

val_loss, val_acc = model.evaluate(val_ds, verbose=2)

log_info(2, f"Accuracy on validation data: {val_acc}")

# %% 
# # Testing inference

log_info(0, "Testing inference")

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(val_ds)
log_info(2, f"Dimensions of predictions {predictions.shape}")

def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap="gray")

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(len(class_names)), class_names, rotation=45)
  plt.yticks([])
  thisplot = plt.bar(range(len(class_names)), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

image_array = []
label_array = []
for images, labels in val_ds:
   for i, image in enumerate(images):
      image_array.append(images[i].numpy().astype("uint8"))
      label_array.append(labels[i])

log_info(1, "Showing images with predictions")

i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], label_array, image_array)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  label_array)
plt.show()

i = 20
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], label_array, image_array)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  label_array)
plt.show()

# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_cols = 5
num_rows = 5
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], label_array, image_array)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], label_array)
plt.tight_layout()
plt.show()


# %% 
# # Export test image

log_info(0, "Export images for testing of C++ implementation")

test_image_filename = "../hls_implementation/testImage.h"

log_info(2, "File path", test_image_filename)

open(test_image_filename, 'w').close() # clear file

log_info(1, "Defining new model with added softmax output layer")
softmax_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

def write_arr_to_cpp(file, arr):
    for i, num in enumerate(arr):
        file.write(f"{num}")
        if i != len(arr)-1:
            file.write(",")
        file.write("\n")
    file.write("};\n")

log_info(1, "Exporting 10 images to ", test_image_filename)
# Export 10 images
for image_index in range(0,10):
    # Flatten image
    test_image_flattened = image_array[image_index].flatten().astype("uint8")

    # Export image
    test_image_file = open(test_image_filename, "a")
    test_image_file.write(f"//Shape: {image_array[image_index].shape}\n")
    test_image_array_size = str(image_array[image_index].shape).replace("(","").replace(")","").replace(",","][").replace(" ","")
    test_image_file.write(f"int test_image{image_index}[{test_image_array_size}] = " + "{ \n")
    write_arr_to_cpp(test_image_file, test_image_flattened)

    # Predict output from image
    npimage = np.array([image_array[image_index]])
    simple_predictions = softmax_model.predict(x=npimage, batch_size=1)

    # Flatten prediction array
    simple_prediction_flat = simple_predictions[0].flatten()

    # Export classification of image
    test_image_file.write(f"//class: {np.argmax(simple_prediction_flat)}={class_names[np.argmax(simple_prediction_flat)]}\n")

    # Export classification array
    simple_prediction_array_size = str(simple_predictions.shape[1:]).replace("(","").replace(")","")[:-1]
    test_image_file.write(f"float prediction{image_index}[{simple_prediction_array_size}] = " + "{\n")
    write_arr_to_cpp(test_image_file, simple_prediction_flat)
    test_image_file.close()

# %% 
# # Show confusion matrix

log_info(0, "Show confusion matrix")

predicted_props = softmax_model.predict(train_ds)
predicticted_classes = tf.argmax(predicted_props, axis=1)
label_array = []
for images, labels in train_ds:
   for i, image in enumerate(images):
      label_array.append(labels[i])

def plot_cm(labels, predictions, dataset_name):
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix ' + dataset_name)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.xticks([i+0.5 for i in range(len(class_names))], class_names, rotation=45)
    plt.yticks([i+0.5 for i in range(len(class_names))], class_names, rotation=45)
    plt.show()

    num_mislabels = 0
    for index, label in enumerate(labels):
        if predictions[index].numpy() != label:
            num_mislabels += 1
    log_info(2, f"Total wrong labels: {num_mislabels}")
plot_cm(label_array, predicticted_classes, "training data")

predicted_props = softmax_model.predict(val_ds)
predicticted_classes = tf.argmax(predicted_props, axis=1)
label_array = []
for images, labels in val_ds:
   for i, image in enumerate(images):
      label_array.append(labels[i])
plot_cm(label_array, predicticted_classes, "validation data")

# %% 
# # Save model

if SAVE_MODEL:
    log_info(0, "Save model")
    # Save the entire model as a `.keras` zip archive.
    model_name = "my_model.tf"
    model.save(model_name)
    log_info(1, f"Model saved as {model_name}")

# %% 
# # Export layer info

log_info(0, "Export layer info (weights, dims, bias...)")

layer_info_filename = "../hls_implementation/layerInfo.hpp"

log_info(1, "Layerinfo filename ", layer_info_filename)

open(layer_info_filename, 'w').close() # clear file
layer_info_file = open(layer_info_filename, "a")

constant_arrays_prefix = "const"
datatype = "fixed"

log_info(2, f"Using datatype '{constant_arrays_prefix} {datatype}'")

# Input
log_info(1, "Exporting input dimensions")

input_array_size = str(model.layers[0].input_shape[1:]).replace("(","").replace(")","").replace(",","][").replace(" ","")
layer_info_file.write(f"//Input dimensions to neural network\n")
for i, dim in enumerate(model.layers[0].input_shape[1:]):
    layer_info_file.write(f"#define input_dim{i+1} {dim}\n")

def write_lines(file):
    file.write("//------------------------------------------------------------------\n")

write_lines(layer_info_file)

# All else

log_info(1, "Exporting layer info")

for layer_index, layer in enumerate(model.layers):
    log_info(2, f"Layer {layer_index}: {layer.name}")
    # Export name
    layer_info_file.write(f"//Layer {layer_index}: {layer.name}\n")

    # conv2d padding and strides
    if "conv2d" in layer.name or "max_pooling2d" in layer.name:
        
        log_info(2, f"Strides: {layer.get_config()['strides']}")
        layer_info_file.write(f"//strides: {layer.get_config()['strides']}\n")
        
        log_info(2, f"Padding: {layer.get_config()['padding']}")
        layer_info_file.write(f"//padding: {layer.get_config()['padding']}\n")
    
    weights_n_bias = layer.get_weights()

    if weights_n_bias == []:
        log_info(2, f"{layer.name} in {layer.input_shape} out {layer.output_shape}")
    else:
        # Weights
        weights = weights_n_bias[0]
        flat_weights = weights.flatten()
        layer_dims = weights.shape
        log_info(2, f"{layer.name} in {layer.input_shape} w {layer_dims} out {layer.output_shape}")

        # Export weights
        weights_array_size = str(weights.shape).replace("(","").replace(")","").replace(",","][").replace(" ","")
        layer_info_file.write(f"{constant_arrays_prefix} {datatype} layer_{layer_index}_weights[{weights_array_size}] = " + "{ \n")
        write_arr_to_cpp(layer_info_file, flat_weights)


        # Bias    
        bias = weights_n_bias[1]
        flat_bias = bias.flatten()
        bias_dims = bias.shape
        
        # Export bias (values added the output of each node)
        bias_array_size = str(bias.shape).replace("(","").replace(")","")[:-1].replace(",","][").replace(" ","")
        layer_info_file.write(f"{constant_arrays_prefix} {datatype} layer_{layer_index}_bias[{bias_array_size}] = " + "{ \n")
        write_arr_to_cpp(layer_info_file, flat_bias)

    
    
    # Layer outputs
    if "conv2d" in layer.name or "max_pooling2d" in layer.name or "dense" in layer.name or "flatten" in layer.name:

        # Create output array
        output_array_size = str(layer.output_shape[1:]).replace("(","").replace(")","").replace(", ","][").replace(",","")
        # Export out size
        for i, dim in enumerate(layer.output_shape[1:]):
            layer_info_file.write(f"#define layer_{layer_index}_dim{i+1} {dim}\n")



    log_info(2, "---")

    write_lines(layer_info_file)

# Export CNN output dimensions
    
log_info(1, "Exporting output dimensions")


layer_info_file.write(f"//Output dimensions of neural network\n")
for i, dim in enumerate(model.layers[-1].output_shape[1:]):
    layer_info_file.write(f"#define output_dim{i+1} {dim}\n")

layer_info_file.close()

log_info(1, f"Successfully exported layer info to {layer_info_filename}")

input("Press the <ENTER> key to end...")