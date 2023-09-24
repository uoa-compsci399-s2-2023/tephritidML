#### Imports
from enum import Enum
import os, shutil
from glob import glob

from PIL import Image, ImageOps, ImageFilter
import numpy as np
from scipy.special import softmax
from tqdm import tqdm
import tensorflow as tf


# Keras models
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.xception import Xception, preprocess_input

# Keras utilities
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.python.keras.utils.generic_utils import CustomObjectScope # Needed because MobileNet uses this custom 'relu6' function

#SageMaker Packing
import tarfile

### Configs
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config = config)
class Freeze(Enum):
  inception_v3 = (0,312)
  mobilenet = (0,83)
  xception = (0,133)

class Input_Size(Enum):
  inception_v3 = (299,299)
  mobilenet = (224,224)
  xception = (299,299)
### Functions
def preprocess_images(inputDir, outputDir, size=299):
    """
    Prepares the images for training/testing
    """
    for i, filename in tqdm(enumerate(os.listdir(inputDir))):
        filename_raw, ext = os.path.splitext(filename)
        # print("'{}'".format(ext))
        if ext.lower() in ['.jpg', '.tif', '.png', '.bmp']:
            in_file_path = os.path.join(inputDir, filename)
            out_file_path = os.path.join(outputDir, filename_raw + '.png')
            # print("{}: converting {} => {}".format(i+1, in_file_path, out_file_path))
            preprocess_image(in_file_path, out_file_path, size)

def preprocess_image(in_path, out_path, size=299):
    """
    Prepares the images for training/testing:
    - greyscale
    - resize (changes aspect). Note: assumes all are the same size, which they aren't! They have been cropped.
    """
    img = Image.open(in_path, mode='r')

    # Denoise
    # img = img.filter(ImageFilter.UnsharpMask) # POOR
    img = img.filter(ImageFilter.BLUR) # Works well
    
    # Grey scale
    img = np.array(ImageOps.grayscale(img)).astype(float)
    
    """
    # Standardise to 0..255 - NOT USED (doesn't alter the brightness)
    min = np.min(img)
    max = np.max(img)
    # print("min={} max={}".format(min, max))
    img = img * (255.0 / (max-min)) # Adjust contrast
    # print("New min={} max={}".format(np.min(img), np.max(img)))
    img = img - np.min(img)
    # print("Final min={} max={}".format(np.min(img), np.max(img)))
    """
    
    # Normalise to +- 2SD
    # print("OLD mean={} min={} max={}".format(np.mean(img), np.min(img), np.max(img)))
    vmin = int(np.mean(img) - (2 * np.std(img)))
    vmax = int(np.mean(img) + (2 * np.std(img)))
    img = np.clip(img, vmin, vmax)
    img -= vmin
    img *= 256. / (vmax - vmin)

    # print("NEW mean={} min={} max={}".format(np.mean(img), np.min(img), np.max(img)))
    
    """
    mean = np.mean(img)
    print("mean={}".format(mean))
    img = img + (127 - mean) # Adjust brightness
    print("New mean={}".format(np.mean(img)))
    """
    
    img = Image.fromarray(img.astype('uint8'),'L')
    
    # Resize - changes the aspect ratio
    img = img.resize((size, size))
    # img = img.convert('RGB')
    
    img.save(out_path)

def reorg_folders(in_path, out_path, index_file, label_name):
    '''
    - Get the list of files and their class from the spreadsheet
    - For each image:
        - filename = out_path + class/filename
        - create class folder if missing
        - copy the file
    '''
    
    # spreadsheet columns
    COLUMNS = ['file', 'country', 'genus', 'species' , 'fullName', 'fullName2', 'fullName3', 'accessionNumber', 'view', 'project']
    filename_col = 0
    label_col = COLUMNS.index(label_name)
    
    # Read the csv
    with open(index_file, 'r') as ifile:
        index = [line.split(',')  for line in ifile.readlines()]

    for i,item in tqdm(enumerate(index[1:])):
        filename = item[filename_col]
        filename_raw, ext = os.path.splitext(filename)
        filename = filename_raw + '.png'
        label = item[label_col].strip().lower().replace(' ', '_')

        in_file_path = os.path.join(in_path, filename)
        out_file_dir = os.path.join(out_path, label)
        
        # Create class folder if needed
        if not os.path.exists(out_file_dir):
            os.makedirs(out_file_dir)
   
        # Copy image to class folder
        shutil.copy(in_file_path, out_file_dir)
    print('ALL DONE, ITS A WRAP.')
    
def train(model_name, train_data_dir, log_dir, dataset_name, 
            epochs = 200, batch_size = 16, lr = 1e-4, fine_tune = True, patience = 30):
  """
  learns a new domain from an imagenet network
  """

  print("Retraining an Imagenet network for dataset '{}'".format(dataset_name))
  print("  Training data:", train_data_dir)
  print("  Model log dir:", log_dir)
  print("  Model:", model_name)
  print("  Epochs:", epochs)
  print("  Batch size:", batch_size)
  print("  Learning rate:", lr)
  print("  Fine_tune:", fine_tune)

  num_classes = len(os.listdir(train_data_dir))
  

  if model_name == 'InceptionV3':
    freeze_between = Freeze.inception_v3.value
    image_size = Input_Size.inception_v3.value
    image_shape = image_size + (3,)
    base_model = InceptionV3(include_top = False, weights = 'imagenet', input_shape = image_shape)
  elif model_name == 'Xception':
    freeze_between = Freeze.xception.value
    image_size = Input_Size.xception.value
    image_shape = image_size + (3,)
    base_model = Xception(include_top = False, weights = 'imagenet', input_shape = image_shape)
  elif model_name == 'MobileNet':
    freeze_between = Freeze.mobilenet.value
    image_size = Input_Size.mobilenet.value
    image_shape = image_size + (3,)
    base_model = MobileNet(include_top = False, weights = 'imagenet', input_shape = image_shape)
  else:
    print("ERROR: Unknown model {}: supported models are 'InceptionV3', 'MobileNet' and 'Xception'".format(model_name))
    return None

  # Create and randomly initialize the dense top layer
  x = base_model.output
  x = GlobalAveragePooling2D()(x)

  predictions = Dense(num_classes, activation = 'softmax')(x)

  model = Model(inputs = base_model.inputs, outputs = predictions)

  # Retraining: freeze all layers except the new head
  for i in range(freeze_between[0], freeze_between[1]):
    model.layers[i].trainable = False 
    
  # compile model for training 
  model.compile(optimizer = tf.keras.optimizers.Adam(lr), loss = 'categorical_crossentropy', metrics = ['accuracy'])
  model.summary()
  
  # define data generators
  # Set the augmentations. Small datasets need to be aggressive...
  # TODO: should be passed in by caller...

  # Minor augmentation (for wasps)
  train_datagen = ImageDataGenerator(preprocessing_function = preprocess_input, # scales to -1...1 for Xception
                                     rotation_range = 25, # Some are tilted (prev 25)
                                     width_shift_range = 0.1,
                                     height_shift_range = 0.1,
                                     zoom_range = 0.1,
                                     # brightness_range = (0.5,1.5)
                                     # channel_shift_range = 20, # [0..255] # not for monochrome
                                     )

                                     
  #valid_datagen = ImageDataGenerator(rescale = 1./255)
  # valid_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)

  train_generator = train_datagen.flow_from_directory(train_data_dir, target_size = image_size, batch_size = batch_size)
  # valid_generator = valid_datagen.flow_from_directory(valid_data_dir, target_size = image_size, batch_size = batch_size)
    
  # refit model and return
  print("Retraining {} model...".format(model_name))

  early_stopping_monitor = EarlyStopping(monitor = 'accuracy', verbose = 1, patience = patience)
  log_file_name = os.path.join(log_dir, '{}_{}_final_transfer_earlyStop_log.csv'.format(dataset_name, model_name))
  csv_logger = CSVLogger(log_file_name)
  callbacks = [early_stopping_monitor, csv_logger]
  model.fit(train_generator, epochs = epochs, callbacks = callbacks)
  print("THIS RAN")
  # Fine-tune the lower layers (if required)
  if fine_tune:
    # Set the number of fine-tune epochs and learning rate. TODO - is there a smart way to tune this?
    fine_tune_epochs = epochs
    fine_tune_lr = lr/10

    print("Fine-tuning the {} network for a further {} epochs with lr = {}".format(model_name, fine_tune_epochs, fine_tune_lr))
    for i in range(freeze_between[0], freeze_between[1]):
      model.layers[i].trainable = True   

    model.compile(optimizer = tf.keras.optimizers.Adam(fine_tune_lr), loss = 'categorical_crossentropy', metrics = ['accuracy'])     
    model.fit(train_generator, epochs = fine_tune_epochs, callbacks = callbacks)
  else:
    print("Fine-tuning NOT REQUESTED")

  # Save the model to disk
  # model_file_name = os.path.join(model_dir, '{}_{}_transfer.h5'.format(dataset_name, model_name))
  # model.save(model_file_name)

  # Save the model for SageMaker
  model.save("export/Servo/1")
  with tarfile.open("model.tar.gz", "w:gz") as tar:
    tar.add("export")

  # All done, clear keras' global state to avoid memory leaks
  K.clear_session()
  
  print("SageMaker Model saved")
  return
  print("Done. Model saved to {}".format(model_file_name))
  return model_file_name

## Execution
##### Flags
preprocess = False
##### Pre-Processing Data
print("Root dir: {}".format(os.getcwd()))
if preprocess:
    inputDir = "./img/img_raw/"
    outputDir = "./img/img_processed"
    sortedDir = "./img/img_sorted"
    labelFile = "./labels/fruitfly_annotationfile.csv"
    preprocess_images(inputDir, outputDir)
    reorg_folders(outputDir, sortedDir, labelFile, 'notUsed')
model_name = "Xception"
train_data_dir = "./img/img_sorted"
log_dir = "logs"
dataset_name = "trupanea_v2_full"
train(model_name, train_data_dir, log_dir, dataset_name)
