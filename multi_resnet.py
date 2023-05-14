# -*- coding: utf-8 -*-

## Header =====================================================================

'''
Top level explanation is not exhaustive - code and comment inspection is highly 
recommended.

Create, train, validate and test a modified higher-channel input modified
and pretrained ResNet50v2 model for image/array based regression.

Important: Adjust settings in the USER TO EDIT section and via the input files.
Important: Additional changes have to be made at users risk inside this script.

Create, train, validate and test a custom regression model. The model consists
of a multi-channel (>3) normalised bitmap/array input layer, an ImageNet 
pretrained topless ResNet50v2 model and a head of global average pooling, and
alternating dense and dropout layers with a single neuron output.

The strategy and code for creating the multi-channel approach is taken from:
https://towardsdatascience.com/implementing-transfer-learning-from-rgb-to-multi
-channel-imagery-f87924679166
Sijuade Oguntayo
23/03/2021

The input data is as follows:

Any 'label' files - CSV containing filename stems of format 'XXXXXXXXXcN...'
e.g. t01p02s01c2 in column 1 and float style 0-1 normalised wear values in 
column 2.

Arrays - CSV files of size height x width and float style numbers normalised to
0-1 range.

Warning - A reduction in speed has been detected after a couple of batches in
all of the loops. This has not been fixed but the issue is expected to be high
memory usage / numpy related memory leakage.

Warning - The tensorflow GPU package is used to achieve parallel processing.
It is recommended that this is installed as a seperate environment using conda 
and incompatibilities are handled on a case by case basis (in this case a 
revert). Also, it is recommended to run this code in the base environment using
Spyder, but due to incompatibilities link to the correct environment via
    
Spyder>Tools>Preferences>Python interpreter>Select...
e.g. C:\\Users\\jakub\\anaconda3\\envs\\tf_gpu\\python.exe
Restart Spyder
'''
## USER TO EDIT - START =======================================================

# Full filepath of the csv file containing the normalised training data labels
label_fp = (r"C:\Users\jakub\OneDrive\Desktop\Emergency_Backup\02_Experimental"
            r"\07_Labels\Normalised_Wears.txt")
# Full filepath of the csv file containing the normalised testing data labels
label_test_fp = (r"C:\Users\jakub\OneDrive\Desktop\Emergency_Backup"
               r"\02_Experimental\07_Labels\Test_Wears.txt")
# Filepath to the parent folder containing all of the normalised bitmap arrays
array_fp = (r"C:\Users\jakub\OneDrive\Desktop\Emergency_Backup\02_Experimental"
            r"\05_Converted")
# Full filepath of the csv file for loss and metric outputs
val_outputs_fp = (r"C:\Users\jakub\OneDrive\Desktop\Emergency_Backup"
                  r"\02_Experimental\09_Metrics\Val_ResNet50v2_DeepDrop_6.csv")
# Full filepath of the csv file for test labels and predicted outputs
tes_outputs_fp = (r"C:\Users\jakub\OneDrive\Desktop\Emergency_Backup"
                  r"\02_Experimental\09_Metrics\Tes_ResNet50v2_DeepDrop_6.csv")
# Full filepath of the folder which will be created and contain the t.model
model_save_fp = (r"C:\Users\jakub\OneDrive\Desktop\Emergency_Backup"
                 r"\02_Experimental\10_Models\ResNet50v2_DeepDrop_6")

# Signal types (number must match 'input_channel' and >3) as on array filenames
sig_types = ["_frx","_fry","_frz","_acx","_acy","_acz","_mic"]

img_width = 128 # Height of bitmap arrays
img_height = 128 # Width of bitmap arrays
input_channel = 7 # Number of input arrays (>3)
output_channel = 3 # Original number of arrays (3 expected consistent with RGB)
validation_split = 0.2 # Proportion of training data for validation purposes
batch_size = 32 # All process batch size (32 recommended)
n_epochs = 20 # Number of training epochs
# Epoch at which 'warmpu' ends - all layers trainable and learning rate reduced
n_epoch_recompile = 9 # Pythonic numbering!

## USER TO EDIT - END =========================================================

## Libraries ==================================================================

# Python version: 3.9.16
# Spyder-kernels version: 2.2.1 (kernel linked and ran from base environment)

# Import libraries - internal
import os
import time
import random
from math import ceil

# Import libraries - 3rd party
import numpy as np # 1.23.5
import tensorflow as tf # 2.6.0 (GPU VERSION)
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError
from tqdm import tqdm # 4.65.0

## Define functions ===========================================================

def multify_weights(kernel, out_channels):
    '''
    For ResNet50v2, return the expanded and averaged weights for the first
    non-input layer as to enable transfer learning from a 3 channel RGB input
    to a higher channel number BW set of inputs.
    
    Inputs:
    kernel (unknown) - Keras layer weights (as from: layer.get_weights()[0]).
    out_channels (integer) - Increase in the number of channels required (>1).
        
    Outputs:
    sdata (numpy array) - Signal data trimmed of irrelevant columns.
    dtypes (list) - Type of sensor (3 chars).
    dtype_main (string) - Signal labels (3 chars).
    nlength (integer) - Signal length.
    '''
    
    # Get weight means in one dimension
    mean_1d = np.mean(kernel, axis=-2).reshape(kernel[:,:,-1:,:].shape)
    # Tile the result over the required (expanded) size of layer
    tiled = np.tile(mean_1d, (out_channels, 1))
    
    return(tiled)


def weightify(model_orig, custom_model, layer_modify):
    '''
    For a custom ResNet50v2 model (custom_model) with adjusted input layer,
    copy over the weigths from a pretrained ResNet50v2 (model_orig) and modify
    the first non-input layer (index 2) to have averaged weights from the RGB
    version and it's dimensions expanded.
    
    Inputs:
    model_orig (Keras Model) - Pretrained ResNet50v2.
    custom_model (Keras Model) - Custom input layer adjusted ResNet50v2 from 
                                 config.
    layer_modify (string) - Name of layer to be modified (index 2).
        
    Outputs:
    N/A - changes made to the custom_model instead.
    '''
    
    # Package layer to modify name in a list
    layer_to_modify = [layer_modify]

    # Get custom model configuration and extract the layer names
    conf = custom_model.get_config()
    layer_names = [conf['layers'][x]['name'] for x in 
                   range(len(conf['layers']))]

    # Loop through layers of the original model
    for layer in model_orig.layers:
        # If layer exists in both custom and original...
        if layer.name in layer_names:
            #If layer has weights...
            if layer.get_weights() != []:
                # Get layer name from the custom model
                target_layer = custom_model.get_layer(layer.name)
                
                # If layer is set to modify, average and expand
                # (see multify_weights)
                if layer.name in layer_to_modify:    
                    # Get original weights
                    kernels = layer.get_weights()[0]
                    biases  = layer.get_weights()[1]
                    
                    # Adjust weights as in multify_weights and set
                    res = (kernels,
                           multify_weights(kernels,
                                           input_channel - output_channel))
                    kernels_extra_channel = np.concatenate(res, axis=-2)
                    target_layer.set_weights([kernels_extra_channel, biases])
                    
                    # Set layer to be untrainable (optional - changes later)
                    target_layer.trainable = True

                else:
                    # Copy weights over
                    target_layer.set_weights(layer.get_weights())
                    # Set rest of model to be untrainable (optional - changes^)
                    target_layer.trainable = False


def build_model():
    '''
    Build a custom ResNet50v2 model with an adjusted input layer (more
    channels), ImageNet pretrained weights (first layer averaged and expanded)
    as well as a custom regression head.
    
    Inputs:
    N/A
        
    Outputs:
    model_final (Keras Model) - Partially pretrained custom model.
    '''
    
    # Create input layer
    inputs = Input((img_height, img_width, input_channel))
    
    # Load pretrained ResNet50v2
    model_original = ResNet50V2(weights='imagenet', include_top=False)

    # Get configuration (structure)
    config = model_original.get_config()
    # Adjust input layer in configuration
    config["layers"][0]["config"]["batch_input_shape"] = (None, img_height,
                                                          img_width,
                                                          input_channel)
    
    # \/\/\/ May be beneficial to increase feature sensitivity - tested but not
    # validated in any form
    #config["layers"][2]["config"]["strides"] = (1, 1)
    
    # Build skeleton of custom model
    model_custom = tf.keras.models.Model.from_config(config)
    # Modify the name of the first non-input layer
    modify_name = config["layers"][2]["config"]["name"]

    # Copy weights over from pretrained, average and expand layer /\
    weightify(model_original, model_custom, modify_name)
    # Link to input layer
    model_custom_tensor = model_custom(inputs)

    # Build custom head
    model_custom_tensor = GlobalAveragePooling2D()(model_custom_tensor)
    model_custom_tensor = Dense(512, activation="relu")(model_custom_tensor)
    # model_custom_tensor = Dropout(rate=0.4,seed=1)(model_custom_tensor)
    # model_custom_tensor = Dense(512, activation="relu")(model_custom_tensor)
    # model_custom_tensor = Dropout(rate=0.4,seed=2)(model_custom_tensor)
    # model_custom_tensor = Dense(512, activation="relu")(model_custom_tensor)
    model_custom_tensor = Dropout(rate=0.3,seed=1)(model_custom_tensor)
    model_custom_tensor = Dense(512, activation="relu")(model_custom_tensor)
    model_custom_tensor = Dropout(rate=0.2,seed=2)(model_custom_tensor)
    output = Dense(1, activation='sigmoid')(model_custom_tensor)

    # Combine all components into a model
    model_final = Model(inputs, output)

    return model_final

def get_arrays(filestems):
    '''
    Load a batch of input arrays/bitmaps from the predefined folder
    using the filestems and sensor types.
    
    Inputs:
    filestems (list) - Filename stems e.g. [t01p02s01c1, ...].
        
    Outputs:
    out_arrays (np.array) - Batch of arrays ready for input into model.
    '''
    
    # Initialise output array in the right size
    out_arrays_size = (len(filestems),img_width,img_height,input_channel)
    out_arrays = np.empty(out_arrays_size, dtype='float')
    
    # For all items in input (filestems)...
    for filestem_idx in range(len(filestems)):
        
        # Load arrays from each sensor
        filestem = filestems[filestem_idx]
        filenames = [filestem + sig_type + ".csv" for sig_type in sig_types]
        out_fps = [os.path.join(array_fp,filename) for filename in filenames]
        arrays = [np.loadtxt(out_fp,'float','#',',') for out_fp in out_fps]
        
        # Stitch these together and arrange in the output array
        out_array = np.empty(out_arrays_size[-3:], dtype='float')
        np.stack(arrays, axis=2, out=out_array)
        out_arrays[filestem_idx,:,:,:] = out_array
    
    return(out_arrays)

## Main: Create Model =========================================================

# Print useful information for understanding if a GPU is visible to tensorflow
print(tf.__version__)
print(tf.config.list_physical_devices())

# Build the model
model = build_model()
model.summary() # Print model summary to console

## Main: Prep Inputs ==========================================================

# Load labels csv as numpy array of strings (ignore lines starting with #)
labels_matrix = np.loadtxt(label_fp,'str','#',',')
# Shuffle to get random order of training files and validation files
np.random.seed(0)
np.random.shuffle(labels_matrix)

# Position to split into training and validation sets (rounding favours valid.)
split_pos = ceil(labels_matrix.shape[0] * validation_split)

# Split into filestems and labels lists, change labels to float
filestems = labels_matrix[:,0].tolist()
labels = [float(label) for label in labels_matrix[:,1].tolist()]

# Split filestems
filestems_validate = filestems[-split_pos:]
filestems_train = filestems[:-split_pos]

# Split labels (positionally in sync with filestems)
labels_validate = labels[-split_pos:]
labels_train = labels[:-split_pos]

# Initialise metric histories
loss_history = []
val_loss_history = []
met_history = []
val_met_history = []

## Main: Compile, Train, Validate =============================================

# Compile initial (warmup) model
model.compile(optimizer=Adam(learning_rate=1e-5), loss='mae',
              metrics=[RootMeanSquaredError()])

start = time.time() # Note start time

# Run through epochs
for epoch in range(n_epochs):
    
    # Reshuffle training set
    shuffle_train = list(zip(filestems_train, labels_train))
    random.seed(epoch)
    random.shuffle(shuffle_train)
    shuffled_filestems_train, shuffled_labels_train = zip(*shuffle_train)
    
    # Initialise metrics lists for current epoch
    losses, mets = [], []
    val_losses, val_mets = [], []
    
    # If warmup phase is over, recompile the model with new settings
    if epoch == n_epoch_recompile:
        # Set all layers to trainable
        for layer in model.layers:
            layer.trainable = True
        
        # Recompile the model
        model.compile(optimizer=Adam(learning_rate=1e-4), loss='mae',
                      metrics=[RootMeanSquaredError()])
        model.summary() # Print model summary to console
    else:
        pass
    
    # Number of batches (round up to include one that isn't full)
    n_batches = ceil(len(filestems_train)/batch_size)
    
    # Loop through all of the batches
    for n_batch in tqdm(range(n_batches)):
        # Index of first item to be used in batch
        first = n_batch*batch_size
        # Extract batch of filestems and labels
        batch_filestems = shuffled_filestems_train[first:first+batch_size]
        batch_labels = np.array(shuffled_labels_train[first:first+batch_size])
        # Get input arrays for this batch using the filestems
        batch_arrays = get_arrays(batch_filestems)

        # Train model on this batch and record metrics
        loss, met = model.train_on_batch(batch_arrays, batch_labels)
        
        # Append metrics to lists (length multiplication undone at averaging)
        losses.append(loss * len(batch_filestems)) 
        mets.append(met * len(batch_filestems))  
    
    # Calculate average metric training values for this epoch
    loss_avg = sum(losses) / (len(losses) * batch_size)
    met_avg = sum(mets) / (len(mets) * batch_size)
    
    # Numer of validation data batches (named splits for readability)
    n_splits = ceil(len(filestems_validate)/(batch_size))
    
    # Loop through all of the splits
    for n_split in tqdm(range(n_splits)):
        # Index of first item to be used in split
        first = n_split*batch_size
        # Extract split of filestems and labels
        split_filestems = filestems_validate[first:first+batch_size]
        split_labels = np.array(labels_validate[first:first+batch_size])
        # Get input arrays for this split using the filestems
        split_arrays = get_arrays(split_filestems)
        
        # Evaluate the model on this split and record metrics
        val_loss, val_met = model.evaluate(split_arrays, split_labels)
        
        # Append metrics to lists (length multiplication undone at averaging)
        val_losses.append(val_loss * len(split_filestems))
        val_mets.append(val_met * len(split_filestems))    
    
    # Calculate average metric validation values for this epoch
    val_loss_avg = sum(val_losses) / (len(val_losses) * batch_size)
    val_met_avg = sum(val_mets) / (len(val_mets) * batch_size)
    
    # Append epoch summary information to lists
    loss_history.append(loss_avg)
    met_history.append(met_avg)
    val_loss_history.append(val_loss_avg)
    val_met_history.append(val_met_avg)
    
    # Sleep to prevent missprint in console
    time.sleep(1)

    # Print summary information at epoch end
    print('')
    print('Epoch: %d, Train: MAE Loss %.3f, RMSE Metric %.3f' %
			(epoch+1, loss_avg, met_avg))
    print('Epoch: %d, Validation: MAE Loss %.3f, RMSE Metric %.3f' %
			(epoch+1, val_loss_avg, val_met_avg))
    print('')
    
## Main: Test =================================================================

# Load training labels (intentionally a different set)
labels_test_matrix = np.loadtxt(label_test_fp,'str','#',',')

# Convert labels into a list of floats
filestems_test = labels_test_matrix[:,0].tolist()
labels_test = [float(label) for label in labels_test_matrix[:,1].tolist()]

# Number of test batches (round up to include one that isn't full)
n_test_batches = ceil(len(filestems_test)/batch_size)

# Initialise predictions list
all_predictions = []

# Loop through all of the test batches
for n_test_batch in tqdm(range(n_test_batches)):
    # Index of first item to be used in test batch
    first = n_test_batch*batch_size
    # Extract test  of filestems and labels
    batch_test_filestems = filestems_test[first:first+batch_size]
    
    # Get input arrays for this test batch using the filestems
    batch_arrays = get_arrays(batch_test_filestems)
    
    # Get label predictions from model and change format
    predictions_np = model.predict_on_batch(batch_arrays)
    predictions = predictions_np.tolist()
    
    # Append predictions to list
    all_predictions.append(predictions)

# Change format of predcitions list to be plain floats
all_predictions = [item[0] for sublist in all_predictions for item in sublist]

# Note end time and print completion time
end = time.time()
print("Time to train, validate and test the model:", end - start)

## Main: Save Model, Metrics, Test Predictions ================================

# Save the trained model
model.save(model_save_fp)

# Save the metric histories in a csv format
with open(val_outputs_fp,'a+') as f:
    for idx in range(len(loss_history)):
        string = str(loss_history[idx]) + ","
        string += str(val_loss_history[idx]) + ","
        string += str(met_history[idx]) + ","
        string += str(val_met_history[idx])
        f.write(string + "\n")
    
# Save the test predictions and labels
with open(tes_outputs_fp,'a+') as f:
    for idx in range(len(labels_test)):
        string = str(all_predictions[idx]) + ","
        string += str(labels_test[idx])
        f.write(string + "\n")