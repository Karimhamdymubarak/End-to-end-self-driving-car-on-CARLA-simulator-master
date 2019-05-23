"""
Created on Tue Mar 19 01:04:59 2019
@author: omar sha3rawy
"""

#imports
import time
import tensorflow as tf
from tensorflow import keras
from keras.optimizers import Adam
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, LSTM, Conv1D, Multiply
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D, Add, Concatenate
from keras.models import Model, load_model
from keras.utils import plot_model
from keras import backend as K
# gpu memory optimizing 
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.tensorflow_backend.set_session(tf.Session(config=config))
import h5py
import numpy as np

#training setup
EPOCHS = 40
#number of files for each branch 
data_size = 8000
#number of files included in the batch (default step=1 - batchsize=32)
step = 1 
#save weight every n epochs (default SaveEvery=1 - save after every epoch)
SaveEvery=1
# path to the dataset
dataset_path = '/media/dell1/1.6TBVolume/Model/Data/AgentHuman/SeqTrain'
# saving path for model and weights
saving_path = '/media/dell1/1.6TBVolume/Model/Experiements/original'

#data generator and prepration 
def fetch(start, step, classes):
    
    file_size=32
    batch_size=file_size*step
    images = np.zeros((batch_size, 88, 200, 3))
    speed_vec = np.zeros((batch_size,))
    steering_vec = np.zeros((batch_size,))
    throttle_vec = np.zeros((batch_size,))
    brake_vec = np.zeros((batch_size,))
        
    for i in range(0, step):
        
        filename=dataset_path+'/'+classes+'/'+classes+'_Batch'+'/data_0'+str(start+i)+'.h5'
        # load files and extract labels and features        
        with h5py.File(filename, 'r') as hdf:
            imgs = hdf.get('rgb')
            imgs = np.array(imgs[:,:,:])
            targets = hdf.get('targets')
            targets = np.array(targets)
        # normalize the images
        images[i*file_size:i*file_size+file_size]=imgs/255.0
        steering_vec[i*file_size:i*file_size+file_size]=targets[:,0] 
        speed_vec[i*file_size:i*file_size+file_size]=targets[:,10]
        throttle_vec[i*file_size:i*file_size+file_size]=targets[:,1]
        brake_vec[i*file_size:i*file_size+file_size]=targets[:,2]
    
    return images, speed_vec, steering_vec, throttle_vec,brake_vec

#Define layers blocks
def conv_block(last_activations,kernel,stride,number_of_filters,drop=None):

    con = Conv2D(number_of_filters, (kernel, kernel), strides = stride,padding='same', activation='elu',kernel_initializer = 'he_normal')(last_activations)
    con = BatchNormalization()(con)
    if drop != None :
        con = Dropout(drop)(con)
    return con

def fc_block(last_activations,number_of_units,drop=None):
    
    fc = Dense(number_of_units, activation = 'elu',kernel_initializer = 'he_normal')(last_activations)
    if drop != None :
        fc = Dropout(drop)(fc)
    return fc

"Big hero 4 archetecture"

"Image module"

#image input layer
input_image=(88, 200, 3)
image = Input(input_image)
'layer 1 - CONV'
X=conv_block(image,5,2,24,drop=None)
'layer 2 - CONV'
X=conv_block(X,5,2,36,drop=None)
'layer 3 - CONV'
X=conv_block(X,5,2,48,drop=None)
'layer 4 - CONV'
X=conv_block(X,3,1,64,drop=None)
'layer 5 - CONV'
X=conv_block(X,3,1,64,drop=None)
'flatten layer'
X = Flatten(name = 'Flatten')(X)
'layer 1 - FC'
X=fc_block(X,1164,drop=0.005)
'layer 2 - FC'
X=fc_block(X,512,drop=0.05)

"Speed module"

#speed input layer'
speed=(1,)
speed_input = Input(speed)
'layer 1 - FC'
Y=fc_block(speed_input,128,drop=None)
'layer 2 - FC'
Y=fc_block(Y,128,drop=None)


"concatenation layer"

#concate the 2 modules
j=Concatenate()([X, Y])

"start branching"

"left branch"

'layer 1 - FC'
left_branch=fc_block(j,256,drop=None)
'layer 2 - FC'
left_branch=fc_block(left_branch,256,drop=0.1)
'layer 3 - FC'
left_branch=fc_block(left_branch,128,drop=0.001)
'output layer - FC'
left_branch_steering = Dense(1, activation = None, kernel_initializer = 'he_normal',name="left_branch_steering")(left_branch)
left_branch_gas = Dense(1, activation = None, kernel_initializer = 'he_normal',name="left_branch_gas")(left_branch)
left_branch_brake = Dense(1, activation = None, kernel_initializer = 'he_normal',name="left_branch_brake")(left_branch)

"right branch"

'layer 1 - FC'
right_branch=fc_block(j,256,drop=None)
'layer 2 - FC'
right_branch=fc_block(right_branch,256,drop=0.1)
'layer 3 - FC'
right_branch=fc_block(right_branch,128,drop=0.001)
'output layer - FC'
right_branch_steering = Dense(1, activation = None, kernel_initializer = 'he_normal',name="right_branch_steering")(right_branch)
right_branch_gas = Dense(1, activation = None, kernel_initializer = 'he_normal',name="right_branch_gas")(right_branch)
right_branch_brake = Dense(1, activation = None, kernel_initializer = 'he_normal',name="right_branch_brake")(right_branch)

"follow branch"

'layer 1 - FC'
follow_branch=fc_block(j,256,drop=None)
'layer 2 - FC'
follow_branch=fc_block(follow_branch,256,drop=0.1)
'layer 3 - FC'
follow_branch=fc_block(follow_branch,128,drop=0.001)
'output layer - FC'
follow_branch_steering = Dense(1, activation = None, kernel_initializer = 'he_normal',name="follow_branch_steering")(follow_branch)
follow_branch_gas = Dense(1, activation = None, kernel_initializer = 'he_normal',name="follow_branch_gas")(follow_branch)
follow_branch_brake = Dense(1, activation = None, kernel_initializer = 'he_normal',name="follow_branch_brake")(follow_branch)

"straight branch"

'layer 1 - FC'
str_branch=fc_block(j,256,drop=None)
'layer 2 - FC'
str_branch=fc_block(str_branch,256,drop=0.1)
'layer 3 - FC'
str_branch=fc_block(str_branch,128,drop=0.001)
'output layer - FC'
str_branch_steering = Dense(1, activation = None, kernel_initializer = 'he_normal',name="str_branch_steering")(str_branch)
str_branch_gas = Dense(1, activation = None, kernel_initializer = 'he_normal',name="str_branch_gas")(str_branch)
str_branch_brake = Dense(1, activation = None, kernel_initializer = 'he_normal',name="str_branch_brake")(str_branch)

"speed branch"

'layer 1 - FC'
speed_branch=fc_block(X,256,drop=0.01)
'layer 2 - FC'
speed_branch=fc_block(speed_branch,256,drop=0.01)
'output layer - FC'
speed_branch_output = Dense(1, activation = None, kernel_initializer = 'he_normal',name="speed_branch_output")(speed_branch)

"Define the model"

BH1 = Model(inputs = [image, speed_input],outputs = [left_branch_steering, left_branch_gas,left_branch_brake,
                                                     right_branch_steering, right_branch_gas,right_branch_brake,
                                                     follow_branch_steering, follow_branch_gas,follow_branch_brake,
                                                     str_branch_steering, str_branch_gas,str_branch_brake,
                                                     speed_branch_output])

"Compile the model"

# define custom loss function to do masking on branches
def masked_loss_function(y_true, y_pred):
    mask_value=-2
    mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())
    return keras.losses.mean_absolute_error(y_true * mask, y_pred * mask)

#define custom optimizer
opt=Adam(lr=0.0002, beta_1=0.7, beta_2=0.85, decay=1e-6)

BH1.compile(optimizer = opt, loss = {'left_branch_steering': masked_loss_function,
                                                         'left_branch_gas': masked_loss_function,
                                                         'left_branch_brake': masked_loss_function,                                            
                                                         'right_branch_steering': masked_loss_function,
                                                         'right_branch_gas': masked_loss_function,
                                                         'right_branch_brake': masked_loss_function,                                                         
                                                         'follow_branch_steering': masked_loss_function,
                                                         'follow_branch_gas': masked_loss_function,
                                                         'follow_branch_brake': masked_loss_function,
                                                         'str_branch_steering': masked_loss_function,
                                                         'str_branch_gas': masked_loss_function,
                                                         'str_branch_brake': masked_loss_function,
                                                         'speed_branch_output': masked_loss_function},
                                                          loss_weights = {'left_branch_steering': 0.4275,
                                                         'left_branch_gas': 0.4275,
                                                         'left_branch_brake': 0.0475,
                                                         'right_branch_steering': 0.4275,                                                        
                                                         'right_branch_gas': 0.4275,
                                                         'right_branch_brake': 0.0475,                                                      
                                                         'follow_branch_steering': 0.4275,                                                        
                                                         'follow_branch_gas': 0.4275,
                                                         'follow_branch_brake': 0.0475,                                                       
                                                         'str_branch_steering': 0.4275,                                                        
                                                         'str_branch_gas': 0.4275,
                                                         'str_branch_brake': 0.0475,                                                            
                                                         'speed_branch_output': 0.05})
    
"save the the model"

BH1.save(saving_path+'/'+'BH1_Nvidia.h5')
BH1_history = []
#measure the train time
start = time.time() 

"start training"

print("start training")
#define mask batch to decativate unwanted branches 
masked=np.full((32,), -2)
# training loop
for EPOCH in range(0, EPOCHS):
    #measure epoch time 
    start_epoch = time.time()
    # loop on the dataset for each branch
    for i in range(0, int(data_size/step), step):
        # activate left branch and deactivae others
        frames, speed_vec, steering_labels, gaz_labels,brake_labels = fetch(i, step, 'Left')
        hist1 = BH1.fit(x = [frames, speed_vec], y = [steering_labels, gaz_labels,brake_labels,
                                    masked, masked, masked,
                                    masked, masked, masked,
                                    masked, masked, masked,
                                    speed_vec],
                                    batch_size = 32*step,
                                    epochs = 1,
                                    shuffle = True,
                            #        callbacks=[PlotLossesCallback()],
                                    verbose = 0)
        BH1_history.append(hist1.history['loss'])
        
        # activate right branch and deactivae others
        frames, speed_vec, steering_labels, gaz_labels,brake_labels = fetch(i, step, 'Right')
        hist1 = BH1.fit(x = [frames, speed_vec], y = [masked, masked, masked,
                                    steering_labels, gaz_labels,brake_labels,
                                    masked, masked, masked,
                                    masked, masked, masked,
                                    speed_vec],
                                    batch_size = 32*step,
                                    epochs = 1,
                                    shuffle = True,
                            #        callbacks=[PlotLossesCallback()],
                                    verbose = 0)
        BH1_history.append(hist1.history['loss'])
        
        # activate follow branch and deactivae others
        frames, speed_vec, steering_labels, gaz_labels,brake_labels = fetch(i, step, 'Follow')
        hist1 = BH1.fit(x = [frames, speed_vec], y = [masked, masked, masked,
                                    masked, masked, masked,
                                    steering_labels, gaz_labels,brake_labels,
                                    masked, masked, masked,
                                    speed_vec],
                                    batch_size = 32*step,
                                    epochs = 1,
                                    shuffle = True,
                            #        callbacks=[PlotLossesCallback()],
                                    verbose = 0)
        BH1_history.append(hist1.history['loss'])
        
        # activate Straight branch and deactivae others
        frames, speed_vec, steering_labels, gaz_labels,brake_labels = fetch(i, step, 'Straight')
        hist1 = BH1.fit(x = [frames, speed_vec], y = [masked, masked, masked,
                                    masked, masked, masked,
                                    masked, masked, masked,
                                    steering_labels, gaz_labels,brake_labels,
                                    speed_vec],
                                    batch_size = 32*step,
                                    epochs = 1,
                                    shuffle = True,
                            #        callbacks=[PlotLossesCallback()],
                                    verbose = 0)
        BH1_history.append(hist1.history['loss'])
        #print the loss each 800 batch
        if (i+1)%800==0 :
            print ("BH1 : loss at "+str(i+1)+" = "+str(hist1.history['loss']))
            print("iteration= "+str((i+1)/800)+" of total number of iteration = 10 at epoch number "+str(EPOCH+1)+"\n")

    end_epoch = time.time()
    print('\nEpoch Time = ' + str((end_epoch-start_epoch)/60) +" minute "+ '\n')
    #save weights every epoch
    if (EPOCH+1)%SaveEvery==0 :
        BH1.save(saving_path +'/'+'BH1_Nividia_at_epoch_'+str(EPOCH+1)+'.h5')
        print('Weights Saved Successfully\n')   

end = time.time()
print('\nTotal Training Time = '+str((end-start)/60)+" minute "+ '\n')

