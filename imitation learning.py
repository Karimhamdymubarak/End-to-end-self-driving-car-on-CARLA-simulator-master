# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 02:45:08 2019

@author: Ali Abdelkhalek
"""
#imports
from __future__ import print_function
import os
import scipy
from tensorflow import keras
import tensorflow as tf
import numpy as np
from carla.agent import Agent
from carla.carla_server_pb2 import Control
from agents.imitation.imitation_learning_network import load_imitation_learning_network

# define custom loss function to do masking on branches
def masked_loss_function(y_true, y_pred):
    mask_value=-2
    mask = keras.backend.cast(keras.backend.not_equal(y_true, mask_value), keras.backend.floatx())
    return keras.losses.mean_absolute_error(y_true * mask, y_pred * mask)

class ImitationLearning(Agent):
    def __init__(self, city_name, avoid_stopping, memory_fraction=0.25, image_cut=[115, 510]):
        Agent.__init__(self)
        self._image_size = (88, 200, 3)
        self._avoid_stopping = avoid_stopping

        # Reading models and weights

        dir_path = 'E:\GP\org data less'
        self._model_path = dir_path + '/BH1_Nvidia.h5'
        self._weights_path = dir_path + '/BH1_Nividia_at_epoch_40.h5'
        self._image_cut = image_cut
        self.model = keras.models.load_model(self._model_path, custom_objects={'masked_loss_function': masked_loss_function})
        self.model.load_weights(self._weights_path)
    
    #gets features from simulator
    def run_step(self, measurements, sensor_data, directions, target):

        control = self._compute_action(sensor_data['CameraRGB'].data,
                                       measurements.player_measurements.forward_speed, directions)
        return control

    #get the action and pass them to the simulator 
    def _compute_action(self, rgb_image, speed, direction):
        #cut dummy parts from images
        rgb_image = rgb_image[self._image_cut[0]:self._image_cut[1], :]
        image_input = scipy.misc.imresize(rgb_image, [self._image_size[0],
                                                      self._image_size[1]])
        image_input = image_input.astype(np.float32)
        image_input = np.multiply(image_input, 1.0 / 255.0)

        steer, acc, brake = self._control_function(image_input, speed, direction)

        # This a bit biased, but is to avoid fake breaking

        if brake < 0.1:
            brake = 0.0

        if acc > brake:
            brake = 0.0

        # We limit speed to 35 km/h to avoid
        if speed > 10.0 and brake == 0.0:
            acc = 0.0

        control = Control()
        control.steer = steer
        control.throttle = acc
        control.brake = brake

        control.hand_brake = 0
        control.reverse = 0

        return control
    
    #predict actions using the model 
    def _control_function(self, image_input, speed, control_input):

        image_input = image_input.reshape(
            (1, self._image_size[0], self._image_size[1], self._image_size[2]))

        # Normalize with the maximum speed from the training set ( 90 km/h)
        speed = np.array(speed * 3.6)

        speed = speed.reshape((1, 1))

        print('current speed',speed)

        output_all = self.model.predict([image_input, speed])

        if control_input == 2 or control_input == 0.0:

        	predicted_steers = (output_all[6][0])
        	predicted_acc = (output_all[7][0])
        	predicted_brake = (output_all[8][0])

        elif control_input == 3:
        	predicted_steers = (output_all[0][0])
        	predicted_acc = (output_all[1][0])
        	predicted_brake = (output_all[2][0])

        elif control_input == 4:

        	predicted_steers = (output_all[3][0])
        	predicted_acc = (output_all[4][0])
        	predicted_brake = (output_all[5][0])

        else:

        	predicted_steers = (output_all[9][0])
        	predicted_acc = (output_all[10][0])
        	predicted_brake = (output_all[11][0])

        if self._avoid_stopping:
            predicted_speed = (output_all[12][0])
            real_speed = speed 

            real_predicted = predicted_speed 
            if real_speed < 2.0 and real_predicted > 3.0:
                # If (Car Stooped) and
                #  ( It should not have stopped, use the speed prediction branch for that)

                predicted_acc = 1 * (5.6  - speed) + predicted_acc

                predicted_brake = 0.0

                predicted_acc = predicted_acc[0][0]


        return predicted_steers, predicted_acc, predicted_brake

