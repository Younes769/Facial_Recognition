import cv2
import os
import random
import numpy as np
from matplotlib import pyplot as plt
# Import tensorflow dependencies - Functional API
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Input, Flatten
from tensorflow.keras import layers
import tensorflow as tf

inputimage = Input(shape=(...))  
verificatioimage = Input(shape=(...)) 
output = Dense(2)(inputimage)  
model = Model(inputs=[inputimage, verificatioimage], outputs=output)