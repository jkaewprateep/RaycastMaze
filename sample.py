"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
PY GAME

https://pygame-learning-environment.readthedocs.io/en/latest/user/games/raycastmaze.html

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import os
from os.path import exists

import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import ple
from ple import PLE
# from ple.games.raycast import RayCastPlayer
from ple.games.raycastmaze import RaycastMaze
from pygame.constants import K_w, K_a, K_d, K_s, K_h

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
None
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
print(physical_devices)
print(config)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Environment
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
game_console = RaycastMaze(init_pos=(1, 1), resolution=1, move_speed=20, turn_speed=13, map_size=10, height=512, width=512)
p = PLE(game_console, fps=30, display_screen=True, reward_values={})
p.init()

obs = p.getScreenRGB()	# (512, 512, 3)
print( tf.constant( obs ).shape )

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Variables
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
actions = { "none_1": K_h, "left_1": K_a, "down_1": K_s, "right1": K_d, "up___1": K_w }

nb_frames = 100000000000

global lives
global reward
global steps
global gamescores

action = 0	
steps = 0
lives = 0
reward = 0
gamescores = 0

learning_rate = 0.00001
momentum = 0.1
batch_size=10

fig = plt.figure()
image = plt.imread( "F:\\Pictures\\Cats\\samples\\03.png" )
im = plt.imshow( image )

################ Mixed of data input  ###############
global DATA
DATA = tf.zeros([ 1, 1, 42, 42, 3 ], dtype=tf.float32)
global LABEL
LABEL = tf.zeros([1, 1, 1, 1], dtype=tf.float32)

for i in range(15):
	DATA_row = -9999 * tf.ones([ 1, 1, 42, 42, 3 ], dtype=tf.float32)		
	DATA = tf.experimental.numpy.vstack([DATA, DATA_row])
	LABEL = tf.experimental.numpy.vstack([LABEL, tf.constant(0, shape=(1, 1, 1, 1))])
	
for i in range(15):
	DATA_row = 9999 * tf.ones([ 1, 1, 42, 42, 3 ], dtype=tf.float32)			
	DATA = tf.experimental.numpy.vstack([DATA, DATA_row])
	LABEL = tf.experimental.numpy.vstack([LABEL, tf.constant(9, shape=(1, 1, 1, 1))])	
	
DATA = DATA[-30:,:,:,:]
LABEL = LABEL[-30:,:,:,:]
####################################################

checkpoint_path = "F:\\models\\checkpoint\\" + os.path.basename(__file__).split('.')[0] + "\\TF_DataSets_01.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)

if not exists(checkpoint_dir) : 
	os.mkdir(checkpoint_dir)
	print("Create directory: " + checkpoint_dir)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Class / Functions
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def random_action( ): 
	
	temp = tf.random.normal([1, 5], 0.2, 0.8, tf.float32)
	temp = tf.math.multiply(temp, tf.constant([ 0.0000099, 99999999, 0.0000099, 99999999, 0.0000099 ], shape=(5, 1), dtype=tf.float32))
	temp = tf.nn.softmax(temp[0])
	action = int(tf.math.argmax(temp))

	return action
	
def animate( i ):

	global DATA
	global LABEL

	action = random_action( );
	reward = p.act(list(actions.values())[action])
	obs = p.getScreenRGB()
	
	compressed_image = image_preprocessing( obs );
	DATA, LABEL = update_DATA( action, compressed_image )
	
	action = predict_action( );
	print( 'action: ' + str( list(actions.keys())[action] ) )
	
	image = tf.keras.utils.array_to_img( compressed_image )
	im.set_array( image )
	return im,

def image_preprocessing( observation ):
    
	original_image = tf.image.resize(observation, [42, 42])
	original_image = tf.image.rot90( original_image, k=3, name=None )
	original_image = tf.image.flip_left_right( original_image )
	# image = tf.image.rgb_to_grayscale( tf.cast( tf.keras.utils.img_to_array( original_image ), dtype=tf.float32 ) )
	# image = tf.expand_dims( image, axis=0 )

	# image = tf.keras.layers.Normalization(mean=3., variance=2.)(image)
	# image = tf.keras.layers.Normalization(mean=4., variance=6.)(image)
	# image = tf.squeeze( image )

	
	# image = tf.keras.utils.array_to_img( original_image * 255 )
	# image = tf.keras.utils.img_to_array( image )
	# image = tf.cast( original_image, dtype=tf.int32 )
	image = tf.squeeze( original_image )
	# image = tf.expand_dims( image, axis=2 )
	# image = tf.keras.utils.array_to_img( image )
	
	return image

def update_DATA( action, observation ):

	global DATA
	global LABEL
	
	# contrl = 0
	# coff_0 = 0
	# coff_1 = 0
	# coff_2 = 0
	# coff_3 = 0
	# coff_4 = 0
	# coff_5 = 0
	# coff_6 = 0
	# coff_7 = 0
	# coff_8 = 0
	# coff_9 = 0
	# coff_10 = 0
	# coff_11 = 0
	# coff_12 = 0
	# coff_13 = 0
	# coff_14 = 0
	
	# temp = tf.zeros([ 1, 1, 1, ( 42 * 42 * 3 ) ])
	# DATA_row = tf.constant([ contrl, coff_0, coff_1, coff_2, coff_3, coff_4, coff_5, coff_6, coff_7, coff_8, coff_9, coff_10, coff_11, coff_12, coff_13, coff_14 ], shape=(1, 1, 1, 16), dtype=tf.float32)
	# DATA_row = tf.concat([DATA_row, temp], 0)
	
	DATA_row = tf.constant( observation, shape=( 1, 1, 42, 42, 3 ) )
	DATA = tf.experimental.numpy.vstack([DATA, DATA_row])
	DATA = DATA[-30:,:,:,:]
	
	LABEL = tf.experimental.numpy.vstack([LABEL, tf.constant(action, shape=(1, 1, 1, 1))])
	LABEL = LABEL[-30:,:,:,:]
	
	DATA = DATA[-30:,:,:,:]
	LABEL = LABEL[-30:,:,:,:]

	return DATA, LABEL

def predict_action( ):
	global DATA
	
	temp = DATA[0,:,:,:,:]
	# print( temp.shape )

	predictions = model.predict(tf.expand_dims(tf.squeeze(temp), axis=0 ))
	score = tf.nn.softmax(predictions[0])

	return int(tf.math.argmax(score))

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Callback
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
class custom_callback(tf.keras.callbacks.Callback):

	def __init__(self, patience=0):
		self.best_weights = None
		self.best = 999999999999999
		self.patience = patience
	
	def on_train_begin(self, logs={}):
		self.best = 999999999999999
		self.wait = 0
		self.stopped_epoch = 0

	def on_epoch_end(self, epoch, logs={}):
		if(logs['accuracy'] == None) : 
			pass
		
		if logs['loss'] < self.best :
			self.best = logs['loss']
			self.wait = 0
			self.best_weights = self.model.get_weights()
		else :
			self.wait += 1
			if self.wait >= self.patience:
				self.stopped_epoch = epoch
				self.model.stop_training = True
				print("Restoring model weights from the end of the best epoch.")
				self.model.set_weights(self.best_weights)
		
		# if logs['loss'] <= 0.2 and self.wait > self.patience :
		if self.wait > self.patience :
			self.model.stop_training = True

custom_callback = custom_callback(patience=8)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: DataSet
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
print( DATA.shape )
dataset = tf.data.Dataset.from_tensor_slices((DATA, LABEL))

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Model Initialize
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
input_shape = (42, 42, 3)

model = tf.keras.models.Sequential([
	tf.keras.layers.InputLayer(input_shape=input_shape),
	
	tf.keras.layers.Reshape((1, 42 * 42 * 3)),
	tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True, return_state=False)),
	tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))

])
		
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(192))
model.add(tf.keras.layers.Dense(5))
model.summary()

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Optimizer
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
optimizer = tf.keras.optimizers.SGD(
    learning_rate=learning_rate,
    momentum=momentum,
    nesterov=False,
    name='SGD',
)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Loss Fn
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""								
lossfn = tf.keras.losses.MeanSquaredLogarithmicError(reduction=tf.keras.losses.Reduction.AUTO, name='mean_squared_logarithmic_error')

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Model Summary
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
model.compile(optimizer=optimizer, loss=lossfn, metrics=['accuracy'])

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: FileWriter
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
if exists(checkpoint_path) :
	model.load_weights(checkpoint_path)
	print("model load: " + checkpoint_path)
	input("Press Any Key!")

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Training
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
history = model.fit(dataset, epochs=1, callbacks=[custom_callback])
model.save_weights(checkpoint_path)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Tasks
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
for i in range(nb_frames):

	ani = animation.FuncAnimation(fig, animate, interval=50, blit=True)
	plt.show()
