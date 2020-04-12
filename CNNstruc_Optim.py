from StrucOptiPeri import *
import plaidml.keras
plaidml.keras.install_backend()
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time

X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))

X = X/255

for Conv_size in Conv_sizes:
    for Conv_layer in Conv_layers:
        for Dense_size in Dense_sizes:
            for Dense_layer in Dense_layers:
                # INIT
                NAME = "C_vs_D_CNN_Conv_size_layer{},{}_Dense_size_layer{},{}_time--{}".format(
                    Conv_size, Conv_layer, Dense_size, Dense_layer, int(time.time()))

                tensorboard = TensorBoard(log_dir='logs/{name}'.format(name=NAME))

                # START MODEL

                model = Sequential()
                print("Conv size is {}, layer is {};\n".format(Conv_size, Conv_layer))
                print("Dense size is {}, layer is {}.\n".format(Dense_size, Dense_layer))

                model.add(Conv2D(Conv_size, (3, 3), input_shape=X.shape[1:]))
                model.add(Activation("relu"))
                model.add(MaxPooling2D(pool_size=(2, 2)))

                for l in range(Conv_layer-1):
                    model.add(Conv2D(Conv_size, (3, 3)))
                    model.add(Activation("relu"))
                    model.add(MaxPooling2D(pool_size=(2, 2)))
                    model.add(Dropout(0.2))

                model.add(Flatten())

                for l in range(Dense_layer):
                    model.add(Dense(Dense_size))
                    model.add(Activation("relu"))

                model.add(Dense(1))
                model.add(Activation('sigmoid'))

                model.compile(loss="binary_crossentropy",
                              optimizer="adam",
                              metrics=["accuracy"])

                model.fit(X, y, batch_size=32, epochs=5, validation_split=0.3, callbacks=[tensorboard])


