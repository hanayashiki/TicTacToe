from keras import Sequential
from keras.layers import Input, Dense, Softmax, Activation, Lambda, Multiply, Add, Reshape
from keras.models import Model
import keras.backend as K
import tensorflow as tf

from game import Game

class ModelBase():
    INPUT_DIM = 3 * 9
    OUTPUT_DIM = 9
    model = Sequential()

    def GetModel(self) -> Model:
        return self.model

    def Predict(self, array):
        return self.model.predict_on_batch(array)

    def LoadWeights(self, path : str):
        self.model.load_weights(path)

    def SaveWeights(self, path : str):
        self.model.save_weights(path)




class ModelDenseSigmoidMasked(ModelBase):
    def __init__(self, inputDim=ModelBase.INPUT_DIM, outputDim=ModelBase.OUTPUT_DIM):
        super(ModelDenseSigmoidMasked, self).__init__()
        # Input is the representation of the current game according to Game
        # Output is supposed to be the winning rate

        def Mask(x):
            empties = tf.gather(x, indices=[ 1 + 3 * i for i in range(Game.BOARD_SIZE) ] ,axis=1)
            return empties  # reversed empties, to filter out non-empty choices

        inputs = Input(shape=(inputDim,))
        dense = Dense(units=outputDim, input_dim=inputDim, name="dense")(inputs)
        act = Activation(name="sigmoid", activation="sigmoid")(dense)
        mask = Lambda(Mask, name="mask", output_shape=(self.OUTPUT_DIM, ))(inputs)
        masked = Multiply(name="merger")([mask, act])

        self.model = Model(inputs=inputs, outputs=masked)

        print("Use model: ")
        self.model.summary()
        self.model.compile(loss='mean_squared_error', optimizer='sgd')

        pass


class ModelTwoDensesSigmoidMasked(ModelBase):
    def __init__(self, inputDim=ModelBase.INPUT_DIM, outputDim=ModelBase.OUTPUT_DIM):
        super(ModelTwoDensesSigmoidMasked, self).__init__()
        # Input is the representation of the current game according to Game
        # Output is supposed to be the winning rate

        def Mask(x):
            empties = tf.gather(x, indices=[ 1 + 3 * i for i in range(Game.BOARD_SIZE) ] ,axis=1)
            return empties  # reversed empties, to filter out non-empty choices

        inputs = Input(shape=(inputDim,))
        dense1 = Dense(units=12, input_dim=inputDim, name="dense1")(inputs)
        # Forgot to add non-linearity to dense1
        dense2 = Dense(units=outputDim, name="dense2")(dense1)
        act = Activation(name="sigmoid", activation="sigmoid")(dense2)
        mask = Lambda(Mask, name="mask", output_shape=(self.OUTPUT_DIM, ))(inputs)
        masked = Multiply(name="merger")([mask, act])

        self.model = Model(inputs=inputs, outputs=masked)

        print("Use model: ")
        self.model.summary()
        self.model.compile(loss='mean_squared_error', optimizer='sgd')

class ModelTwoDensesReluSigmoidMasked(ModelBase):
    def __init__(self, inputDim=ModelBase.INPUT_DIM, midDim=12, outputDim=ModelBase.OUTPUT_DIM):
        super(ModelTwoDensesReluSigmoidMasked, self).__init__()
        # Input is the representation of the current game according to Game
        # Output is supposed to be the winning rate

        def Mask(x):
            empties = tf.gather(x, indices=[ 1 + 3 * i for i in range(Game.BOARD_SIZE) ] ,axis=1)
            return empties  # reversed empties, to filter out non-empty choices

        inputs = Input(shape=(inputDim,))
        dense1 = Dense(units=midDim, input_dim=inputDim, name="dense1", activation="relu")(inputs)
        dense2 = Dense(units=outputDim, name="dense2")(dense1)
        act = Activation(name="sigmoid", activation="sigmoid")(dense2)
        mask = Lambda(Mask, name="mask", output_shape=(self.OUTPUT_DIM, ))(inputs)
        masked = Multiply(name="merger")([mask, act])

        self.model = Model(inputs=inputs, outputs=masked)

        print("Use model: ")
        self.model.summary()
        self.model.compile(loss='categorical_crossentropy', optimizer='sgd')


class ModelThreeDensesReluSigmoidMasked(ModelBase):
    def __init__(self, inputDim=ModelBase.INPUT_DIM, midDim1=12, midDim2=12, outputDim=ModelBase.OUTPUT_DIM, loss='categorical_crossentropy'):
        super(ModelThreeDensesReluSigmoidMasked, self).__init__()
        # Input is the representation of the current game according to Game
        # Output is supposed to be the winning rate

        def Mask(x):
            empties = tf.gather(x, indices=[ 1 + 3 * i for i in range(Game.BOARD_SIZE) ] ,axis=1)
            return empties  # reversed empties, to filter out non-empty choices

        inputs = Input(shape=(inputDim,))
        dense1 = Dense(units=midDim1, input_dim=inputDim, name="dense1", activation="relu")(inputs)
        dense2 = Dense(units=midDim2, input_dim=inputDim, name="dense2", activation="relu")(dense1)
        dense3 = Dense(units=outputDim, name="dense3")(dense2)
        act = Activation(name="sigmoid", activation="sigmoid")(dense3)
        mask = Lambda(Mask, name="mask", output_shape=(self.OUTPUT_DIM, ))(inputs)
        masked = Multiply(name="merger")([mask, act])

        self.model = Model(inputs=inputs, outputs=masked)

        print("Use model: ")
        self.model.summary()
        self.model.compile(loss=loss, optimizer='sgd')

class ModelThreeDensesReluReluAdamMasked(ModelBase):
    def __init__(self, inputDim=ModelBase.INPUT_DIM, midDim1=12, midDim2=12, outputDim=ModelBase.OUTPUT_DIM, loss='categorical_crossentropy'):
        super(ModelThreeDensesReluReluAdamMasked, self).__init__()
        # Input is the representation of the current game according to Game
        # Output is supposed to be the winning rate

        def Mask(x):
            empties = tf.gather(x, indices=[ 1 + 3 * i for i in range(Game.BOARD_SIZE) ], axis=1)
            return empties  # reversed empties, to filter out non-empty choices
        def NoneZero(x):
            empties = 0.001 * Mask(x)
            return empties  # reversed empties, to filter out non-empty choices

        inputs = Input(shape=(inputDim,))
        dense1 = Dense(units=midDim1, input_dim=inputDim, name="dense1", activation="relu")(inputs)
        dense2 = Dense(units=midDim2, input_dim=inputDim, name="dense2", activation="relu")(dense1)
        dense3 = Dense(units=outputDim, name="dense3")(dense2)
        act = Activation(name="relu", activation="relu")(dense3)
        mask = Lambda(Mask, name="mask", output_shape=(self.OUTPUT_DIM, ))(inputs)
        noneZero = Lambda(NoneZero, name="nonezero", output_shape=(self.OUTPUT_DIM,))(inputs)
        masked = Multiply(name="merger")([mask, act])
        masked = Add(name="adder")([masked, noneZero])

        self.model = Model(inputs=inputs, outputs=masked)

        print("Use model: ")
        self.model.summary()
        self.model.compile(loss=loss, optimizer='adagrad')