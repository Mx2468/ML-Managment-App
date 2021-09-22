"""
A module representing different machine learning models

Author: Maksymilian Sekula
Version: 1.2
Date: 1/5/2021

TODO:
    - Research different models & layers
    - Make classes for some scikit learn models
    -
"""

import tensorflow as tf
import abc


class ANNKerasModel(abc.ABCMeta):
    """
    An abstract class representing a keras tensorflow model
    Currently uses the functional API of creating models with tf.keras modules (instead of standalone keras)
    Currently limited to only dense layers for hidden and output layers
    """
    __model = tf.keras.Model()
    __last_layer = object
    __input = object
    __output = object

    def __init__(self):
        pass

    def create_model(self, layer_info, optimiser, loss_function, other_losses=None):
        """
        Creates a model automatically from data about the model
        :param layer_info: A 2D array with the sequence of attributes for the layers in the following order:
            1. number of neurons
            2. activation function (ignored for the input layer)
        :param optimiser: The optimiser function to be used in the model
        :param loss_function: The loss function used by the model
        :param other_losses: Optional array of other loss functions to be used by the model - Default is None
        """

        # Input layer
        input_layer_data = layer_info[0]
        self.add_input_layer(input_layer_data[0])

        # Hidden layers
        for i in range(1, len(layer_info) - 1):
            layer_data = layer_info[i]
            self.add_dense_layer(layer_data[0], layer_data[1])

        # Output Layer
        output_layer_data = layer_info[len(layer_info) - 1]
        self.add_output_layer(output_layer_data[0], output_layer_data[1])

        self.compile_model(optimizer_algorithm=optimiser, loss_metric=loss_function, array_of_metrics=other_losses)

        return self.__model

    def create_wrapped_model(self, layer_info, optimiser, loss_function, other_losses=None, iteration_info=None):
        """
        Creates a model instance wrapped for sklearn
        :param layer_info: A 2D array with the sequence of attributes for the layers in the following order:
            1. number of neurons
            2. activation function (ignored for the input layer)
        :param optimiser: The optimiser function to be used in the model
        :param loss_function: The loss function used by the model
        :param other_losses: Optional array of other loss functions to be used by the model - Default is None
        :param iteration_info: List containing the number of epochs and batch size for the model -
        needed for the sklearn wrapper
        :return: The compiled and wrapped model object
        """
        self.create_model(layer_info, optimiser, loss_function, other_losses)

        # Resolves info for iteration if it is provided and wraps the model
        if iteration_info is not None:
            epochs, batch_size = ANNKerasModel.__parse_iteration_info(iteration_info)
            self.wrap_model_sklearn(epochs, batch_size)

        return self.__model

    @staticmethod
    def __parse_iteration_info(iteration_info):
        """
        A static method that parses an array of iteration info

        :param iteration_info: An array of integers with two elements:
            1. Epochs
            2. Batch Size
        :return: The epochs followed by the batch size, in that order.
        """
        if iteration_info is None:
            print("There are no elements in the argument - using default values")
            epochs = 1
            batch_size = 1
        else:
            try:
                epochs = int(iteration_info[0])
                batch_size = int(iteration_info[1])
            except:
                print("The format of the iteration_info argument was incorrect - using default values")
                epochs = 1
                batch_size = 1

        return epochs, batch_size

    def add_input_layer(self, input_shape):
        """
        Adds an initial layer to the model

        :param input_shape: The input shape of the ANN model (the number of columns)
        :type input_shape: int
        """
        self.__input = tf.keras.Input(shape=(input_shape,))
        self.__last_layer = self.__input

    def add_dense_layer(self, number_of_neurons, activation_function):
        """
        Incrementally adds a dense keras layer to the model

        :param number_of_neurons: The number of neurons in the layer
        :param activation_function: The activation function to be used for the layer
        :type number_of_neurons: int
        :type activation_function: function
        """

        temp_layer = tf.keras.layers.Dense(number_of_neurons, activation=activation_function)
        self.__last_layer = temp_layer(self.__last_layer)

    @abc.abstractmethod
    def add_output_layer(self, numberOfNeurons, activationFunction):
        """
        Adds a dense layer as an output to the model

        :param numberOfNeurons: The number of neurons in the output layer
        :param activationFunction: The activation function used in the final layer (recommended to use linear, but sometimes needs to be customised)
        :type numberOfNeurons: int
        :type activationFunction: function
        """
        pass

    def compile_model(self, optimizer_algorithm, loss_metric, array_of_metrics):
        """
        Compiles the layers into a model, see: https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile

        :param optimizer_algorithm: Optimisation algorithm to be used by the model
        :param loss_metric: The metric the model will use to base optimisation around
        :param array_of_metrics: List of metrics to be evaluated by the model during testing and training
        """
        self.__model = tf.keras.Model(inputs=self.__input, outputs=self.__output)
        self.__model.compile(optimizer=optimizer_algorithm, loss=loss_metric, metrics=array_of_metrics)

    @abc.abstractmethod
    def wrap_model_sklearn(self, epochs, batch_size):
        """
        Wraps the model in a class for handling with sklearn methods
        :param epochs: The number of iterations that the model will run for over the dataset
        :param batch_size: The amount of tensors that will be processed by the model at a time during training
        """
        pass

    def get_last_layer(self):
        """
        :return: The last layer created in the model
        """
        return self.__last_layer

    def set_last_layer(self, last_layer):
        """"""
        self.__last_layer = last_layer

    def set_output_layer(self, new_output_layer):
        """"""
        self.__last_layer = new_output_layer
        self.__output = new_output_layer

    def get_model(self):
        """Returns the ANN model made in the class"""
        return self.__model


class RegressionANNKerasModel(ANNKerasModel, tf.keras.wrappers.scikit_learn.KerasRegressor):
    """A class representing a keras artificial neural network model used for regression"""

    def __init__(self):
        super().__init__()

    def add_output_layer(self, number_of_neurons=1, activation_function=tf.keras.activations.linear):
        tempLayer = tf.keras.layers.Dense(number_of_neurons, activation=activation_function)
        self.set_output_layer(tempLayer(self.get_last_layer()))

    def wrap_model_sklearn(self, epochs, batch_size):
        return tf.keras.wrappers.scikit_learn.KerasRegressor(build_fn=self.get_model(), epochs=epochs,
                                                             batch_size=batch_size)


class ClassificationANNKerasModel(ANNKerasModel):
    """A class representing a keras artificial neural network model used for classification"""

    def __init__(self):
        super().__init__()

    def add_output_layer(self, numberOfNeurons=1, activationFunction=tf.keras.activations.softmax):
        tempLayer = tf.keras.layers.Dense(numberOfNeurons, activation=activationFunction)
        self.set_output_layer(tempLayer(self.get_last_layer()))

    def wrap_model_sklearn(self, epochs, batch_size):
        tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=self.get_model(), epochs=epochs, batch_size=batch_size)

#Test Code

aModel = RegressionANNKerasModel()
aModel.add_input_layer(10)
aModel.add_dense_layer(10, tf.keras.activations.linear)
aModel.add_output_layer(1, tf.keras.activations.linear)
aModel.compile_model(tf.optimizers.Adamax(learning_rate=0.0001), tf.keras.losses.MeanSquaredError(), [tf.keras.losses.MeanAbsoluteError(), tf.keras.losses.MeanAbsolutePercentageError()])
aModel.get_model().save("TestModel1")
model = tf.keras.models.load_model("TestModel1", compile=False)
model.summary()