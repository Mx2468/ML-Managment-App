import sklearn
import pandas
import keras
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from tensorflow.python.client import device_lib
import MLModel

device_lib.list_local_devices()

class Pipeline:
    __pipeline = object
    __x = object
    __y = object
    __xTraining = object
    __xTesting = object
    __yTraining = object
    __yTesting = object
    __pipelineArray = []
    __dataFrame = pandas.DataFrame

    def __init__(self):
        """ """
        self.__dataFrame = pandas.read_csv("A:\ML datasets\Melbourne Housing Market\Melbourne_housing_FULL.csv")

    def deleteColumn(self, columnName):
        """Deletes the column of the dataframe, given its name.
        :argument columnName:
        """
        del self.__dataFrame[columnName]

    def removeAllIncompleteInformation(self):
        """Delete all records in the data frame that contain empty fields"""
        self.__dataFrame.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)

    def setIndependentVariable(self, columnName):
        """ """
        self.__y = self.__dataFrame[columnName]
        self.__x = self.__dataFrame
        del self.__x[columnName]

    def splitDataSet(self, percentTrain, shuffle):
        """ """
        self.__xTraining, self.__xTesting, self.__yTraining, self.__yTesting = sklearn.model_selection.train_test_split(self.__x, self.__y, train_size=percentTrain, shuffle=shuffle)

    def oneHotEncoding(self, columnsToEncode):
        """ """
        self.__dataFrame = pandas.get_dummies(self.__dataFrame, columns=columnsToEncode)

    def addEstimator(self, estimatorObject):
        """ """
        self.__pipelineArray.append(('Estimator1', estimatorObject))

    def scaleData(self):
        """ """
        self.__pipelineArray.append(('Scaler', sklearn.preprocessing.StandardScaler()))

    def makePipeline(self):
        """ """
        self.__pipeline = sklearn.pipeline.Pipeline(steps=self.__pipelineArray)

    def fitToDataset(self):
        """ """
        self.__pipeline.fit(self.__xTraining, self.__yTraining)

    def crossValidation(self):
        cross = sklearn.model_selection.RepeatedStratifiedKFold(n_splits=5, n_repeats=1)
        sklearn.model_selection.cross_val_score(estimator=self.__pipeline, X=self.__x, y=self.__y, cv=cross)

    def getPipeline(self):
        return self.__pipeline

    def getDataSplits(self):
        return self.__xTraining, self.__yTraining, self.__xTesting, self.__yTesting


"""Test"""

myPipeline = Pipeline()

myPipeline.removeAllIncompleteInformation()

myPipeline.deleteColumn('Address')
myPipeline.deleteColumn('Date')

myPipeline.oneHotEncoding(['Suburb', 'CouncilArea', 'Type', 'Method', 'SellerG', 'Regionname'])

myPipeline.scaleData()

myPipeline.setIndependentVariable('Price')

myPipeline.splitDataSet(0.7, True)

"""
model1 = MLModel.RegressionANNKerasModel()

# Problem with model(?) - Stratified k fold claims to have only 1 members in y under this model, but works fine in the one further below
model1.addInputLayer(626)
model1.addDenseLayer(626, tf.nn.swish)
model1.addOutputLayer()
optimizer = tf.keras.optimizers.Adamax(learning_rate=0.0001)
wrappedModel = model1.compileModel(optimizerAlgorithm=optimizer, lossMetric="mean_squared_error",arrayOfMetrics=['mae', 'mape'], epochs=100, batch_size=500)




def aModel():
    model = tf.keras.Sequential([
        tf.keras.layers.Input((626,)),
        tf.keras.layers.Dense(units=626, activation=tf.nn.swish),
        tf.keras.layers.Dense(units=626, activation=tf.nn.swish),
        tf.keras.layers.Dense(units=626, activation=tf.nn.swish),
        tf.keras.layers.Dense(units=1, activation=tf.nn.leaky_relu)
      ])
    optimizer = tf.keras.optimizers.Adamax(learning_rate=0.0001)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae', 'mape'])
    return model


wrappedModel = keras.wrappers.scikit_learn.KerasRegressor(build_fn=aModel, epochs=1000, batch_size=7110)


myPipeline.addEstimator(wrappedModel)
myPipeline.makePipeline()
myPipeline.crossValidation()
"""


"""
model1 = RegressionANNKerasModel()
model1.add_input_layer(806)
model1.add_dense_layer(806*2, tf.keras.activations.relu)
model1.add_dense_layer(800, tf.keras.activations.relu)
model1.add_dense_layer(400, tf.keras.activations.relu)
model1.add_dense_layer(200, tf.keras.activations.relu)
model1.add_dense_layer(50, tf.keras.activations.relu)
model1.add_output_layer()
model1.compile_model(tf.keras.optimizers.Adamax(learning_rate=0.0001), loss_metric='mse', array_of_metrics=['mae','mape'])

# Reads in CSV file as a pandas dataframe
df = pandas.read_csv("A:\ML datasets\Melbourne Housing Market\Melbourne_housing_FULL.csv")

df.drop(labels=['Address', 'Date'], axis=1, inplace=True)

missingValueIndex = {'CouncilArea': 'Undefined', 'Regionname': 'Undefined', 'SellerG': 'Unknown', 'Method': 'Unknown', 'Type': 'Unknown'}
df.fillna(value=missingValueIndex, axis=0, inplace=True)

df.interpolate(method='polynomial', order=7, axis=0, inplace=True)

df = pandas.get_dummies(df, columns=['Suburb', 'CouncilArea', 'Type', 'Method', 'SellerG', 'Regionname'])

df.dropna(inplace=True)

labels = df['Price']

del df['Price']

features = df

features_scaled = StandardScaler().fit_transform(features, labels)

x_train, x_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.2, shuffle=True)

model1.get_model().fit(x=x_train, y=y_train.values, batch_size=1024, epochs=100, verbose=1)


#print(model1.get_model().summary())

#model1.get_model().save("TestModel1")

#model1 = tf.keras.models.load_model("TestModel1")
"""