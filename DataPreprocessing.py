# Author: Maksymilian Sekula
"""
TODO: List how datasets need to be pre-processed for each ML model"""

import sklearn
import pandas
import abc


class DataPreProcessor:
    """
    A class to pre-process datasets, currently only uses a Pandas dataframe
    :param df: A pandas dataframe
    """
    __dataFrame = pandas.DataFrame
    __x = object
    __y = object
    __xTraining = object
    __xTesting = object
    __yTraining = object
    __yTesting = object

    def __init__(self, df):
        """
        Constructor for the data
        :argument df:
        """
        self.__dataFrame = df

    def deleteColumn(self, columnName):
        """
        Delete the column of the dataframe, given its name.
        :param: columnName
        :type columnName: str
        """
        del self.__dataFrame[columnName]

    def removeAllIncompleteInformation(self):
        """ Delete all records in the data frame that contain empty fields. """
        self.__dataFrame.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)

    def oneHotEncoding(self, categories):
        """
        :param categories: A list containing strings
        :type categories: list-like
        :return: A pandasdataframe with the categories specified in the list one-hot encoded
        """
        self.__dataFrame = pandas.get_dummies(self.__dataFrame, columns=categories)

    def standardScaling(self, features):
        """
        :param features:
        :return:
        """
        sklearn.preprocessing.StandardScaler().fit_transform(x=features)

    def splitData(self, testFraction):
        """


        :param testFraction:
        """
        self.__xTraining, self.__xTesting, self.__yTraining, self.__yTesting = sklearn.model_selection.train_test_split(self.__x, self.__y, test_size = testFraction, shuffle = True)