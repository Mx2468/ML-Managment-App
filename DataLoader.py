# Author: Maksymilian Sekula

"""
A module of classes to load datasets into the program
TODO: Research different type of files to load from
"""
import os.path
import pandas


class DataLoader:
    """

    """
    def __init__(self):
        pass

    def load_dataset_from_file(self, filepath):
        """
        Loads a dataset from a file

        :param filepath:
        :type filepath: str
        :return: A pandas dataframe, containing the data in the file
        """
        if os.path.exists(filepath):
            if filepath.endswith(".csv"):
                return pandas.read_csv(filepath)
        else:
            return None
