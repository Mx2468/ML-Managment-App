"""
Implements a tkinter GUI front end to navigate the program with

Application extends Tk class - prevents needed to instantiate this in the main method and makes it easier to switch frames

Each class representing a "page" or "window" in the application should extend the tkinter Frame class

"""
import tkinter as tk
from tkinter import filedialog
import DataLoader


class GUIApp(tk.Tk):
    """
    A class to represent the top-level application
    Extends tkinter Tk class to avoid instantiating objects - only this object is instantiated to start the app
    """
    def __init__(self):
        super().__init__()

        # Size of window set to half the dimensions of the screen resolution
        resolution = str(super().winfo_screenwidth()//2)+"x"+str(super().winfo_screenheight()//2)
        super().geometry(resolution)

        self._frame = tk.Frame(self)
        self.switch_frame(MainPage)

    def switch_frame(self, new_frame):
        """
        Switches the current frame out for another frame.

        :param new_frame: The class type of the frame to switch to
        :type new_frame: type
        """
        self._frame.destroy()
        self._frame = new_frame(self)
        self._frame.pack()

    def get_window(self):
        """
        :return: The current window object
        """
        return self.__window


class MainPage(tk.Frame):
    """
    A class to represent the main page that the user first sees when opening the application
    """
    def __init__(self, containerWindow):
        super().__init__(containerWindow)

        welcome_label = tk.Label(self, text="Welcome to my application!").pack(side=tk.TOP, pady=10)

        load_data_button = tk.Button(self, text="Load Data", command=lambda: containerWindow.switch_frame(DatasetPage))\
            .pack(side=tk.TOP, pady=10)

        tk.Button(self, text="Help").pack(after=load_data_button, side=tk.TOP, pady=10)


class DatasetPage(tk.Frame):
    """
    A page to load a dataset
    """
    def __init__(self, containerWindow):
        """
        Constructor for the page

        :param containerWindow: The top-level window (tk.Tk instance)
        :type containerWindow: tk.Tk
        """
        super().__init__(containerWindow)

        filepath = tk.filedialog.askopenfilename(initialdir="/", title="Select a Dataset", filetypes=[("CSV files", "*.csv*")])

        if filepath == "":
            containerWindow.switch_frame(FileErrorPage)
        else:
            data_loader = DataLoader.DataLoader()
            dataset = data_loader.load_dataset_from_file(filepath)
            file_label = tk.Label(self, text=filepath).pack()
            tk.Label(self, text=dataset.head(100)).pack(after=file_label)

class FileErrorPage(tk.Frame):
    """
    A page to inform the user that the dataset selected was incorrect
    """
    def __init__(self, containerWindow):
        super().__init__(containerWindow)
        tk.Label(self, text="The file selected was incorrect, please select a csv file")\
            .pack(side=tk.TOP, pady=10)
        tk.Button(self, text="Try Again", command=lambda: containerWindow.switch_frame(DatasetPage))\
            .pack(side=tk.TOP, pady=10)



class MakeModelPage(tk.Frame):
    """
    A page on which a user can start to make a model
    """
    def __init__(self, containerWindow):
        super().__init__(containerWindow)
        title_label = tk.Label(self, text="Make new model").pack()
        tk.Button(self, text="Tensorflow ANN Model", command=lambda: containerWindow.switch_frame())\
            .pack(after=title_label, side=tk.BOTTOM, pady=10)



class LoadModelPage(tk.Frame):
    """
    A page on which a user can make a model.
    """
    def __init__(self, containerWindow):
        super().__init__(containerWindow)


gui = GUIApp()
gui.mainloop()
