import os
import sys

def resource_path(relative_path):
    """
    Return the absolute path to the resource.

    Works both in development and in a PyInstaller bundle.
    """
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)