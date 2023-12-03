import os
from .models import *
from .utils import *

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
LOGS_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "tasks", "pli", "mujoco_xml_files")
