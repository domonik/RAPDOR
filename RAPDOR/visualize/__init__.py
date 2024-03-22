import os
import yaml
import json


VISDIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(os.path.dirname(VISDIR), "dashConfig.yaml")
assert os.path.exists(CONFIG_FILE), "Config file does not exist"
with open(CONFIG_FILE, "r") as handle:
    CONFIG = yaml.load(handle, Loader=yaml.SafeLoader)

EXTERNEL_CONFIG_FILE = os.getenv('RAPDOR_CONFIG_FILE')
if EXTERNEL_CONFIG_FILE:
    with open(EXTERNEL_CONFIG_FILE, "r") as handle:
        EXTERNEL_CONFIG = yaml.load(handle, Loader=yaml.SafeLoader)

    CONFIG.update(EXTERNEL_CONFIG)


if CONFIG["display"]["mode"]:
    DISPLAY = True
    DISPLAY_FILE = CONFIG["display"]["file"]
    if not os.path.exists(DISPLAY_FILE):
        raise ValueError(f"Running in Display Mode but cannot find the file to display:\n Expected File {DISPLAY_FILE}")
    if CONFIG["display"]["custom_tutorial_dialog"]:
        TUTORIAL_DIALOG_FILE = CONFIG["display"]["custom_tutorial_dialog"]
    else:
        TUTORIAL_DIALOG_FILE = os.path.join(os.path.dirname(VISDIR), "visualize", "assets", "tutorialDisplayMode.json")
    if CONFIG["display"]["welcome_files"]:
        WELCOME_FILES = CONFIG["display"]["welcome_files"]
    else:
        WELCOME_FILES = [os.path.join(os.path.dirname(VISDIR), "visualize", "assets", "defaultWelcome.md")]
    DEFAULT_COLUMNS =  CONFIG["display"]["default_columns"] if CONFIG["display"]["default_columns"] else []


else:
    DISPLAY = False
    DISPLAY_FILE = None
    WELCOME_TEXT = None
    WELCOME_FILES = None
    DEFAULT_COLUMNS = ["Gene"]

    TUTORIAL_DIALOG_FILE = os.path.join(os.path.dirname(VISDIR), "visualize", "assets", "tutorial.json")

DISABLED = DISPLAY
MAX_KERNEL_SLIDER = CONFIG["kernel"]["max"]

with open(TUTORIAL_DIALOG_FILE, "r") as handle:
    TUTORIAL_DIALOG = json.load(handle)

BOOTSH5 = "col-12 justify-content-center px-0 py-2"
BOOTSROW = "row  px-4 px-md-4 py-1"