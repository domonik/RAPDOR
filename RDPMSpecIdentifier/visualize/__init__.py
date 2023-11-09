import os
mode = os.getenv('RDPMS_DISPLAY_MODE')
file = os.getenv('RDPMS_DISPLAY_FILE')

if mode == "True":
    DISPLAY = True
    if file:
        DISPLAY_FILE = file
    else:
        DISPLAY_FILE = "RDPMSpecIdentifier.json"
    if not os.path.exists(DISPLAY_FILE):
        raise ValueError(f"Running in Display Mode but cannot find the file to display:\n Expected File {DISPLAY_FILE}")
else:
    DISPLAY = False
    DISPLAY_FILE = None
DISABLED = DISPLAY


BOOTSH5 = "col-12 justify-content-center px-0"
BOOTSROW = "row  px-4 px-md-4 py-1"