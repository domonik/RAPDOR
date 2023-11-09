import os
mode = os.getenv('RDPMS_DISPLAY_MODE')
if mode == "True":
    DISPLAY = True
else:
    DISPLAY = False
DISABLED = DISPLAY


BOOTSH5 = "col-12 justify-content-center px-0"
BOOTSROW = "row  px-4 px-md-4 py-1"