import os
mode = os.getenv('RDPMS_DISPLAY_MODE')
if mode == "True":
    DISPLAY = True
else:
    DISPLAY = False
DISABLED = DISPLAY