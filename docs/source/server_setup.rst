
.. _server-setup:

Server Setup
############

If you want to setup a server that displays your pre-analyzed data for other users, you need to
use a WSGI server e.g. gunicorn.

Here is example python code that can be used for server setup:

..  code-block:: python

    import os
    os.environ["RDPMS_DISPLAY_MODE"] = "True"
    from RDPMSpecIdentifier.visualize.appDefinition import app
    from RDPMSpecIdentifier.visualize.runApp import get_app_layout
    from RDPMSpecIdentifier.datastructures import RDPMSpecData

    input = "preAnalyzedData.json"

    with open(input) as handle:
        jsons = handle.read()
    rdpmsdata = RDPMSpecData.from_json(jsons)

    app.layout = get_app_layout(rdpmsdata)

    server = app.server


After you set up this startup code in a file called `main.py` you can run the server via:

.. note::

    Make sure to set the environment variable :code:`RDPMS_DISPLAY_MODE` before importing the app.
    Else, it wont have an effect.


.. code-block:: bash

    gunicorn -b 0.0.0.0:8080 main:server
