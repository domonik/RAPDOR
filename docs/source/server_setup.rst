
.. _server-setup:

Server Setup
############

If you want to setup a server that displays your pre-analyzed data for other users, you need to
use a WSGI server e.g. gunicorn.

Here is example python code that can be used for server setup:

..  code-block:: python

    import os
    os.environ["RAPDOR_CONFIG_FILE"] = "customConfig.yaml"
    from RAPDOR.visualize.appDefinition import app
    from RAPDOR.visualize.runApp import get_app_layout
    from RAPDOR.datastructures import RAPDORData

    app.layout = get_app_layout()

    server = app.server


After you set up this startup code in a file called `main.py` you can run the server via:

.. note::

    If you want to use custom settings, make sure to set the environment variable :code:`RAPDOR_CONFIG_FILE` before importing the app.
    Else, it wont have an effect.


.. code-block:: bash

    gunicorn -b 0.0.0.0:8080 main:server


Custom Settings:
----------------

You can change the default settings via a custom config file as specified above. Below you find the default yaml file
that is loaded from the RAPDOR module. Fields are overwritte via using a custom file and setting the global
variable :code:`RAPDOR_CONFIG_FILE`.

.. literalinclude:: ../../RAPDOR/dashConfig.yaml
  :language: YAML