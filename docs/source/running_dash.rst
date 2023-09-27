.. _running-dash-tutorial:

Running Dash Interface
######################

.. note::
    To run the Dash interface make sure you prepared your data according to :ref:`Data Prepatation<data-prep-tutorial>`.

Command Line Executables
------------------------

you can run the Dash Interface via the following command:

.. code-block:: bash

    RDPMSpecIdentifier Dash


Further flags (e.g. if you want to upload data already) can be found in the :ref:`CLI Documentation<cli-doc>`

Once you executed that command make sure to not close the terminal.
You can then open the corresponding page in a browser (e.g. Firefox, Chrome).
Per default the app will be run under the following address. However, this can be changed via flags.

    `http://127.0.0.1:8080/ <http://127.0.0.1:8080/>`_

You can upload your design and intensities files that you prepared earlier
(see :ref:`Data Prepatation<data-prep-tutorial>`) via the upload page.


DISPLAY mode
------------

You can run the app in display mode. This will disable most of the buttons and is ment for displaying pre-analyzed data.
To achieve this you need to set the environment variable :code:`RDPMS_DISPLAY_MODE=True`.

.. note::

    You need to add the pre-analyzed data when you run the dash app e.g. via the :code:`--input` flag.
    Otherwise you can setup your own server (see :ref:`Server Setup<server-setup>`)



