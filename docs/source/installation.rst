Installation
############



Linux or Windows
****************


Executable
----------

.. warning::
    Executables are highly experimental. Depending on your OS and installed libraries they might not work.
    We can not guarantee that they work for your computer setup. Thus, we recommend installing the pip package instead.

We provide executable versions of the tool for some operating systems (OS).
If you are not familiar with virtual environments and the command line you can use the RAPDOR them.
Those are provided via github. You can get the latest versions for your OS via the assets dropdown here:

`https://github.com/domonik/RAPDOR/releases <https://github.com/domonik/RAPDOR/releases>`_

Please read the instructions in the following tutorial in order to run the tool:

:ref:`Using Executables<executable-tutorial>`



Pip
---

.. note::
    Best practice is to install RAPDOR into an encapsulated environment e.g. via Conda:

    .. code-block::

        conda create -n rapdor_env
        conda activate rapdor_env
        conda install python


You can install RAPDOR via:

.. code-block::

    pip install RAPDOR

Conda
-----

Building a conda package is in planning but not done yet


macOS
*****

.. note::
    Unfortunately we don´t have the opportunity to test mac os versions of the tool.
    However all dependencies are available for mac. We thus expect that at least the pypi version can be installed just
    like in linux