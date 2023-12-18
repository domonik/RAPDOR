.. _data-prep-tutorial:

Prepare Your Data
#################

No matter whether you run the RDPMSpecIdentifier as a graphical user interface (GUI), a command line tool or within python,
you need to make sure your data is in the expected format.

GUI and CLI
+++++++++++

For the CLI and the GUI you need to have two tabular files (e.g. tsv or csv files).

Intensity Table
---------------

The first table contains an id, your Mass Spec Intensities of all fractions, conditions and replicates.
It might also include additional columns with data. For instance it can look like the example below.

.. note::
    It is highly recommended that the first column of the df contains no duplicates. It might not breakt the code but
    lead to unexpected behaviour





.. csv-table::
   :file: _static/example_table.csv
   :header-rows: 1
   :delim: tab


Design Table
------------
The second tabular file has to be a design table. This will map the intensity columns from the Intensity Table to
conditions.




.. csv-table::
   :file: _static/example_design.csv
   :header-rows: 1
   :delim: tab


.. note::
    The design table must have those column headers. Further only two level Treatments are supported by now.

    .. list-table::
       :header-rows: 1

       * - Header
         - Explanation
       * - Name
         - The Names of the Columns that contain the Intensities in the Intensity Table
       * - Treatment
         - A column containing names of the different treatments.
       * - Fraction
         - The fraction to which those intensities belong to
       * - Replicate
         - The Replicate to which those intensities belong to


If your tabular files look like expected you can use them for the GUI or CLI.
