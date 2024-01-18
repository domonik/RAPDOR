Using Python
############

.. currentmodule:: RAPDOR.datastructures


.. note::
    It is possible to run the whole analysis in python without touching the Dash GUI.


Once you prepared your data as described in the :ref:`prepare your data<data-prep-tutorial>` section, you can
use the python API to analyze the data.
First you need to load your data in python. Note that the test data has log transformed intensities, which is specified
via the logbase parameter in the :class:`RAPDORData` construction

.. code-block:: python

    import pandas as pd
    from RAPDOR.datastructures import RAPDORData
    intensities = pd.read_csv("https://raw.githubusercontent.com/domonik/RAPDOR/main/RAPDOR/tests/testData/testFile.tsv", sep="\t")
    design = pd.read_csv("https://raw.githubusercontent.com/domonik/RAPDOR/main/RAPDOR/tests/testData/testDesign.tsv", sep="\t")
    rapdordata = RAPDORData(intensities, design, logbase=2)


Next you want to normalize the fractions and calculate samplewise distances. This is done via a single call.
Here we use the recommended Jensen-Shannon-Distance and smooth the initial intensities with an averaging kernel
of size 3.

.. code-block:: python

    rapdordata.normalize_and_get_distances(method="Jensen-Shannon-Distance", kernel=3)





Since we only have 3 replicates per condition for this dataset, it is not reasonable to calculate a p-value. Instead we
will calculate stats like the Jensen-Shannon-Distance of the means and the ANOSIM R value. We will then rank the table
based on the R value and the mean distance. This is done via:

.. code-block:: python

    rapdordata.calc_all_scores()
    rapdordata.rank_table(["ANOSIM R", "Mean Distance"], ascending=[False, False])



Finally we want to export the file as a csv table and as a JSON file. The JSON is further needed to display your data in
a Dash webserver interface. You can learn more about it in the :ref:`server setup section<server-setup>`.

.. code-block:: python

    rapdordata.export_csv(file="test.tsv", sep= "\t")
    rapdordata.to_json(file="test.json")


