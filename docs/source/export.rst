Data Export
###########

.. currentmodule:: RDPMSpecIdentifier.datastructures


TSV
---

You can export analyzed data as a csv file. This will exclude the data columns containing
intensities. However, it will not store information about how your data was analyzed.

.. code-block:: python

        df = pd.read_csv("../testData/testFile.tsv", sep="\t", index_col=0)
        design = pd.read_csv("../testData/testDesign.tsv", sep="\t")
        rdpmsdata = RDPMSpecData(df, design, logbase=2)
        rdpmsdata.to_csv("path_to_file", sep="\t")


JSON
----

The JSON format will keep the current state of the object. This is the recommended format.
if you want to display your data later on a server (see :ref:`Server Setup<server-setup>`).
You can export it either via the Dash button or via the  :func:`~RDPMSpecData.to_json` function in python:
