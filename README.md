# RAPDOR

**R**apid **A**NOSIM using **P**robability **D**istance for estimati**O**n of **R**edistribution (RAPDOR) is a 
package to identify changes in distribution profiles originating from two different conditions. 

## [Getting Started](https://domonik.github.io/RAPDOR/)

For latest installation details, documentation and various tutorials please have a look at our documentation page here:

https://domonik.github.io/RAPDOR/

# Minimal documentation
## System requirements

### Hardware requirements

`RAPDOR` only requires a standard computer with sufficient RAM depending on the analyzed dataset. 

### Software requirements

#### OS

`RAPDOR` is supported for *Windows*, *macOS* and *Linux* and is constantly tested via  GitHub actions for the following
systems:

- Linux: Ubuntu 22.04

####  Dependencies

`RAPDOR` requires `Python 3` and `pip` to be installed. It is tested for Python versions `3.10` to `3.12`.

Note that there is an issue with a dependency that disallows the use of Python `3.13` at the current state


```text
statsmodels,
numpy
scipy
plotly>=5.16
pandas
dash>=2.5
dash_bootstrap_components
scikit-learn
dash_loading_spinners
kaleido
dash_daq
dash_extensions
pyYAML
dash[diskcache]
```

## Installation

## Install from PyPi

We highly reccomend to install RAPDOR into a fresh setup conda environment that only contains a supported Python 
version. This usually avoids dependency conflicts.


```shell
conda create -n rapdor_env
conda activate rapdor_env
conda install python==3.12
```

```shell
pip3 install RAPDOR
```

Installing RAPDOR usually takes less than a minute.

## Demo

### GUI

The following tutorial shows how to run the graphical user interface:

https://domonik.github.io/RAPDOR/v0.1.5/running_dash.html

The graphical user interface itself has in inbuilt demo that can be accessed via clicking the Tutorial button in the
upper right corner. 

### Python

If you want to learn how to analyze data using the Python API itself please follow:

https://domonik.github.io/RAPDOR/v0.1.5/python_analysis.html





