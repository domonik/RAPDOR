from setuptools import setup, find_packages, Extension
import versioneer


NAME = "RAPDOR"
DESCRIPTION = "Package for identification of RNA dependent Proteins from mass spec data "

LONGDESC = DESCRIPTION #Todo: write Readme




cmds = versioneer.get_cmdclass()
setup(
    name=NAME,
    version=versioneer.get_version(),
    cmdclass=cmds,
    author="domonik",
    author_email="dominik.rabsch@gmail.com",
    packages=find_packages(),
    package_dir={"RAPDOR": "./RAPDOR"},
    license="LICENSE",
    url="https://github.com/domonik/RAPDOR",
    description=DESCRIPTION,
    long_description=LONGDESC,
    long_description_content_type="text/markdown",
    include_package_data=True,
    package_data={
        "RAPDOR.visualize": ["assets/*"],
        "RAPDOR": ["tests/*.py", "tests/testData/*", "dashConfig.yaml"],
    },
    install_requires=[
        "statsmodels",
        "numpy",
        "scipy",
        "plotly>=5.16",
        "pandas",
        "dash>=2.5",
        "dash_bootstrap_components",
        "scikit-learn",
        "dash_loading_spinners",
        "kaleido",
        "dash_daq",
        "dash_extensions",
        "pyYAML",
        "dash[diskcache]"
    ],
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    scripts=[
        "RAPDOR/executables.py",
        "versioneer.py"
    ],
    entry_points={
        "console_scripts": [
            "RAPDOR = RAPDOR.executables:main"
        ]
    },
)