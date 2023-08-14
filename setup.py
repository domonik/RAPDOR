from setuptools import setup, find_packages, Extension
import versioneer


NAME = "RDPMSpecIdentifier"
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
    package_dir={"RDPMSpecIdentifier": "./RDPMSpecIdentifier"},
    license="LICENSE",
    url="https://github.com/domonik/RDPMSpecIdentifier",
    description=DESCRIPTION,
    long_description=LONGDESC,
    long_description_content_type="text/markdown",
    include_package_data=True,
    package_data={
        "RDPMSpecIdentifier.visualize": ["assets/*"],
        "RDPMSpecIdentifier": ["tests/*.py", "tests/test_data/*"],
        "RDPMSpecIdentifier.qtInterface": ["Icon.svg", "RDPMSpecIdentifier_dark_no_text.svg", "01_style.css"],
    },
    install_requires=[
        "statsmodels",
        "numpy",
        "scipy",
        "plotly",
        "pandas",
        "dash>=2.5",
        "dash_bootstrap_components",
        "scikit-learn",
        "dash_loading_spinners",
        "kaleido",
        "PyQt5",
        "dash_daq"
    ],
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    scripts=[
        "RDPMSpecIdentifier/executables.py",
        "versioneer.py"
    ],
    entry_points={
        "console_scripts": [
            "RDPMSpecIdentifier = RDPMSpecIdentifier.executables:main"
        ]
    },
)