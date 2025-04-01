 # setup for the trustworthai python library with required packages

from setuptools import find_packages, setup

setup(
    name="trustworthai",
    version="0.1.0",
    packages=find_packages(exclude=["examples", "misc experiments", "slurm_cluster", "other_works_demos"]),

    install_requires = [
        "numpy==1.20",
        "pandas==1.2",
        "openpyxl",
        "statsmodels",
        "matplotlib",
        "proplot",
        "seaborn",
        "nibabel",
        "SimpleITK",
	"itk",
        "pydicom",
        "pyrobex",
        "itkwidgets",
        "natsort",
        "torch",
        "torchvision",
        "deepspeed",
        "monai",
        "tqdm",
        "connected-components-3d",
        "scipy",
        "scikit-image",
        "scikit-learn",
        "kornia",
        "tensorboard",
        "pytorch-lightning",
        "jupyterlab==3.6.5",
        "smbprotocol",
        "torchinfo",
    ]

)
