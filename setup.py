from setuptools import find_packages, setup

setup(
    name="wmhuq",
    version="1.0",
    packages=find_packages(exclude=['weights']),

    install_requires = [
        "numpy==1.26.4",
        "pandas==2.0.0",
        "SimpleITK==2.3.1",
        "natsort==8.4.0",
        "torch",#==2.4.0",
        "torchvision",#==0.19.0",
        "pytorch-lightning",#==2.2.1",
        "connected-components-3d==3.18.0",
        "scipy==1.12.0",
        "scikit-image==0.22.0",
        "scikit-learn==1.4.1.post1",
    ]
)
