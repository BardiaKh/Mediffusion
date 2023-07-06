import setuptools

setuptools.setup(
    name="bkh_ddpm",
    version="0.3.4",
    author="Bardia Khosravi",
    author_email="bardiakhosravi95@gmail.com",
    description="Personal take on DDPMs",
    url="https://github.com/BardiaKh/DDPM",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "bkh_pytorch_utils>=0.8.0",
        "torchextractor>=0.3.0",
        "OmegaConf>=2.0.0",
    ],
    python_requires='>=3.8',
)
