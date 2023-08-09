import setuptools

setuptools.setup(
    name="mediffusion",
    version="0.5.0",
    author="Bardia Khosravi",
    author_email="bardiakhosravi95@gmail.com",
    description="Diffusion Models for Medical Imaging",
    url="https://github.com/BardiaKh/MeDiffusion",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "git+https://github.com/BardiaKh/PytorchUtils.git@42eba26",
        "bkh_pytorch_utils>=0.8.0",
        "torchextractor>=0.3.0",
        "OmegaConf>=2.0.0",
    ],
    python_requires='>=3.8',
)
