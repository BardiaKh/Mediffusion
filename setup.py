import setuptools

setuptools.setup(
    name="mediffusion",
    version="0.5.4",
    author="Bardia Khosravi",
    author_email="bardiakhosravi95@gmail.com",
    description="Diffusion Models for Medical Imaging",
    url="https://github.com/BardiaKh/Mediffusion",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "bkh_pytorch_utils @ git+https://github.com/BardiaKh/PytorchUtils.git@42eba26", #v: 0.8.3
        "torchextractor>=0.3.0",
        "OmegaConf>=2.0.0",
    ],
    package_data={
        'mediffusion': ['./default_config/default.yaml'],
    },
    python_requires='>=3.8',
)
