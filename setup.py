import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setuptools.setup(
    name="mediffusion",
    version="0.6.3",
    author="Bardia Khosravi",
    author_email="bardiakhosravi95@gmail.com",
    description="Diffusion Models for Medical Imaging",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BardiaKh/Mediffusion",
    packages = setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "bkh-pytorch-utils==0.9.2",
        "torchextractor==0.3.0",
        "OmegaConf==2.3.0",
    ],
    package_data={
        'mediffusion': ['./default_config/default.yaml'],
    },
    python_requires='>=3.8',
)