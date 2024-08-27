from setuptools import setup, find_packages

def read_requirements():
    """Read the requirements.txt file and return a list of dependencies."""
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return fh.read().splitlines()


# Read the contents of README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ChemIC-ml",
    version="1.3",
    description="Chemical images classification project. Program for training the deep neural network model and web service for classification  chemical images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Dr.Aleksei Krasnov",
    author_email="a.krasnov@digital-science.com",
    license="MIT",
    python_requires=">=3.10,<3.12",
    classifiers=[
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],

    url="https://github.com/ontochem/ChemIC.git",
    packages=find_packages(exclude=["tests", "tests.*", "models", "Benchmark"]),
    package_dir={'chemic': 'chemic'},
    install_requires=read_requirements(),
)
