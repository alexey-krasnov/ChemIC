from setuptools import setup, find_packages

# Read the version from __init__.py
with open("chemic/__init__.py", "r", encoding="utf-8") as fh:
    exec(fh.read())

def read_requirements():
    """Read the requirements.txt file and return a list of dependencies."""
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return fh.read().splitlines()


# Read the contents of README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ChemIC-ml",
    version=__version__,
    description="Chemical images classification project. Program for training the deep neural network model and web service for classification  chemical images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Dr.Aleksei Krasnov",
    author_email="dr.aleksei.krasnov@gmail.com",
    license="MIT",
    python_requires=">=3.10,<3.13",
    classifiers=[
        "Topic :: Software Development :: Build Tools",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],

    url="https://github.com/alexey-krasnov/ChemIC.git",
    packages=find_packages(exclude=["tests", "tests.*", "models", "Benchmark"]),
    package_dir={'chemic': 'chemic'},
    install_requires=read_requirements(),
)
