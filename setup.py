from setuptools import setup, find_packages
from codecs import open
from os import path

__version__ = "1.9.4"

here = path.abspath(path.dirname(__file__))

# get the dependencies and installs
with open(path.join(here, "requirements.txt"), encoding="utf-8") as f:
    all_reqs = f.read().split("\n")

install_requires = [x.strip() for x in all_reqs if "git+" not in x]
dependency_links = [
    x.strip().replace("git+", "") for x in all_reqs if x.startswith("git+")
]

setup(
    name="pdroot",
    version=__version__,
    description="utilities for working with ROOT files and pandas",
    long_description="See github for fully rendered README",
    url="https://github.com/aminnj/pdroot",
    download_url="https://github.com/aminnj/pdroot/tarball/" + __version__,
    license="BSD",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 2.7",
    ],
    keywords="",
    packages=find_packages(exclude=["docs", "tests*", "examples"]),
    include_package_data=True,
    author="Nick Amin",
    install_requires=install_requires,
    dependency_links=dependency_links,
    author_email="amin.nj@gmail.com",
)
