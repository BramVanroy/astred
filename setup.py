from pathlib import Path
from setuptools import setup

extras = {"stanza": ["stanza"],
          "spacy": ["spacy"],
          "align": ["awesome_align @ git+https://github.com/BramVanroy/awesome-align.git@astred_compat"]}

extras["parsers"] = extras["stanza"] + extras["spacy"]
extras["all"] = extras["stanza"] + extras["spacy"] + extras["align"]
extras["dev"] = extras["all"] + ["isort>=5.5.4", "black", "flake8", "pytest", "pytest_cases"],

setup(
    name="astred",
    version="0.1.0",
    description="A collection of syntactic metrics to calculate (dis)similarities between source and target sentences.",
    long_description=Path("README.rst").read_text(encoding="utf-8"),
    long_description_content_type="text/x-rst",
    keywords="nlp tree-edit-distance ted syntax compling computational-linguistics syntactic-distance translation",
    packages=["astred"],
    url="https://github.com/BramVanroy/astred",
    author="Bram Vanroy",
    author_email="bramvanroy@hotmail.com",
    license="Apache 2.0",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Text Processing",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent"
    ],
    project_urls={
        "Issue tracker": "https://github.com/BramVanroy/astred/issues",
        "Source": "https://github.com/BramVanroy/astred"
    },
    python_requires=">=3.7",
    install_requires=[
        "apted",
        "nltk"
    ],
    extras_require=extras
)
