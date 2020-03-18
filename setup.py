from pathlib import Path
from setuptools import setup

with Path('README.rst').open(encoding='utf-8') as fhin:
    long_description = fhin.read()

setup(
    name='astred',
    version='0.0.1',
    description='A',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    keywords='nlp tree-edit-distance ted syntax compling astred syntactic-distance translation',
    packages=['astred'],
    url='https://github.com/BramVanroy/syntactically-aware-cross',
    author='Bram Vanroy',
    author_email='bramvanroy@hotmail.com',
    license='BSD 2',
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Text Processing',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Operating System :: OS Independent'
    ],
    project_urls={
        'Bug Reports': 'https://github.com/BramVanroy/syntactically-aware-cross',
        'Source': 'https://github.com/BramVanroy/syntactically-aware-cross',
    },
    python_requires='>=3.6',
    install_requires=[
        'apted',
        'nltk',
        'numpy',
        'pandas',
        'spacy_stanfordnlp',
        'stanfordnlp',
        'tqdm'
    ],
    entry_points={
        #'console_scripts': ['main_parse=astred.__main__:main']
    }
)
