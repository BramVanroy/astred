from pathlib import Path
from setuptools import setup

setup(
    name='astred',
    version='0.1.0',
    description='A collection of syntactically aware metrics to calculate equivalence. Also provides useful utility'
                ' scripts for those who work with dependency trees.',
    long_description=Path('README.rst').read_text(encoding="utf-8"),
    long_description_content_type='text/x-rst',
    keywords='nlp tree-edit-distance ted syntax compling computational-linguistics syntactic-distance translation',
    packages=['astred'],
    url='https://github.com/BramVanroy/astred',
    author='Bram Vanroy',
    author_email='bramvanroy@hotmail.com',
    license='Apache 2.0',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Text Processing',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Operating System :: OS Independent'
    ],
    project_urls={
        'Issue tracker': 'https://github.com/BramVanroy/astred',
        'Source': 'https://github.com/BramVanroy/astred'
    },
    python_requires='>=3.6',
    install_requires=[
        'apted',
        'nltk',
        'stanza'
    ],
    extras_require={
        'dev': [
            'isort @ git+git://github.com/timothycrosley/isort.git@e63ae06ec7d70b06df9e528357650281a3d3ec22#egg=isort',
            'black',
            'flake8',
            'pytest'
        ]
    }
)
