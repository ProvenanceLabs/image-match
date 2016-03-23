"""
image_match is a simple package for finding approximate image matches from a
corpus. It is similar, for instance, to pHash <http://www.phash.org/>, but
includes a database backend that easily scales to billions of images and
supports sustained high rates of image insertion: up to 10,000 images/s on our
cluster!

Based on the paper An image signature for any kind of image, Goldberg et
al <http://www.cs.cmu.edu/~hcwong/Pdfs/icip02.ps>.
"""
import io
import os
import re

from setuptools import setup, find_packages


def read(*names, **kwargs):
    with io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get('encoding', 'utf8')
    ) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(
        r'^__version__ = [\'"]([^\'"]*)[\'"]', version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError('Unable to find version string.')


tests_require = [
    'coverage',
    'pep8',
    'pyflakes',
    'pylint',
    'pytest',
    'pytest-cov',
    'pytest-xdist',
]

dev_require = [
    'ipdb',
    'ipython',
]

docs_require = [
    'recommonmark>=0.4.0',
    'Sphinx>=1.3.5',
    'sphinxcontrib-napoleon>=0.4.4',
    'sphinx-rtd-theme>=0.1.9',
]


setup(
    name='image_match',
    version=find_version('image_match', '__init__.py'),
    description='image_match is a simple package for finding approximate '\
                'image matches from a corpus.',
    long_description=__doc__,
    url='https://github.com/ascribe/image-match/',
    author='Ryan Henderson',
    author_email='ryan@ascribe.io',
    license='Apache License 2.0',
    zip_safe=True,

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Database',
        'Topic :: Database :: Database Engines/Servers',
        'Topic :: Software Development',
        'Natural Language :: English',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX :: Linux',
        'Topic :: Multimedia :: Graphics',
    ],

    packages=find_packages(),

    setup_requires=[
        'pytest-runner',
    ],
    install_requires=[
        'scipy>=0.17,<0.18',
        'scikit-image>=0.12,<0.13',
        'cairosvg>1,<2',
        'elasticsearch>=2.3,<2.4',
    ],
    tests_require=tests_require,
    extras_require={
        'test': tests_require,
        'dev':  dev_require + tests_require + docs_require,
        'docs':  docs_require,
    },
)
