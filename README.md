[![PyPI](https://img.shields.io/pypi/status/image-match.svg?maxAge=2592000)](https://pypi.python.org/pypi/image-match)
[![PyPI](https://img.shields.io/pypi/v/image-match.svg)](https://pypi.python.org/pypi/image-match)
[![Documentation Status](https://readthedocs.org/projects/image-match/badge/?version=latest)](https://image-match.readthedocs.org/en/latest/)

# image-match
image-match is a simple (now Python 3!) package for finding approximate image matches from a
corpus.  It is similar, for instance, to [pHash](http://www.phash.org/), but
includes a database backend that easily scales to billions of images and
supports sustained high rates of image insertion: up to 10,000 images/s on our
cluster!

Based on the paper [_An image signature for any kind of image_, Wong et
al](http://www.cs.cmu.edu/~hcwong/Pdfs/icip02.ps).  There is an existing
[reference implementation](https://www.pureftpd.org/project/libpuzzle) which
may be more suited to your needs.

The folks over at [Pavlov](https://usepavlov.com/) have released an excellent
[containerized version of image-match](https://github.com/pavlovml/match) for
easy scaling and deployment.

## Quick start

### [Install and setup image-match](http://image-match.readthedocs.io/en/latest/start.html)

Once you're up and running, read these two (short) sections of the documentation to get a feel 
for what image-match is capable of:

### [Image signatures](http://image-match.readthedocs.io/en/latest/signatures.html)
### [Storing and searching images](http://image-match.readthedocs.io/en/latest/searches.html)
