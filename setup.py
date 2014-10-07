#!/usr/bin/env python
"""SymPsi: Symbolic quantum mechanics.
"""

DOCLINES = __doc__.split('\n')

CLASSIFIERS = """\
Development Status :: 4 - Beta
Intended Audience :: Science/Research
License :: OSI Approved :: BSD License
Programming Language :: Python
Programming Language :: Python :: 3
Topic :: Scientific/Engineering
Operating System :: MacOS
Operating System :: POSIX
Operating System :: Unix
Operating System :: Microsoft :: Windows
"""

import os
import sys
import numpy as np
from numpy.distutils.core import setup

MAJOR = 0
MINOR = 1
MICRO = 0
ISRELEASED = False
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)
REQUIRES = ['sympy (>= 0.7.5)']
PACKAGES = ['sympsi']
NAME = "sympsi"
AUTHOR = "Robert Johansson, Eunjong Kim"
AUTHOR_EMAIL = "robert@riken.jp"
LICENSE = "BSD"
DESCRIPTION = DOCLINES[0]
LONG_DESCRIPTION = "\n".join(DOCLINES[2:])
KEYWORDS = "symbolic quantum mechanics"
URL = "http://github.com/jrjohansson/sympsi"
CLASSIFIERS = [_f for _f in CLASSIFIERS.split('\n') if _f]
PLATFORMS = ["Linux", "Mac OSX", "Unix", "Windows"]


def git_short_hash():
    try:
        return "-" + os.popen('git log -1 --format="%h"').read().strip()
    except:
        return ""

FULLVERSION = VERSION
if not ISRELEASED:
    FULLVERSION += '.dev' + git_short_hash()


def write_version_py(filename='sympsi/version.py'):
    cnt = """\
# THIS FILE IS GENERATED FROM SYMPSI SETUP.PY
short_version = '%(version)s'
version = '%(fullversion)s'
release = %(isrelease)s
"""
    with open(filename, 'w') as f:
        f.write(cnt % {'version': VERSION, 'fullversion':
                FULLVERSION, 'isrelease': str(ISRELEASED)})

write_version_py()


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration(None, parent_package, top_path)
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    config.add_subpackage('sympsi')
    config.get_version('sympsi/version.py')

    return config


setup(
    name=NAME,
    packages=PACKAGES,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    license=LICENSE,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    keywords=KEYWORDS,
    url=URL,
    classifiers=CLASSIFIERS,
    platforms=PLATFORMS,
    requires=REQUIRES,
    configuration=configuration
)
