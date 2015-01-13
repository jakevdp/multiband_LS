from distutils.core import setup

DESCRIPTION = "Methods for periodic fitting of multiband data"
LONG_DESCRIPTION = open('README.md').read()
NAME = "multiband_LS"
AUTHOR = "Jake VanderPlas"
AUTHOR_EMAIL = "jakevdp@uw.edu"
MAINTAINER = "Jake VanderPlas"
MAINTAINER_EMAIL = "jakevdp@uw.edu"
URL = 'http://github.com/jakevdp/multiband_LS'
DOWNLOAD_URL = 'http://github.com/jakevdp/multiband_LS'
LICENSE = 'BSD 3-clause'

import supersmoother
VERSION = supersmoother.__version__

setup(name=NAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      url=URL,
      download_url=DOWNLOAD_URL,
      license=LICENSE,
      packages=['multiband_LS',
                'multiband_LS.tests',
            ],
      classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4'],
     )
