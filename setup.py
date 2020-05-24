#!/usr/bin/env python

from distutils.core import setup

with open('requirements.txt', 'rt') as f:
    install_requires = [l.strip() for l in f.readlines()]


setup(name='pyMapVBVD',
      version='0.2.0',
      description='Python twix reader',
      author=['Will Clarke'],
      author_email=['william.clarke@ndcn.ox.ac.uk'],
      url='https://github.com/wexeee/pymapvbvd',
      packages=['mapVBVD'],
      install_requires=install_requires         
     )
