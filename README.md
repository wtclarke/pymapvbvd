# pyMapVBVD

![PyPI](https://img.shields.io/pypi/v/pyMapVBVD)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pyMapVBVD)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5909806.svg)](https://doi.org/10.5281/zenodo.5909806)


Python port of the Matlab mapVBVD tool for reading Siemens raw data 'twix' (.dat) files.

## Installation
`conda install -c conda-forge pymapvbvd`
or
`pip install pymapvbvd`

## Use

I have attempted to replicate the syntax of the original matlab code, but there are a few differences due to differing variable types.

This package contains a demo Jupyter notebook 'Demo.ipynb' which can be run on the demo data found in tests/test_data. There is unsuppressed water SVS MRS, from both a 7T VB scanner and a VE Prisma. There is also imaging data (3D GRE and EPI) from the [ISMRMRD test dataset](https://doi.org/10.5281/zenodo.33166).

Run using the following:
```
import mapvbvd
twixObj = mapvbvd.mapVBVD(filename)
```

For multi raid files (VD+) twixObj is a list, for single files it is a AttrDict containing data with keys relating to the MDH flags and a header object. The MDH flags present can be retrieved as a list using `twixObj.MDH_flags()`, whilst the header is found at `twixObj.hdr`.

## Data

Data can be accessed using e.g. for image MDH
```
data = twixObj.image['']
data = twixObj.image[:,:,:]
data = twixObj.image[0::2,0,:]
```

To remove singleton dimensions `twixObj.image.squeeze = True`.

To retrieve the data in an unsorted format (i.e. Col,Cha,NAcq) use `twixObj.image.unsorted()`.

## Headers

Header information is contained in a dict `twixObj.hdr`
`twixObj.hdr.keys()` provides a list of the data containers.
Access them manually using e.g. `twixObj.hdr['MeasYaps']` or `twixObj.hdr.MeasYaps`.
These objects are in turn a final level of dictionaries. The actual data values can be accessed either manually using tuples of key values e.g.
```
twixObj.hdr.MeasYaps[('sTXSPEC','asNucleusInfo','0','tNucleus')]
```
or you can search for keys and values by custom methods.
```
matching_keys = twixObj.search_header_for_keys(('sTXSPEC', 'asNucleusInfo'))
key_value = twixObj.search_header_for_val('MeasYaps',('sTXSPEC', 'asNucleusInfo', '0', 'tNucleus'))
```

`search_header_for_keys` takes the keyword argument regex (default True) to either search via case insensitive regular expressions or via exact matches only. Specify top_lvl to restrict to just some parameter sets.

## Other info

Thanks to Mo Shahdloo the latest version now implements OS removal, ramp sample regridding, averaging and line reflection.

Set the appropriate flags to enable these features
```
twixObj.image.flagRemoveOS = True
twixObj.image.flagRampSampRegrid = True
twixObj.refscanPC.flagIgnoreSeg = True
twixObj.refscanPC.flagDoAverage = True
```

Some of the auxiliary parts of mapVBVD remain unimplemented. Please feel free to contribute these parts! As this is a port the code is intended to resemble the original matlab code, it is not "good" python code.

## Credit where credit is due
This code is a port of Philipp Ehses' original Matlab code. I am incredibly grateful to him for releasing this code to the MR community. There are a number of other names in the original code comments and log, these are: Felix Breuer, Wolf Blecher, Stephen Yutzy, Zhitao Li, Michael VÃlker, Jonas Bause and Chris Mirke.

More recent thanks to Mo Shahdloo and Aaron Hess for edits, and Alex Craven for performance enhancements.

This port is released under the MIT licence and no credit is sought for its use.
