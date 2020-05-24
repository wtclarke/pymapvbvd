Python port of the Matlab mapVBVD tool for reading Siemens raw data 'twix' (.dat) files.

## Use

I have attempted to replicate the syntax of the original matlab code, but there are a few differences due to differing variable types.

This package contains a demo Jupyter notebook 'Demo.ipynb' which can be run on the two bits of demo data found in tests/test_data. Both of these data are unsuppressed water SVS MRS, one from a 7T VB scanner and the other from a VE Prisma.

Run using the following:
```
twixObj = mapVBVD(filename)
```

For multi raid files (VD+) twixObj is a list, for single files it is a AttDict containing data with keys relating to the MDH flags and a header object. The MDH flags present can be retrieved as a list using `twixObj.MDH_flags()`, whilst the header is found at `twixObj.hdr`.

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
Acess them manually using e.g. `twixObj.hdr['MeasYaps']` or `twixObj.hdr.MeasYaps`.  
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
Much of the auxillery parts of mapVBVD (e.g. line reflection, OS removal) is not yet implemented. Please feel free to contribute these parts! As this is a port the code is intended to resemble the original matlab code, it is not "good" python code!

## Credit where credit is due
This code is a port of Philipp Ehses' original Matlab code. I am incredibly grateful to him for releasing this code to the MR community. There are a number of other names in the original code comments and log, these are: Felix Breuer, Wolf Blecher, Stephen Yutzy, Zhitao Li, Michael VÃlker, Jonas Bause and Chris Mirke.

This port is released under the MIT licence and no credit is sought for its use.