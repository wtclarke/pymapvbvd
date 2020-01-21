Python port of the Matlab mapVBVD tool for reading Siemens raw data .dat 'twix' files.

I have attempted to replicate the syntax of the original matlab code, but there are a few differences due to differing variable types.

Run using the following:
```
twixObj = mapVBVD(filename)
```

For multi raid files (VD+) twixObj is a list, for single files it is a dict containing data with keys relating to the evalinfo flags and a header object. `twixObj.keys()`

Data can be accessed using e.g.
```
data = twixObj[currKey]['']
data = twixObj[currKey][:,:,:]
data = twixObj[currKey][:,0,:]
```

To remove singleton dimensions `twixObj['image'].squeeze = True`.

Header information is contained in a dict `twixObj['hdr']`
`twixObj['hdr'].keys()` provides a list of the data containers.
Acess them using e.g. `twixObj['hdr']['MeasYaps']`
These objects are in turn a final level of dictionaries. The actual data values can be accessed using tuples of key values.

Much of the auxillery parts of mapVBVD (e.g. line reflection, OS removal) is not yet implemented. Please feel free to contribute these parts! As this is a port the code is intended to resemble the original matlab code, it is not "good" python code!


