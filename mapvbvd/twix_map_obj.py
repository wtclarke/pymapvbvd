import copy
import logging
import time
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RectBivariateSpline
from tqdm.auto import trange


def complex_interp(src_grid, xi, yi, z):
    interpolator_r = RectBivariateSpline(src_grid[0], src_grid[1], z.real)
    interpolator_i = RectBivariateSpline(src_grid[0], src_grid[1], z.imag)
    zz_r = interpolator_r(xi, yi)
    zz_i = interpolator_i(xi, yi)
    return zz_r + 1j * zz_i


class twix_map_obj:

    @property
    def filename(self):
        return self.fname

    @property
    def rampSampTrj(self):
        return self.rstrj

    @property
    def dataType(self):
        return self.dType

    @property
    def fullSize(self):
        if self.full_size is None:
            self.clean()
        return self.full_size

    # @fullSize.setter
    # def fullSize(self, val):
    #     self.full_size = val

    @property
    def dataSize(self):
        out = self.fullSize.copy()

        if self.removeOS:
            ix = self.dataDims.index('Col')
            out[ix] = self.NCol / 2

        if self.average_dim[0] | self.average_dim[1]:
            print('averaging in col and cha dim not supported, resetting flag')
            self.average_dim[0:2] = False

        out[self.average_dim] = 1
        return out

    @property
    def sqzSize(self):
        return self.dataSize[self.dataSize > 1].astype(int)

    @property
    def sqzDims(self):
        out = []
        squeezedDim = self.dataSize > 1
        for sd, dim in zip(squeezedDim, self.dataDims):
            if sd:
                out.append(dim)
        return out

    @property
    def flagRemoveOS(self):
        return self.removeOS

    @flagRemoveOS.setter
    def flagRemoveOS(self, removeOS):
        self.removeOS = removeOS

    @property
    def flagAverageDim(self):
        return self.average_dim

    @flagAverageDim.setter
    def flagAverageDim(self, val):
        self.average_dim = val

    @property
    def flagDoAverage(self):
        ix = self.dataDims.index('Ave')
        return self.average_dim[ix]

    @flagDoAverage.setter
    def flagDoAverage(self, bval):
        ix = self.dataDims.index('Ave')
        self.average_dim[ix] = bval

    @property
    def flagAverageReps(self):
        ix = self.dataDims.index('Rep')
        return self.average_dim[ix]

    @flagAverageReps.setter
    def flagAverageReps(self, bval):
        ix = self.dataDims.index('Rep')
        self.average_dim[ix] = bval

    @property
    def flagAverageSets(self):
        ix = self.dataDims.index('Set')
        return self.average_dim[ix]

    @flagAverageSets.setter
    def flagAverageSets(self, bval):
        ix = self.dataDims.index('Set')
        self.average_dim[ix] = bval

    @property
    def flagIgnoreSeg(self):
        ix = self.dataDims.index('Seg')
        return self.average_dim[ix]

    @flagIgnoreSeg.setter
    def flagIgnoreSeg(self, bval):
        ix = self.dataDims.index('Seg')
        self.average_dim[ix] = bval

    @property
    def flagSkipToFirstLine(self):
        return self.skipToFirstLine

    @flagSkipToFirstLine.setter
    def flagSkipToFirstLine(self, bval):
        if bval != self.skipToFirstLine:
            self.skipToFirstLine = bval

            if bval:
                self.skipLin = np.min(self.Lin)
                self.skipPar = np.min(self.Par)
            else:
                self.skipLin = 0
                self.skipPar = 0

            self.full_size[2] = np.maximum(1, self.NLin - self.skipLin)
            self.full_size[3] = np.maximum(1, self.NPar - self.skipPar)

    @property
    def flagRampSampRegrid(self):
        return self.regrid

    @flagRampSampRegrid.setter
    def flagRampSampRegrid(self, bval):
        if bval and self.rstrj is None:
            raise Exception('No trajectory for regridding available')
        self.regrid = bval

    # TODO: flagDoRawDataCorrect
    @property
    def flagDoRawDataCorrect(self):

        return False

    @flagDoRawDataCorrect.setter
    def flagDoRawDataCorrect(self, bval):
        pass

    # TODO: RawDataCorrectionFactors
    @property
    def RawDataCorrectionFactors(self):
        return []

    @RawDataCorrectionFactors.setter
    def RawDataCorrectionFactors(self, bval):
        pass

    def __init__(self, dataType, fname, version, rstraj=None, **kwargs):
        self.ignoreROoffcenter = kwargs.get('ignoreROoffcenter', False)
        self.removeOS = kwargs.get('removeOS', True)
        self.regrid = kwargs.get('regrid', True)
        self.doAverage = kwargs.get('doAverage', False)
        self.averageReps = kwargs.get('averageReps', False)
        self.averageSets = kwargs.get('averageSets', False)
        self.ignoreSeg = kwargs.get('ignoreSeg', False)
        self.squeeze = kwargs.get('squeeze', False)

        self.dType = dataType.lower()
        self.fname = fname
        self.softwareVersion = version

        # self.IsReflected      = logical([]);
        # self.IsRawDataCorrect = logical([]); %SRY
        self.NAcq = 0
        self.isBrokenFile = False

        self.dataDims = ['Col', 'Cha', 'Lin', 'Par', 'Sli', 'Ave', 'Phs',
                         'Eco', 'Rep', 'Set', 'Seg', 'Ida', 'Idb', 'Idc', 'Idd', 'Ide']

        @dataclass
        class FRI:
            szScanHeader: int  # bytes
            szChannelHeader: int  # bytes
            iceParamSz: int
            sz: np.array = np.zeros(2)
            shape: np.array = np.zeros(2)
            cut: np.array = None

        if self.softwareVersion == 'vb':
            self.freadInfo = FRI(0, 128, 4)

        elif self.softwareVersion == 'vd':
            self.freadInfo = FRI(192, 32, 24)  # vd version supports up to 24 ice params

        else:
            raise ValueError('software version not supported')

        self.rstrj = rstraj
        if rstraj is None:
            self.regrid = False

        self.NCol = None
        self.NCha = None
        self.Lin = None
        self.Ave = None
        self.Sli = None
        self.Par = None
        self.Eco = None
        self.Phs = None
        self.Rep = None
        self.Set = None
        self.Seg = None
        self.Ida = None
        self.Idb = None
        self.Idc = None
        self.Idd = None
        self.Ide = None

        self.centerCol = None
        self.centerLin = None
        self.centerPar = None
        self.cutOff = None
        self.coilSelect = None
        self.ROoffcenter = None
        self.timeSinceRF = None
        self.IsReflected = None
        self.scancounter = None
        self.timestamp = None
        self.pmutime = None
        self.IsRawDataCorrect = None
        self.slicePos = None
        self.iceParam = None
        self.freeParam = None

        self.memPos = None

        self.NLin = None
        self.NPar = None
        self.NSli = None
        self.NAve = None
        self.NPhs = None
        self.NEco = None
        self.NRep = None
        self.NSet = None
        self.NSeg = None
        self.NIda = None
        self.NIdb = None
        self.NIdc = None
        self.NIdd = None
        self.NIde = None

        self.skipLin = None
        self.skipPar = None
        self.full_size = None

        # Flags
        self.average_dim = np.full(16, False, dtype=bool)
        self.average_dim[self.dataDims.index('Ave')] = self.doAverage
        self.average_dim[self.dataDims.index('Rep')] = self.averageReps
        self.average_dim[self.dataDims.index('Set')] = self.averageSets
        self.average_dim[self.dataDims.index('Seg')] = self.ignoreSeg

        if self.dType == 'image' or self.dType == 'phasestab':
            self.skipToFirstLine = False
        else:
            self.skipToFirstLine = True

    def __str__(self):
        data_sz = np.array2string(self.fullSize, formatter={"float": lambda x: "%.0f" % x}, separator=",")
        data_sz_sqz = np.array2string(self.sqzSize, formatter={"int": lambda x: "%i" % x}, separator=",")
        des_str = ('***twix_map_obj***\n'
                   f'File: {self.fname}\n'
                   f'Software: {self.softwareVersion}\n'
                   f'Number of acquisitions read {self.NAcq}\n'
                   f'Data size is {data_sz}\n'
                   f'Squeezed data size is {data_sz_sqz} ({self.sqzDims})\n'
                   f'NCol = {self.NCol:0.0f}\n'
                   f'NCha = {self.NCha:0.0f}\n'
                   f'NLin  = {self.NLin:0.0f}\n'
                   f'NAve  = {self.NAve:0.0f}\n'
                   f'NSli  = {self.NSli:0.0f}\n'
                   f'NPar  = {self.NPar:0.0f}\n'
                   f'NEco  = {self.NEco:0.0f}\n'
                   f'NPhs  = {self.NPhs:0.0f}\n'
                   f'NRep  = {self.NRep:0.0f}\n'
                   f'NSet  = {self.NSet:0.0f}\n'
                   f'NSeg  = {self.NSeg:0.0f}\n'
                   f'NIda  = {self.NIda:0.0f}\n'
                   f'NIdb  = {self.NIdb:0.0f}\n'
                   f'NIdc  = {self.NIdc:0.0f}\n'
                   f'NIdd  = {self.NIdd:0.0f}\n'
                   f'NIde  = {self.NIde:0.0f}')
        return des_str

    def __repr__(self):
        return str(self)

    def readMDH(self, mdh, filePos, useScan):
        #         % extract all values in all MDHs at once
        #         %
        #         % data types:
        #         % Use double for everything non-logical, both ints and floats. Seems the
        #         % most robust way to avoid unexpected cast-issues with very nasty side effects.
        #         % Examples: eps(single(16777216)) == 2
        #         %           uint32( 10 ) - uint32( 20 ) == 0
        #         %           uint16(100) + 1e5 == 65535
        #         %           size(1 : 10000 * uint16(1000)) ==  [1  65535]
        #         %
        #         % The 1st example always hits the timestamps.

        self.NAcq = np.sum(useScan)
        sLC = mdh.sLC.astype(float)
        evalInfoMask1 = mdh.aulEvalInfoMask[useScan, 0]

        # save mdh information for each line
        self.NCol = mdh.ushSamplesInScan[useScan].astype(float)
        self.NCha = mdh.ushUsedChannels[useScan].astype(float)
        self.Lin = sLC[useScan, 0].astype(float)
        self.Ave = sLC[useScan, 1].astype(float)
        self.Sli = sLC[useScan, 2].astype(float)
        self.Par = sLC[useScan, 3].astype(float)
        self.Eco = sLC[useScan, 4].astype(float)
        self.Phs = sLC[useScan, 5].astype(float)
        self.Rep = sLC[useScan, 6].astype(float)
        self.Set = sLC[useScan, 7].astype(float)
        self.Seg = sLC[useScan, 8].astype(float)
        self.Ida = sLC[useScan, 9].astype(float)
        self.Idb = sLC[useScan, 10].astype(float)
        self.Idc = sLC[useScan, 11].astype(float)
        self.Idd = sLC[useScan, 12].astype(float)
        self.Ide = sLC[useScan, 13].astype(float)

        self.centerCol = mdh.ushKSpaceCentreColumn[useScan].astype(float)
        self.centerLin = mdh.ushKSpaceCentreLineNo[useScan].astype(float)
        self.centerPar = mdh.ushKSpaceCentrePartitionNo[useScan].astype(float)
        self.cutOff = mdh.sCutOff[useScan].astype(float)
        self.coilSelect = mdh.ushCoilSelect[useScan].astype(float)
        self.ROoffcenter = mdh.fReadOutOffcentre[useScan].astype(float)
        self.timeSinceRF = mdh.ulTimeSinceLastRF[useScan].astype(float)
        self.IsReflected = np.minimum(evalInfoMask1 & 2 ** 24, 1).astype(bool)
        self.scancounter = mdh.ulScanCounter[useScan].astype(float)
        self.timestamp = mdh.ulTimeStamp[useScan].astype(float)
        self.pmutime = mdh.ulPMUTimeStamp[useScan].astype(float)
        self.IsRawDataCorrect = np.minimum(evalInfoMask1 & 2 ** 10, 1).astype(bool)
        self.slicePos = mdh.SlicePos[useScan].astype(float)
        self.iceParam = mdh.aushIceProgramPara[useScan].astype(float)
        self.freeParam = mdh.aushFreePara[useScan].astype(float)

        self.memPos = filePos[useScan]

    def tryAndFixLastMdh(self):
        isLastAcqGood = False
        cnt = 0

        while not isLastAcqGood and self.NAcq > 0 and cnt < 100:
            try:
                # self.clean()
                self.unsorted(self.NAcq)
                isLastAcqGood = True
            except Exception:
                logging.exception(f'An error occurred whilst trying to fix last MDH. NAcq = {self.NAcq:.0f}')
                self.isBrokenFile = True
                self.NAcq -= 1

            cnt += 1

    def clean(self):
        # Cut mdh data to actual size. Maybe we rejected acquisitions at the end
        # due to read errors.
        if self.NAcq == 0:
            return

        # fields = ['NCol', 'NCha',
        #           'Lin', 'Par', 'Sli', 'Ave', 'Phs', 'Eco', 'Rep',
        #           'Set', 'Seg', 'Ida', 'Idb', 'Idc', 'Idd', 'Ide',
        #           'centerCol', 'centerLin', 'centerPar', 'cutOff',
        #           'coilSelect', 'ROoffcenter', 'timeSinceRF', 'IsReflected',
        #           'scancounter', 'timestamp', 'pmutime', 'IsRawDataCorrect',
        #           'slicePos', 'iceParam', 'freeParam', 'memPos']

        # nack = self.NAcq
        # idx = np.arange(0, nack - 1)

        # for f in fields:
        #     curr = getattr(self, f)
        #     if curr.shape[0] > nack:  # rarely
        #         print('Here')
        #         setattr(self, f, curr[idx])  # 1st dim: samples,  2nd dim acquisitions

        self.NLin = np.max(self.Lin) + 1  # +1 so that size isn't 0
        self.NPar = np.max(self.Par) + 1
        self.NSli = np.max(self.Sli) + 1
        self.NAve = np.max(self.Ave) + 1
        self.NPhs = np.max(self.Phs) + 1
        self.NEco = np.max(self.Eco) + 1
        self.NRep = np.max(self.Rep) + 1
        self.NSet = np.max(self.Set) + 1
        self.NSeg = np.max(self.Seg) + 1
        self.NIda = np.max(self.Ida) + 1
        self.NIdb = np.max(self.Idb) + 1
        self.NIdc = np.max(self.Idc) + 1
        self.NIdd = np.max(self.Idd) + 1
        self.NIde = np.max(self.Ide) + 1

        # ok, let us assume for now that all NCol and NCha entries are
        # the same for all mdhs:

        # WTC not sure if this is a good idea - will keep the same as original for now
        if self.NCol.ndim > 0:
            self.NCol = self.NCol[0]
        if self.NCha.ndim > 0:
            self.NCha = self.NCha[0]

        if self.dType == 'refscan':
            # pehses: check for lines with 'negative' line/partition numbers
            # this can happen when the reference scan line/partition range
            # exceeds the one of the actual imaging scan
            if self.NLin > 65500:  # uint overflow check
                self.Lin = np.mod(self.Lin + (65536 - np.min(self.Lin[self.Lin > 65500])), 65536)
                self.NLin = np.max(self.Lin)

            if self.NPar > 65500:  # %uint overflow check
                self.Par = np.mod(self.Par + (65536 - np.min(self.Par[self.Par > 65500])), 65536)
                self.NPar = np.max(self.Par)

        #         to reduce the matrix sizes of non-image scans, the size
        #         of the refscan_obj()-matrix is reduced to the area of the
        #         actually scanned acs lines (the outer part of k-space
        #         that is not scanned is not filled with zeros)
        #         this behaviour is controlled by flagSkipToFirstLine which is
        #         set to true by default for everything but image scans
        # both used to have a -1 but WTC thinks that in python they won't be needed
        if not self.skipToFirstLine:
            self.skipLin = 0
            self.skipPar = 0
        else:
            self.skipLin = np.min(self.Lin)  # -1
            self.skipPar = np.min(self.Par)  # -1

        NLinAlloc = np.maximum(1, self.NLin - self.skipLin)
        NParAlloc = np.maximum(1, self.NPar - self.skipPar)

        self.full_size = np.array(
            [self.NCol, self.NCha, NLinAlloc, NParAlloc,
             self.NSli, self.NAve, self.NPhs, self.NEco,
             self.NRep, self.NSet, self.NSeg, self.NIda,
             self.NIdb, self.NIdc, self.NIdd, self.NIde],
            dtype=int)

        nByte = self.NCha * (self.freadInfo.szChannelHeader + 8 * self.NCol)

        # size for fread
        self.freadInfo.sz = np.array([2, nByte / 8])
        # reshape size
        self.freadInfo.shape = np.array([self.NCol + self.freadInfo.szChannelHeader / 8, self.NCha])
        # we need to cut MDHs from fread data
        self.freadInfo.cut = self.freadInfo.szChannelHeader / 8 + np.arange(self.NCol)

    def calcRange(self, S):
        self.clean()
        selRange = [np.zeros(1, dtype=int)] * self.dataSize.size
        outSize = np.ones(self.dataSize.shape, dtype=int)

        bSqueeze = self.squeeze
        if S is None or S is slice(None, None, None):
            # shortcut to select all data
            for k in range(0, self.dataSize.size):
                selRange[k] = np.arange(0, self.dataSize[k]).astype(int)
            if not bSqueeze:
                outSize = self.dataSize.astype(int)
            else:
                outSize = self.sqzSize
        else:
            # import pdb; pdb.set_trace()
            for k, s in enumerate(S):
                if not bSqueeze:
                    cDim = k  # nothing to do
                else:
                    # we need to rearrange selRange from squeezed
                    # to original order
                    for i, x in enumerate(self.dataDims):
                        if x == self.sqzDims[k]:
                            cDim = i

                if s == slice(None, None, None):
                    if k < (len(S) - 1):
                        selRange[cDim] = np.arange(0, self.dataSize[cDim]).astype(int)
                    else:  # all later dimensions selected and 'vectorized'!
                        for ll in range(cDim, self.dataSize.size):
                            selRange[ll] = np.arange(0, self.dataSize[ll]).astype(int)
                        outSize[k] = np.prod(self.dataSize[cDim:])
                        break
                elif isinstance(s, slice):
                    tmpTuple = s.indices(self.dataSize[cDim].astype(int))
                    selRange[cDim] = np.arange(tmpTuple[0], tmpTuple[1], tmpTuple[2])
                else:  # numeric
                    selRange[cDim] = np.array([s])

                outSize[k] = selRange[cDim].size

            for r, s in zip(selRange, self.dataSize):
                if np.max(r) > s:
                    raise Exception('selection out of range')
            # To implement indexing

        selRangeSz = np.ones(self.dataSize.shape, dtype=int)
        for idx, k in enumerate(selRange):
            selRangeSz[idx] = k.size

        # now select all indices for the dims that are averaged
        for iDx, k in enumerate(np.nditer(self.average_dim)):
            if k:
                self.clean()
                selRange[iDx] = np.arange(0, self.fullSize[iDx])

        return selRange, selRangeSz, outSize

    def calcIndices(self):
        # calculate indices to target & source(raw)
        LinIx = self.Lin - self.skipLin
        ParIx = self.Par - self.skipPar
        sz = self.fullSize[2:]

        ixToTarget = np.zeros(LinIx.size, dtype=int)
        for i, _ in enumerate(ixToTarget):
            ixToTarget[i] = np.ravel_multi_index((LinIx[i].astype(int), ParIx[i].astype(int), self.Sli[i].astype(int),
                                                  self.Ave[i].astype(int), self.Phs[i].astype(int),
                                                  self.Eco[i].astype(int), self.Rep[i].astype(int),
                                                  self.Set[i].astype(int), self.Seg[i].astype(int),
                                                  self.Ida[i].astype(int), self.Idb[i].astype(int),
                                                  self.Idc[i].astype(int), self.Idd[i].astype(int),
                                                  self.Ide[i].astype(int)), dims=sz.astype(int), order='C')

        # now calc. inverse index (page table: virtual to physical addresses)
        # indices of lines that are not measured are zero
        ixToRaw = np.full(self.fullSize[2:].prod().astype(int), np.nan, dtype=int)

        for i, itt in enumerate(ixToTarget):
            ixToRaw[itt] = i

        return ixToRaw, ixToTarget

    def unsorted(self, ival=None):
        # returns the unsorted data [NCol,NCha,#samples in acq. order]
        if ival:
            mem = np.atleast_1d(self.memPos[ival - 1])
        else:
            mem = self.memPos
        out = self.readData(mem)
        return out

    # Replicate matlab subscripting
    # Overloads []
    def __getitem__(self, key=None):
        # print(f'In [], key is {key}.')
        # import pdb; pdb.set_trace()
        if isinstance(key, slice):  # Handle single input e.g. [:]
            key = (key,)  # make an iterable for calcRange
        elif key == '':
            key = None
        selRange, selRangeSz, outSize = self.calcRange(key)  # True for squeezed data

        # calculate page table (virtual to physical addresses)
        # this is now done every time, i.e. result is no longer saved in
        # a property - slower but safer (and easier to keep track of updates)
        ixToRaw, _ = self.calcIndices()

        tmp = np.arange(0, self.fullSize[2:].prod().astype(int)).reshape(self.fullSize[2:].astype(int))
        # tmpSelRange = [x-1 for x in selRange] # python indexing from 0
        for i, ids in enumerate(selRange[2:]):
            tmp = np.take(tmp, ids.astype(int), i)
        # tmp = tmp[tuple(selRange[2:])]

        ixToRaw = ixToRaw[tmp]
        ixToRaw = ixToRaw.ravel()
        # delete all entries that point to zero (the "NULL"-pointer)
        # notAcquired = np.isnan(ixToRaw)
        ixToRaw = ixToRaw[np.where(ixToRaw > -1)[0]]
        # import pdb; pdb.set_trace()
        # broken ixToRaw = np.delete(ixToRaw, ~notAcquired)# Why do I have to negate bool here? Not clear
        # maar = np.ma.MaskedArray(ixToRaw, mask=~notAcquired)
        # ixToRaw = maar.compressed()

        # calculate ixToTarg for possibly smaller, shifted + segmented
        # target matrix:
        cIx = np.zeros((14, ixToRaw.size), dtype=int)
        if ~self.average_dim[2]:
            cIx[0, :] = self.Lin[ixToRaw] - self.skipLin
        if ~self.average_dim[3]:
            cIx[1, :] = self.Par[ixToRaw] - self.skipPar
        if ~self.average_dim[4]:
            cIx[2, :] = self.Sli[ixToRaw]
        if ~self.average_dim[5]:
            cIx[3, :] = self.Ave[ixToRaw]
        if ~self.average_dim[6]:
            cIx[4, :] = self.Phs[ixToRaw]
        if ~self.average_dim[7]:
            cIx[5, :] = self.Eco[ixToRaw]
        if ~self.average_dim[8]:
            cIx[6, :] = self.Rep[ixToRaw]
        if ~self.average_dim[9]:
            cIx[7, :] = self.Set[ixToRaw]
        if ~self.average_dim[10]:
            cIx[8, :] = self.Seg[ixToRaw]
        if ~self.average_dim[11]:
            cIx[9, :] = self.Ida[ixToRaw]
        if ~self.average_dim[12]:
            cIx[10, :] = self.Idb[ixToRaw]
        if ~self.average_dim[13]:
            cIx[11, :] = self.Idc[ixToRaw]
        if ~self.average_dim[14]:
            cIx[12, :] = self.Idd[ixToRaw]
        if ~self.average_dim[15]:
            cIx[13, :] = self.Ide[ixToRaw]
        # import pdb; pdb.set_trace()

        # make sure that indices fit inside selection range
        for k in range(2, len(selRange)):
            tmp = cIx[k - 2, :]
            for ll in range(0, selRange[k].size):
                cIx[k - 2, tmp == selRange[k][ll]] = ll

        sz = selRangeSz[2:]
        ixToTarg = np.zeros(cIx.shape[1], dtype=int)  # pylint: disable=E1136  # pylint/issues/3139
        for i, _ in enumerate(ixToTarg):
            ixToTarg[i] = np.ravel_multi_index((cIx[0, i].astype(int), cIx[1, i].astype(int), cIx[2, i].astype(int),
                                                cIx[3, i].astype(int), cIx[4, i].astype(int), cIx[5, i].astype(int),
                                                cIx[6, i].astype(int), cIx[7, i].astype(int), cIx[8, i].astype(int),
                                                cIx[9, i].astype(int), cIx[10, i].astype(int), cIx[11, i].astype(int),
                                                cIx[12, i].astype(int), cIx[13, i].astype(int)), dims=sz.astype(int),
                                               order='C')

        mem = self.memPos[ixToRaw]
        # sort mem for quicker access, sort cIxToTarg/Raw accordingly
        ix = np.argsort(mem)
        mem = mem[ix]
        ixToTarg = ixToTarg[ix]
        ixToRaw = ixToRaw[ix]
        # import pdb; pdb.set_trace()
        out = self.readData(mem, ixToTarg, ixToRaw, selRange, selRangeSz, outSize)

        return out

    @staticmethod
    def cast2MinimalUint(N):
        Nmax = np.max(N)
        Nmin = np.min(N)
        if (Nmin < 0) or (Nmax > np.iinfo(np.uint64).max):
            return N

        if Nmax > np.iinfo(np.uint32).max:
            idxClass = np.uint64
        elif Nmax > np.iinfo(np.uint16).max:
            idxClass = np.uint32
        else:
            idxClass = np.uint16

        return N.astype(idxClass)

    def _fileopen(self):
        fid = open(self.fname, 'rb')
        return fid

    def readData(self, mem, cIxToTarg=None, cIxToRaw=None, selRange=None, selRangeSz=None, outSize=None):

        mem = mem.astype(int)
        if outSize is None:
            if selRange is None:
                selRange = [np.arange(0, self.dataSize[0]).astype(int),
                            np.arange(0, self.dataSize[1]).astype(int)]
            else:
                selRange[0] = np.arange(0, self.dataSize[0]).astype(int)
                selRange[1] = np.arange(0, self.dataSize[0]).astype(int)

            outSize = np.concatenate((self.dataSize[0:2], mem.shape)).astype(int)
            selRangeSz = outSize
            cIxToTarg = np.arange(0, selRangeSz[2])
            cIxToRaw = cIxToTarg
        # else:
        #     if np.array_equiv(selRange[0],np.arange(0,self.dataSize()[0]).astype(int)):
        #         selRange[0] = slice(None,None,None)
        #     if np.array_equiv(selRange[1],np.arange(0,self.dataSize()[1]).astype(int)):
        #         selRange[1] = slice(None,None,None)

        out = np.zeros(outSize, dtype=np.csingle)
        out = out.reshape((selRangeSz[0], selRangeSz[1], -1))

        cIxToTarg = twix_map_obj.cast2MinimalUint(cIxToTarg)  # Possibly not needed

        # These parameters were copied for speed in matlab, but just duplicate to keep code similar in python
        szScanHeader = self.freadInfo.szScanHeader
        readSize = self.freadInfo.sz.astype(int)
        readShape = self.freadInfo.shape.astype(int)
        readCut = self.freadInfo.cut.astype(int)
        keepOS = np.concatenate([list(range(int(self.NCol / 4))), list(range(int(self.NCol * 3 / 4), int(self.NCol)))])

        bIsReflected = self.IsReflected[cIxToRaw]
        bRegrid = self.regrid and self.rstrj.size > 1
        # slicedata = self.slicePos[cIxToRaw, :]
        ro_shift = self.ROoffcenter[cIxToRaw] * int(not self.ignoreROoffcenter)
        # %SRY store information about raw data correction
        # bDoRawDataCorrect = this.arg.doRawDataCorrect;
        # bIsRawDataCorrect = this.IsRawDataCorrect( cIxToRaw );
        isBrokenRead = False
        # if (bDoRawDataCorrect)
        #     rawDataCorrect = this.arg.rawDataCorrectionFactors;
        # end

        """
        % MiVÃ¶: Raw data are read line-by-line in portions of 2xNColxNCha float32 points (2 for complex).
        % Computing and sorting(!) on these small portions is quite expensive, esp. when
        % it employs non-sequential memory paths. Examples are non-linear k-space acquisition
        % or reflected lines.
        % This can be sped up if slightly larger blocks of raw data are collected, first.
        % Whenever a block is full, we do all those operations and save it in the final "out" array.
        % What's a good block size? Depends on data size and machine (probably L2/L3/L4 cache sizes).
        % So...? Start with a small block, measure the time-per-line and double block size until
        % a minimum is found. Seems sufficiently robust to end up in a close-to-optimal size for every
        % machine and data.
        """
        blockSz = 2  # size of blocks; must be 2^n; will be increased
        doLockblockSz = False  # whether blockSZ should be left untouched
        tprev = float('inf')  # previous time-per-line
        blockCtr = 0
        blockInit = np.full((readShape[0], readShape[1], blockSz), -np.inf, dtype=np.csingle)  # init with garbage
        block = copy.deepcopy(blockInit)
        sz = 0
        if bRegrid:
            v1 = np.array(range(1, selRangeSz[1] * blockSz + 1))
            rsTrj = [self.rstrj, v1]
            trgTrj = np.linspace(np.min(self.rstrj), np.max(self.rstrj), int(self.NCol))
            trgTrj = [trgTrj, v1]

        # counter for proper scaling of averages/segments
        count_ave = np.zeros((1, 1, out.shape[2]), np.single)  # pylint: disable=E1136  # pylint/issues/3139
        kMax = mem.size  # max loop index

        fid = self._fileopen()

        for k in trange(kMax, desc='read data', leave=False):  # could loop over mem, but keep it similar to matlab
            # skip scan header
            fid.seek(mem[k] + szScanHeader, 0)
            raw = np.fromfile(fid, dtype=np.float32, count=readSize.prod()).reshape(
                (readSize[1], readSize[0]))  # do transpose by switching readSize order
            # With incomplete files fread() returns less than readSize points. The subsequent reshape will
            # therefore error out. We could check if numel(raw) == prod(readSize), but people recommend
            # exception handling for performance reasons. Do it.
            try:
                raw = (raw[:, 0] + 1j * raw[:, 1]).reshape(readShape, order='F')
            except ValueError:
                offset_bytes = mem[k] + szScanHeader
                # remainingSz = readSize(2) - size(raw,1);
                import warnings
                warnstring =\
                    'An unexpected read error occurred at this byte offset:'\
                    f' {offset_bytes} ({offset_bytes / 1024 ** 3} GiB).'\
                    f'\nActual read size is [{raw.shape}], desired size was: [{readSize}].\n'\
                    'Will ignore this line and stop reading.\n'
                warnings.warn(warnstring)
                # Reject this data fragment. To do so, init with the values of blockInit
                raw[0:readShape.prod()] = blockInit[0]
                raw = raw.reshape(readShape)
                isBrokenRead = True  # remember it and bail out later

            block[:, :, blockCtr, None] = copy.deepcopy(raw).reshape(np.append(readShape, 1))
            # fast serial storage in a cache array - this is probably all very dependent on whether I've got things
            # contiguous in memory. I highly doubt that I have on this first pass. WTC
            blockCtr += 1

            # Do expensive computations and reorderings on the gathered block.
            # Unfortunately, a lot of code is necessary, but that is executed much less
            # frequent, so its worthwhile for speed.
            # TODO: Do *everything* block-by-block
            if (blockCtr == blockSz) or (k == kMax - 1) or (isBrokenRead & blockCtr > 1):
                # measure the time to process a block of data
                tic = time.perf_counter()

                # remove MDH data from block:
                block = block[readCut, :, :]

                ix = np.arange(1 + k - blockCtr, k + 1, dtype=int)  # +1 so that it goes to k
                if blockCtr != blockSz:
                    block = block[:, :, 0:blockCtr]

                # if  bDoRawDataCorrect && bIsRawDataCorrect(k): WTC: not implemented yet

                # reflect fids when necessary
                isRefl = np.where(bIsReflected[ix])[0]
                block[:, :, isRefl] = block[-1::-1, :, isRefl]

                if bRegrid:
                    # correct for readout shifts
                    # the nco frequency is always scaled to the max. gradient amp and does account for ramp-sampling
                    deltak = np.max(np.abs(np.diff(rsTrj[0])))
                    fovshift = np.reshape(ro_shift[ix], (1, 1, len(ix)), order='F')
                    adcphase = deltak * (np.array(range(int(self.NCol)))[..., np.newaxis, np.newaxis] * fovshift)
                    fovphase = fovshift * rsTrj[0][..., np.newaxis, np.newaxis]
                    phase_factor = np.exp(1j * 2 * np.pi * (adcphase - fovphase))
                    block *= phase_factor

                    if block.shape != sz:
                        # update grid
                        sz = block.shape
                        ninterp = np.prod(sz[1:])
                        if ninterp == 1:
                            blockdims = [0]
                        else:
                            blockdims = [0, 1]
                        rsTrj[1] = np.array(range(ninterp)) + 1
                        src_grid = tuple(rsTrj[dim] for dim in blockdims)

                    # regrid the data
                    trgTrj[1] = rsTrj[1]
                    trg_grid = tuple(trgTrj[dim] for dim in blockdims)
                    z = np.reshape(block, (sz[0], -1), order='F')
                    # TODO: handle 1d regridding
                    yi, xi = np.meshgrid(trg_grid[1], trg_grid[0], sparse=True)

                    # NOTE: there is some minor differences in regridding precision between python and matlab, don't
                    # expect the same result from regridding

                    # Perform the interpolation with the given values
                    block = np.reshape(complex_interp(src_grid, xi, yi, z), sz,
                                       order='F')

                if self.removeOS:
                    block = np.fft.fft(
                        np.fft.ifft(block, axis=0)[keepOS, ...], axis=0)

                # TODO: average across 1st and 2nd dims
                # import pdb; pdb.set_trace()

                # WTC whilst still using slices rather than just arrays.
                # if (not isinstance(selRange[0],slice)) or (not isinstance(selRange[1],slice)):
                # if isinstance(selRange[0],slice) and (selRange[0]==slice(None,None,None)):
                #     cur1stDim = block.shape[0]
                # else:
                #     cur1stDim = selRange[0].size
                # if isinstance(selRange[1],slice) and (selRange[1]==slice(None,None,None)):
                #     cur2ndDim = block.shape[1]
                # else:
                #     cur2ndDim = selRange[1].size

                cur1stDim = selRange[0].size
                cur2ndDim = selRange[1].size
                cur3rdDim = block.shape[2]
                block = block[selRange[0][:, np.newaxis], selRange[1][np.newaxis, :], :]\
                    .reshape((cur1stDim, cur2ndDim, cur3rdDim))

                toSort = cIxToTarg[ix]
                II = np.argsort(toSort)
                sortIdx = toSort[II]
                block = block[:, :, II]  # reorder according to sorted target indices

                # Mark duplicate indices with 1; we'll have to treat them special for proper averaging
                # Bonus: The very first storage can be made much faster, because it's in-place.
                isDupe = np.concatenate((np.array([False]), np.diff(sortIdx) == 0))

                idx1 = sortIdx[~isDupe]  # acquired once in this block
                idxN = sortIdx[isDupe]  # acquired multiple times

                count_ave[:, :, idx1] += 1

                if idxN.size == 0:
                    # no duplicates
                    if (count_ave[:, :, idx1] == 1).all():  # first acquisition of this line
                        out[:, :, idx1] = block  # fast
                    else:
                        out[:, :, idx1] = out[:, :, idx1] + block  # slow
                else:
                    out[:, :, idx1] = out[:, :, idx1] + block[:, :, ~isDupe]  # slower

                    block = block[:, :, isDupe]
                    for n in range(0, idxN.size):
                        out[:, :, idxN[n]] = out[:, :, idxN[n]] + block[:, :, n]  # snail :-)
                        count_ave[:, :, idxN[n]] += 1

                # At the first few iterations, evaluate the spent time-per-line and decide
                # what to do with the block size.
                if not doLockblockSz:
                    toc = time.perf_counter()
                    t = 1e6 * (toc - tic) / blockSz  # micro seconds

                    if t <= 1.1 * tprev:  # allow 10% inaccuracy. Usually bigger == better
                        # New block size was faster. Go a step further.
                        blockSz = blockSz * 2
                        blockInit = np.concatenate((blockInit, blockInit), axis=2)
                    else:
                        # regression; reset size and lock it
                        blockSz = np.maximum(blockSz / 2, 1).astype(int)
                        blockInit = blockInit[:, :, :blockSz]
                        doLockblockSz = True

                    tprev = t

                blockCtr = 0
                block = blockInit  # reset to garbage

            if isBrokenRead:
                self.isBrokenFile = True
                break

        fid.close()

        #  proper scaling (we don't want to sum our data but average it)
        #  For large "out" bsxfun(@rdivide,out,count_ave) is incredibly faster than
        #  bsxfun(@times,out,count_ave)!
        #  @rdivide is also running in parallel, while @times is not. :-/
        if (count_ave > 1).any():
            count_ave = np.maximum(1, count_ave)
            out /= count_ave

        out = np.ascontiguousarray(out.reshape(outSize))

        return out if not self.squeeze else np.squeeze(out)

    """
      ATH added August 2021
         alter a loop counter, for example if its missing in the mdh, add it in here, or alter it
         the length of the new loop must be = self.NAcq
         fixLoopCounter('Ave',newLlistAve)
    """
    def fixLoopCounter(self, loop, newLoop):
        if newLoop.shape[0] != self.NAcq:
            print('length of new array must equal NAcq: ' + str(self.NAcq))
        self.__dict__[loop] = newLoop
        N = max(newLoop)
        self.__dict__['N' + loop] = N
        self.fullSize[self.dataDims.index(loop)] = N
