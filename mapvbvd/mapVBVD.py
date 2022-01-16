from dataclasses import dataclass

import numpy as np
from tqdm.auto import tqdm

import mapvbvd as pkg
from mapvbvd._attrdict import AttrDict
from .read_twix_hdr import read_twix_hdr, twix_hdr
from .twix_map_obj import twix_map_obj


def get_bit(number, pos):
    return (number >> pos) & 1


def set_bit(v, index, x):
    # Set the index:th bit of v to 1 if x is truthy, else to 0, and return the new value."""
    mask = 1 << index  # Compute mask, an integer with just bit 'index' set.
    v &= mask  # Clear the bit indicated by the mask (if x is False)
    if x:
        v |= mask  # If x was True, set the bit indicated by the mask.
    return v  # Return the result, we're done.


def loop_mdh_read(fid, version, Nscans, scan, measOffset, measLength, print_prog=True):
    if version == 'vb':
        isVD = False
        byteMDH = 128
    elif version == 'vd':
        isVD = True
        byteMDH = 184
        szScanHeader = 192  # [bytes]
        szChannelHeader = 32  # [bytes]
    else:
        isVD = False
        byteMDH = 128
        import warnings
        warnings.warn(f'Software version "{version}" is not supported.')

    cPos = fid.tell()
    n_acq = 0
    allocSize = 4096
    ulDMALength = byteMDH
    isEOF = False

    mdh_blob = np.zeros((byteMDH, 0), dtype=np.uint8)
    szBlob = mdh_blob.shape[1]  # pylint: disable=E1136  # pylint/issues/3139
    filePos = np.zeros(0, dtype=float)

    fid.seek(cPos, 0)

    # constants and conditional variables
    bit_0 = np.array(2 ** 0, dtype=np.uint8)
    bit_5 = np.array(2 ** 5, dtype=np.uint8)
    mdhStart = -byteMDH  # Different to matlab - index = -128

    u8_000 = np.zeros((3, 1), dtype=np.uint8)

    # 20 fill bytes in VD (21:40)
    # Subtract one from Idx numbers to account for indexing from 0 in python
    evIdx = np.array(21 + 20 * isVD, dtype=np.uint8) - 1  # 1st byte of evalInfoMask
    dmaIdx = np.array(np.arange(29, 33) + 20 * isVD, dtype=np.uint8) - 1  # to correct DMA length using NCol and NCha
    if isVD:
        dmaOff = szScanHeader
        dmaSkip = szChannelHeader
    else:
        dmaOff = 0
        dmaSkip = byteMDH

    if print_prog:
        last_progress = 0
        t = tqdm(total=measLength, unit='B', unit_scale=True, unit_divisor=1024,
                 desc='Scan %d/%d, read all mdhs' % (scan + 1, Nscans), leave=False)
    while True:
        #         Read mdh as binary (uint8) and evaluate as little as possible to know...
        #           ... where the next mdh is (ulDMALength / ushSamplesInScan & ushUsedChannels)
        #           ... whether it is only for sync (MDH_SYNCDATA)
        #           ... whether it is the last one (MDH_ACQEND)
        #         evalMDH() contains the correct and readable code for all mdh entries.
        try:
            # read everything and cut out the mdh
            data_u8 = np.fromfile(fid, dtype=np.uint8, count=int(ulDMALength))
            if data_u8.size < int(ulDMALength):
                raise EOFError
            data_u8 = data_u8[mdhStart:]

        except EOFError:
            import warnings
            warningString =\
                '\nAn unexpected read error occurred at this byte offset:'\
                f' {cPos} ({cPos / 1024 ** 3} GiB)\n'
            warningString += 'Will stop reading now.\n'
            warnings.warn(warningString)
            isEOF = True
            break

        bitMask = data_u8[evIdx]  # the initial 8 bit from evalInfoMask are enough
        # print(bitMask)

        if ((data_u8[0:3] == u8_000).all()) or (bitMask & bit_0):

            # ok, look closer if really all *4* bytes are 0
            data_u8[3] = get_bit(data_u8[3], 0)  # ubit24: keep only 1 bit from the 4th byte
            tmp = data_u8[0:4]
            tmp.dtype = np.uint32
            ulDMALength = float(tmp)

            if (ulDMALength == 0) or (bitMask & bit_0):
                cPos = cPos + ulDMALength
                # jump to next full 512 bytes
                if cPos % 512:
                    cPos = cPos + 512 - cPos % 512
                # isEOF = True
                break

        if bitMask & bit_5:  # MDH_SYNCDATA
            data_u8[3] = get_bit(data_u8[3], 0)  # ubit24: keep only 1 bit from the 4th byte
            tmp = data_u8[0:4]
            tmp.dtype = np.uint32
            ulDMALength = float(tmp)
            cPos = cPos + ulDMALength
            continue

        #          pehses: the pack bit indicates that multiple ADC are packed into one
        #          DMA, often in EPI scans (controlled by fRTSetReadoutPackaging in IDEA)
        #          since this code assumes one adc (x NCha) per DMA, we have to correct
        #          the "DMA length"
        #              if mdh.ulPackBit
        #          it seems that the packbit is not always set correctly
        tmp = data_u8[dmaIdx]
        tmp.dtype = np.uint16
        NCol_NCha = tmp  # was float [ushSamplesInScan  ushUsedChannels]
        ulDMALength = dmaOff + (8 * NCol_NCha[0] + dmaSkip) * NCol_NCha[1]

        n_acq = n_acq + 1

        # grow arrays in batches
        if n_acq > szBlob:
            grownArray = np.zeros((mdh_blob.shape[0], allocSize),
                                  dtype=np.uint8)  # pylint: disable=E1136  # pylint/issues/3139
            mdh_blob = np.concatenate((mdh_blob, grownArray), axis=1)

            filePos = np.concatenate((filePos, np.zeros((allocSize))), axis=0)

            szBlob = mdh_blob.shape[1]  # pylint: disable=E1136  # pylint/issues/3139

        mdh_blob[:, n_acq - 1] = data_u8
        filePos[n_acq - 1] = cPos

        if print_prog:
            curr_progress = cPos
            progress = curr_progress - last_progress
            t.update(progress)
            last_progress = curr_progress

        cPos = cPos + int(ulDMALength)

    if isEOF or n_acq == len(filePos):
        n_acq = n_acq - 1  # ignore the last attempt
    # import pdb; pdb.set_trace()
    filePos[n_acq] = cPos

    # discard overallocation:

    mdh_blob = mdh_blob[:, :n_acq]
    filePos = filePos[:n_acq]  # in matlab was converted to row vector

    return mdh_blob, filePos, isEOF


def evalMDH(mdh_blob, version):
    if version == 'vd':
        isVD = True
        mdh_blob = np.concatenate((mdh_blob[0:20, :], mdh_blob[40:, :]), axis=0)  # remove 20 unnecessary bytes
    else:
        isVD = False

    Nmeas = mdh_blob.shape[1]

    ulPackBit = get_bit(mdh_blob[3, :], 2)
    ulPCI_rx = set_bit(mdh_blob[3, :], 7, False)  # keep 6 relevant bits
    ulPCI_rx = set_bit(np.int16(ulPCI_rx), 8, False)
    mdh_blob[3, :] = get_bit(mdh_blob[3, :], 1)  # ubit24: keep only 1 bit from the 4th byte

    data_uint32 = np.ascontiguousarray(mdh_blob[0:76, :].transpose())
    data_uint32.dtype = np.uint32
    data_uint16 = np.ascontiguousarray(mdh_blob[28:, :].transpose())
    data_uint16.dtype = np.uint16
    data_single = np.ascontiguousarray(mdh_blob[68:, :].transpose())
    data_single.dtype = np.single

    @dataclass
    class MDH:  # byte pos
        ulPackBit: np.uint8
        ulPCI_rx: np.uint8
        SlicePos: np.single
        aushIceProgramPara: np.uint16
        aushFreePara: np.uint16
        lMeasUID: np.uint32 = data_uint32[:, 2 - 1]  # 5 :   8
        ulScanCounter: np.uint32 = data_uint32[:, 3 - 1]  # 9 :  12
        ulTimeStamp: np.uint32 = data_uint32[:, 4 - 1]  # 13 :  16
        ulPMUTimeStamp: np.uint32 = data_uint32[:, 5 - 1]  # 17 :  20
        aulEvalInfoMask: np.uint32 = data_uint32[:, 5:7]  # 21 :  28
        ushSamplesInScan: np.uint16 = data_uint16[:, 1 - 1]  # 29 :  30
        ushUsedChannels: np.uint16 = data_uint16[:, 2 - 1]  # 31 :  32
        sLC: np.uint16 = data_uint16[:, 2:16]  # 33 :  60
        sCutOff: np.uint16 = data_uint16[:, 16:18]  # 61 :  64
        ushKSpaceCentreColumn: np.uint16 = data_uint16[:, 19 - 1]  # 66 :  66
        ushCoilSelect: np.uint16 = data_uint16[:, 20 - 1]  # 67 :  68
        fReadOutOffcentre: np.single = data_single[:, 1 - 1]  # 69 :  72
        ulTimeSinceLastRF: np.uint32 = data_uint32[:, 19 - 1]  # 73 :  76
        ushKSpaceCentreLineNo: np.uint16 = data_uint16[:, 25 - 1]  # 77 :  78
        ushKSpaceCentrePartitionNo: np.uint16 = data_uint16[:, 26 - 1]  # 79 :  80

    if isVD:
        mdh = MDH(ulPackBit, ulPCI_rx, data_single[:, 3:10], data_uint16[:, 40:64], data_uint16[:, 64:68])
    else:
        mdh = MDH(ulPackBit, ulPCI_rx, data_single[:, 7:14], data_uint16[:, 26:30], data_uint16[:, 30:34])

    evalInfoMask1 = mdh.aulEvalInfoMask[:, 0]

    @dataclass
    class MASK:
        MDH_ACQEND = np.minimum(evalInfoMask1 & 2 ** 0, 1)
        MDH_RTFEEDBACK = np.minimum(evalInfoMask1 & 2 ** 1, 1)
        MDH_HPFEEDBACK = np.minimum(evalInfoMask1 & 2 ** 2, 1)
        MDH_SYNCDATA = np.minimum(evalInfoMask1 & 2 ** 5, 1)
        MDH_RAWDATACORRECTION = np.minimum(evalInfoMask1 & 2 ** 10, 1)
        MDH_REFPHASESTABSCAN = np.minimum(evalInfoMask1 & 2 ** 14, 1)
        MDH_PHASESTABSCAN = np.minimum(evalInfoMask1 & 2 ** 15, 1)
        MDH_SIGNREV = np.minimum(evalInfoMask1 & 2 ** 17, 1)
        MDH_PHASCOR = np.minimum(evalInfoMask1 & 2 ** 21, 1)
        MDH_PATREFSCAN = np.minimum(evalInfoMask1 & 2 ** 22, 1)
        MDH_PATREFANDIMASCAN = np.minimum(evalInfoMask1 & 2 ** 23, 1)
        MDH_REFLECT = np.minimum(evalInfoMask1 & 2 ** 24, 1)
        MDH_NOISEADJSCAN = np.minimum(evalInfoMask1 & 2 ** 25, 1)
        MDH_VOP = np.minimum(mdh.aulEvalInfoMask[:, 1] & 2 ** (53 - 32),
                             1)  # WTC modified this as the original matlab code didn't make sense
        MDH_IMASCAN = np.ones(Nmeas, dtype=np.uint32)

    mask = MASK()

    noImaScan = (mask.MDH_ACQEND | mask.MDH_RTFEEDBACK | mask.MDH_HPFEEDBACK
                 | mask.MDH_PHASCOR | mask.MDH_NOISEADJSCAN | mask.MDH_PHASESTABSCAN
                 | mask.MDH_REFPHASESTABSCAN | mask.MDH_SYNCDATA
                 | (mask.MDH_PATREFSCAN & ~mask.MDH_PATREFANDIMASCAN))

    mask.MDH_IMASCAN -= noImaScan

    return mdh, mask


def mapVBVD(filename, quiet=False, **kwargs):
    if not quiet:
        print(f'pymapVBVD version {pkg.__version__}')

    # parse kw arguments
    # bReadImaScan = kwargs.get('bReadImaScan', True)
    # bReadNoiseScan = kwargs.get('bReadNoiseScan', True)
    # bReadPCScan = kwargs.get('bReadPCScan', True)
    # bReadRefScan = kwargs.get('bReadRefScan', True)
    # bReadRefPCScan = kwargs.get('bReadRefPCScan', True)
    # bReadRTfeedback = kwargs.get('bReadRTfeedback', True)
    # bReadPhaseStab = kwargs.get('bReadPhaseStab', True)
    # bReadHeader = kwargs.get('bReadHeader', True)

    # ignoreROoffcenter = kwargs.get('ignoreROoffcenter', False)

    # TODO: handle filename input variations
    fid = open(filename, 'rb')

    fid.seek(0, 2)
    fileSize = fid.tell()

    fid.seek(0, 0)
    firstInt = np.fromfile(fid, dtype=np.uint32, count=1, offset=0)
    secondInt = np.fromfile(fid, dtype=np.uint32, count=1, offset=0)

    if (firstInt < 10000) & (secondInt <= 64):
        version = 'vd'
        if not quiet:
            print('Software version: VD')

        NScans = secondInt[0]
        measID = np.fromfile(fid, dtype=np.uint32, count=1, offset=0)  # noqa: F841
        fileID = np.fromfile(fid, dtype=np.uint32, count=1, offset=0)  # noqa: F841
        measOffset = np.zeros(NScans, dtype=np.uint64)
        measLength = np.zeros(NScans, dtype=np.uint64)
        for k in range(NScans):
            measOffset[k] = np.fromfile(fid, dtype=np.uint64, count=1, offset=0)
            measLength[k] = np.fromfile(fid, dtype=np.uint64, count=1, offset=0)
            fid.seek(152 - 16, 1)

    else:
        version = 'vb'
        if not quiet:
            print('Software version: VB')

        # in VB versions, the first 4 bytes indicate the beginning of the raw data part of the file
        measOffset = np.zeros(1, dtype=np.uint64)
        measLength = np.array([fileSize], dtype=np.uint64)
        NScans = 1  # VB does not support multiple scans in one file

    # Read data correction factors
    # TODO: fill in for VB
    if version == 'vb':
        pass

    # data will be read in two steps (two while loops):
    #   1) reading all MDHs to find maximum line no., partition no.,... for
    #      ima, ref,... scan
    #   2) reading the data
    twix_obj = []

    for s in range(NScans):
        cPos = measOffset[s]
        fid.seek(cPos, 0)
        hdr_len = np.fromfile(fid, dtype=np.uint32, count=1, offset=0)

        currTwixObj = AttrDict()
        currTwixObjHdr = twix_hdr()
        # rstraj = 0
        # read header
        currTwixObjHdr, rstraj = read_twix_hdr(fid, currTwixObjHdr)
        currTwixObj.update({'hdr': currTwixObjHdr})

        # declare data objects:
        def mytmo(dtype):
            return twix_map_obj(dtype, filename, version, rstraj, **kwargs)
        currTwixObj.update({'image': mytmo('image')})
        currTwixObj.update({'noise': mytmo('noise')})
        currTwixObj.update({'phasecor': mytmo('phasecor')})
        currTwixObj.update({'phasestab': mytmo('phasestab')})
        currTwixObj.update({'phasestab_ref0': mytmo('phasestab_ref0')})
        currTwixObj.update({'phasestab_ref1': mytmo('phasestab_ref1')})
        currTwixObj.update({'refscan': mytmo('refscan')})
        currTwixObj.update({'refscanPC': mytmo('refscanPC')})
        currTwixObj.update({'refscan_phasestab': mytmo('refscan_phasestab')})
        currTwixObj.update({'refscan_phasestab_ref0': mytmo('refscan_phasestab_ref0')})
        currTwixObj.update({'refscan_phasestab_ref1': mytmo('refscan_phasestab_ref1')})
        currTwixObj.update({'rtfeedback': mytmo('rtfeedback')})
        currTwixObj.update({'vop': mytmo('vop')})

        # TODO: print reader version information
        # if s == 0:
        #     # print(f'Reader version: {currTwixObj['image'].readerVersion})
        #     print('UTC: TODO')

        # jump to first mdh
        cPos += hdr_len
        # print(cPos[0])
        fid.seek(cPos[0], 0)

        # print(f'Scan {s + 1}/{NScans}, read all mdhs:')

        mdh_blob, filePos, isEOF = loop_mdh_read(fid, version, NScans, s, measOffset[s],
                                                 measLength[s], print_prog=not quiet)  # uint8; size: [ byteMDH  Nmeas ]

        cPos = filePos[-1]
        # filePos = filePos[:-1]

        # get mdhs and masks for each scan, no matter if noise, image, RTfeedback etc:
        [mdh, mask] = evalMDH(mdh_blob, version)

        # Assign mdhs to their respective scans and parse it in the correct twix objects.

        # MDH_IMASCAN
        isCurrScan = mask.MDH_IMASCAN.astype(bool)
        if isCurrScan.any():
            currTwixObj.image.readMDH(mdh, filePos, isCurrScan)
        else:
            currTwixObj.pop('image', None)

        # MDH_NOISEADJSCAN
        isCurrScan = mask.MDH_NOISEADJSCAN.astype(bool)
        if isCurrScan.any():
            currTwixObj.noise.readMDH(mdh, filePos, isCurrScan)
        else:
            currTwixObj.pop('noise', None)

        # MDH_PATREFSCAN refscan
        isCurrScan = \
            (mask.MDH_PATREFSCAN | mask.MDH_PATREFANDIMASCAN)\
            & ~(mask.MDH_PHASCOR
                | mask.MDH_PHASESTABSCAN
                | mask.MDH_REFPHASESTABSCAN
                | mask.MDH_RTFEEDBACK
                | mask.MDH_HPFEEDBACK)
        isCurrScan = isCurrScan.astype(bool)
        if isCurrScan.any():
            currTwixObj.refscan.readMDH(mdh, filePos, isCurrScan)
        else:
            currTwixObj.pop('refscan', None)

        # MDH_RTFEEDBACK
        isCurrScan = (mask.MDH_RTFEEDBACK | mask.MDH_HPFEEDBACK) & ~mask.MDH_VOP
        isCurrScan = isCurrScan.astype(bool)
        if isCurrScan.any():
            currTwixObj.rtfeedback.readMDH(mdh, filePos, isCurrScan)
        else:
            currTwixObj.pop('rtfeedback', None)

        # VOP
        isCurrScan = (mask.MDH_RTFEEDBACK & mask.MDH_VOP)
        isCurrScan = isCurrScan.astype(bool)
        if isCurrScan.any():
            currTwixObj.vop.readMDH(mdh, filePos, isCurrScan)
        else:
            currTwixObj.pop('vop', None)

        # MDH_PHASCOR
        isCurrScan = mask.MDH_PHASCOR & (~mask.MDH_PATREFSCAN | mask.MDH_PATREFANDIMASCAN)
        isCurrScan = isCurrScan.astype(bool)
        if isCurrScan.any():
            currTwixObj.phasecor.readMDH(mdh, filePos, isCurrScan)
        else:
            currTwixObj.pop('phasecor', None)

        # refscanPC
        isCurrScan = mask.MDH_PHASCOR & (mask.MDH_PATREFSCAN | mask.MDH_PATREFANDIMASCAN)
        isCurrScan = isCurrScan.astype(bool)
        if isCurrScan.any():
            currTwixObj.refscanPC.readMDH(mdh, filePos, isCurrScan)
        else:
            currTwixObj.pop('refscanPC', None)

        # phasestab MDH_PHASESTABSCAN
        isCurrScan = (mask.MDH_PHASESTABSCAN & ~mask.MDH_REFPHASESTABSCAN)\
            & (~mask.MDH_PATREFSCAN | mask.MDH_PATREFANDIMASCAN)
        isCurrScan = isCurrScan.astype(bool)
        if isCurrScan.any():
            currTwixObj.phasestab.readMDH(mdh, filePos, isCurrScan)
        else:
            currTwixObj.pop('phasestab', None)

        # refscanPS MDH_PHASESTABSCAN
        isCurrScan = (mask.MDH_PHASESTABSCAN & ~mask.MDH_REFPHASESTABSCAN)\
            & (mask.MDH_PATREFSCAN | mask.MDH_PATREFANDIMASCAN)
        isCurrScan = isCurrScan.astype(bool)
        if isCurrScan.any():
            currTwixObj.refscan_phasestab.readMDH(mdh, filePos, isCurrScan)
        else:
            currTwixObj.pop('refscan_phasestab', None)

        # phasestabRef0 MDH_PHASESTABSCAN
        isCurrScan = (mask.MDH_REFPHASESTABSCAN & ~mask.MDH_PHASESTABSCAN)\
            & (~mask.MDH_PATREFSCAN | mask.MDH_PATREFANDIMASCAN)
        isCurrScan = isCurrScan.astype(bool)
        if isCurrScan.any():
            currTwixObj.phasestab_ref0.readMDH(mdh, filePos, isCurrScan)
        else:
            currTwixObj.pop('phasestab_ref0', None)

        # refscanPSRef0 MDH_PHASESTABSCAN
        isCurrScan = (mask.MDH_REFPHASESTABSCAN & ~mask.MDH_PHASESTABSCAN)\
            & (mask.MDH_PATREFSCAN | mask.MDH_PATREFANDIMASCAN)
        isCurrScan = isCurrScan.astype(bool)
        if isCurrScan.any():
            currTwixObj.refscan_phasestab_ref0.readMDH(mdh, filePos, isCurrScan)
        else:
            currTwixObj.pop('refscan_phasestab_ref0', None)

        # phasestabRef1 MDH_PHASESTABSCAN
        isCurrScan = (mask.MDH_REFPHASESTABSCAN & mask.MDH_PHASESTABSCAN)\
            & (~mask.MDH_PATREFSCAN | mask.MDH_PATREFANDIMASCAN)
        isCurrScan = isCurrScan.astype(bool)
        if isCurrScan.any():
            currTwixObj.phasestab_ref1.readMDH(mdh, filePos, isCurrScan)
        else:
            currTwixObj.pop('phasestab_ref1', None)

        # refscanPSRef1 MDH_PHASESTABSCAN
        isCurrScan = (mask.MDH_REFPHASESTABSCAN & mask.MDH_PHASESTABSCAN)\
            & (mask.MDH_PATREFSCAN | mask.MDH_PATREFANDIMASCAN)
        isCurrScan = isCurrScan.astype(bool)
        if isCurrScan.any():
            currTwixObj.refscan_phasestab_ref1.readMDH(mdh, filePos, isCurrScan)
        else:
            currTwixObj.pop('refscan_phasestab_ref1', None)

        if isEOF:
            # recover from read error
            for keys in currTwixObj:
                if keys != 'hdr':
                    currTwixObj[keys].tryAndFixLastMdh()
        else:
            for keys in currTwixObj:
                if keys != 'hdr':
                    currTwixObj[keys].clean()

        twix_obj.append(myAttrDict(currTwixObj))

    fid.close()

    if len(twix_obj) == 1:
        twix_obj = twix_obj[0]
    # breakpoint()
    return twix_obj


# Add some class methods to AttrDict so that we get around the issue of not being able to
# access methods of objects accessed as attributes.
class myAttrDict(AttrDict):
    def __init__(self, *args):
        super().__init__(*args)

    def search_header_for_keys(self, *args, **kwargs):
        """Search header keys for terms.

            Accesses search_for_keys method in header.
            Args:
                search terms        : search terms as list of strings.
                regex (optional)    : Search using regex or for exact strings.
                top_lvl (optional)  : Specify list of parameter sets to search (e.g. YAPS)
                print_flag(optional): If False no output will be printed.
        """
        return self['hdr'].search_for_keys(*args, **kwargs)

    def search_header_for_val(self, top_lvl, keys, **kwargs):
        """Return values for keys found using search terms for terms.

            Args:
                top_lvl         : Specify list of parameter sets to search (e.g. YAPS)
                keys            : search terms as list of strings.
                regex (optional): Search using regex or for exact strings.
        """
        keys = self['hdr'].search_for_keys(keys, print_flag=False, top_lvl=top_lvl, **kwargs)

        out_vals = []
        for key in keys:
            for skey in keys[key]:
                out_vals.append(self['hdr'][key][skey])

        return out_vals

    def MDH_flags(self):
        """Return list of populated MDH flags."""
        MDH = list(self.keys())
        MDH.pop(MDH.index('hdr'))
        return MDH
