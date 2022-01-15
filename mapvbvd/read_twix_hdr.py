import re
import numpy as np
from scipy.integrate import cumtrapz
from itertools import chain

from mapvbvd._attrdict import AttrDict


class twix_hdr(AttrDict):
    def __init__(self, *args):
        super().__init__(*args)

    def __str__(self):
        keystr = '\n'.join([k for k in self.keys()])
        des_str = ('***twix_hdr***\n'
                   f'Top level data structures: \n{keystr}\n')
        return des_str

    def __repr__(self):
        return str(self)

    @staticmethod
    def search_using_tuple(s_terms, key, regex=True):
        if regex:

            def regex_tuple(pattern, key_tuple):
                inner_out = []
                re_comp = re.compile(pattern, re.IGNORECASE)
                for k in key_tuple:
                    m = re_comp.search(k)
                    if m:
                        inner_out.append(True)
                    else:
                        inner_out.append(False)
                return any(inner_out)

            out = []
            for st in s_terms:
                out.append(regex_tuple(st, key))
            return all(out)
        else:
            return all([st in key for st in s_terms])

    def search_for_keys(self, search_terms, top_lvl=None, print_flag=True, regex=True):
        """Search header keys for terms.

            Args:
                search terms        : search terms as list of strings.
                recursive (optional): Search using regex or for exact strings.
                top_lvl (optional)  : Specify list of parameter sets to search (e.g. YAPS)
                print_flag(optional): If False no output will be printed.
        """

        if top_lvl is None:
            top_lvl = self.keys()
        elif isinstance(top_lvl, str):
            top_lvl = [top_lvl, ]

        out = {}
        for key in top_lvl:
            matching_keys = []
            list_of_keys = self[key].keys()
            for sub_key in list_of_keys:
                if twix_hdr.search_using_tuple(search_terms, sub_key, regex=regex):
                    matching_keys.append(sub_key)

            if print_flag:
                print(f'{key}:')
                for mk in matching_keys:
                    print(f'\t{mk}: {self[key][mk]}')

            out.update({key: matching_keys})

        return out


def parse_ascconv(buffer):
    # print(buffer)
    vararray = re.finditer(r'(?P<name>\S*)\s*=\s*(?P<value>\S*)\n', buffer)
    # print(vararray)
    mrprot = AttrDict()
    for v in vararray:
        try:
            value = float(v.group('value'))
        except ValueError:
            value = v.group('value')

        # now split array name and index (if present)
        vvarray = re.finditer(r'(?P<name>\w+)(\[(?P<ix>[0-9]+)\])?', v.group('name'))

        currKey = []
        for vv in vvarray:
            # print(vv.group('name'))
            currKey.append(vv.group('name'))
            if vv.group('ix') is not None:
                # print(vv.group('ix'))
                currKey.append(vv.group('ix'))

        mrprot.update({tuple(currKey): value})

    return mrprot


def parse_xprot(buffer):
    xprot = {}
    tokens = re.finditer(r'<Param(?:Bool|Long|String)\."(\w+)">\s*{([^}]*)', buffer)
    tokensDouble = re.finditer(r'<ParamDouble\."(\w+)">\s*{\s*(<Precision>\s*[0-9]*)?\s*([^}]*)', buffer)
    alltokens = chain(tokens, tokensDouble)

    for t in alltokens:
        # print(t.group(1))
        # print(t.group(2))
        name = t.group(1)

        value = re.sub(r'("*)|( *<\w*> *[^\n]*)', '', t.groups()[-1])
        # value = re.sub(r'\s*',' ',value)
        # for some bonkers reason this inserts whitespace between all the letters!
        # Just look for other whitespace that \s usually does.
        value = re.sub(r'[\t\n\r\f\v]*', '', value.strip())
        try:
            value = float(value)
        except ValueError:
            pass

        xprot.update({name: value})

    return xprot


def parse_buffer(buffer):
    reASCCONV = re.compile(r'### ASCCONV BEGIN[^\n]*\n(.*)\s### ASCCONV END ###', re.DOTALL)
    # print(f'buffer = {buffer[0:10]}')
    # import pdb; pdb.set_trace()

    ascconv = reASCCONV.search(buffer)
    # print(f'ascconv = {ascconv}')
    if ascconv is not None:
        prot = parse_ascconv(ascconv.group(0))
    else:
        prot = AttrDict()

    xprot = reASCCONV.split(buffer)
    # print(f'xprot = {xprot[0][0:10]}')
    if xprot is not None:
        xprot = ''.join([found for found in xprot])
        prot2 = parse_xprot(xprot)

        prot.update(prot2)

    return prot


def read_twix_hdr(fid, prot):
    # function to read raw data header information from siemens MRI scanners
    # (currently VB and VD software versions are supported and tested).

    nbuffers = np.fromfile(fid, dtype=np.uint32, count=1)

    for _ in range(nbuffers[0]):
        tmpBuff = np.fromfile(fid, dtype=np.uint8, count=10)
        bufname = ''.join([chr(item) for item in tmpBuff])
        bufname = re.match(r'^\w*', bufname).group(0)

        fid.seek(len(bufname) - 9, 1)
        buflen = np.fromfile(fid, dtype=np.uint32, count=1)
        tmpBuff = np.fromfile(fid, dtype=np.uint8, count=buflen[0])
        buffer = ''.join([chr(item) for item in tmpBuff])
        buffer = re.sub(r'\n\s*\n', '', buffer)
        prot.update({bufname: parse_buffer(buffer)})

    rstraj = None
    # read gridding info
    if 'alRegridMode' in prot.Meas:
        regrid_mode = int(prot.Meas.alRegridMode.split(' ')[0])
        if regrid_mode > 1:
            ncol = int(prot.Meas.alRegridDestSamples.split(' ')[0])
            dwelltime = float(prot.Meas.aflRegridADCDuration.split(' ')[0]) / ncol
            gr_adc = np.zeros(ncol, dtype=np.single)
            start = float(prot.Meas.alRegridDelaySamplesTime.split(' ')[0])
            time_adc = start + dwelltime * (np.array(range(ncol)) + 0.5)
            rampup_time = float(prot.Meas.alRegridRampupTime.split(' ')[0])
            flattop_time = float(prot.Meas.alRegridFlattopTime.split(' ')[0])
            rampdown_time = float(prot.Meas.alRegridRampdownTime.split(' ')[0])
            ixUp = np.where(time_adc < rampup_time)[0]
            ixFlat = np.setdiff1d(np.where(time_adc <= rampup_time + flattop_time)[0],
                                  np.where(time_adc < rampup_time)[0])
            ixDn = np.setdiff1d(np.setdiff1d(list(range(ncol)), ixFlat), ixUp)
            gr_adc[ixFlat] = 1
            if regrid_mode == 2:
                # trapezoidal gradient
                gr_adc[ixUp] = time_adc[ixUp] / float(prot.Meas.alRegridRampupTime.split(' ')[0])
                gr_adc[ixDn] = 1 - (time_adc[ixDn] - rampup_time - flattop_time) / rampdown_time
            elif regrid_mode == 4:
                gr_adc[ixUp] = np.sin(np.pi / 2 * time_adc[ixUp] / rampup_time)
                gr_adc[ixDn] = np.sin(np.pi / 2 * (1 + (time_adc[ixDn] - rampup_time - flattop_time) / rampdown_time))
            else:
                raise Exception('regridding mode unknown')

            # make sure that gr_adc is always positive (rstraj needs to be strictly monotonic)
            gr_adc = np.maximum(gr_adc, 1e-4)
            rstraj = (np.append(0, cumtrapz(gr_adc)) - ncol / 2) / np.sum(gr_adc)
            rstraj -= np.mean(rstraj[int(ncol / 2) - 1:int(ncol / 2) + 1])
            # scale rstraj by kmax (only works if all slices have same FoV!!!)
            # TODO: these are examples of the keys not arranged correctly
            kmax = prot.MeasYaps[('sKSpace', 'lBaseResolution')] / prot.MeasYaps[
                ('sSliceArray', 'asSlice', '0', 'dReadoutFOV')]
            rstraj *= kmax
    return prot, rstraj
