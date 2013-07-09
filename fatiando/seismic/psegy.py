#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from numpy import *
import struct
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.widgets import Cursor


class SeismicSection:
    """SeismicSection Class
       Used mainly to group traces into one object.
       Has tools such as sorting, stacking (?), muting (?)
    """
    
    def __init__(self, traces, binaryheader):
        pass


class Trace:
    """Trace class."""

    def __init__(self, bintrace, format=1):
        """
        Initialization. Uses the raw binary trace <bintrace> to extract and populate both data and header members of the class
        """
        self._binhdr = bintrace[:240]

        self.offset =    self.__i4(37)
        self.cdp =       self.__i4(21)
        self.trcnumber = self.__i4(25)
        self._format = format
        
        self.data = self.__convertdata(bintrace[240:])

        self.header = self.__loadheader()

        self.nsamples = len(self.data)

        
        if (len(self.data) != self.__i2(115)):
            pass
            #raise exception? npts from header differs from real npts
        else:
            self.npts =  self.__i2(115)
        
        
    def __repr__(self):
         tmp = "Trace " + str(self.header['trace_number']) + "; "
         tmp += "Ensemble " + str(self.header['ensemble']) + "; "
         tmp += "Offset " + str(self.header['offset']) + "; "
         tmp += "Samples " + str(self.header['nsample'])
         return tmp
    
    def summary(self):  #necessary?
        summary = \
        {"Number of samples": self.nsamples, \
        "Offset": self.offset, \
        "Ensemble": self.cdp, \
        "FFID": self.header['FFID'] }
        # what else?
        
        return(summary)
    
    def __convertdata(self, trc):
        trctmp = []

        if (self._format == 1):  # IBM float
            for i in range(0, len(trc), 4):
                trctmp.append(ibm2ieee(trc[i:i+4]))
            return trctmp
        elif (self._format == 5):  # IEEE float
            for i in range(0, len(trc), 4):
                trctmp.append(struct.unpack(">f",trc[i:i+4])[0])
            return trctmp
        
    
    def __i2(self, byte):
        """Returns the 2-byte integer from <byte> position in self.binhedr"""
        return( int(self._binhdr[(byte-1):(byte-1)+2].encode('hex'),16)  )
        
    def __i4(self, byte): 
        """Returns the 4-byte integer from <byte> position in self.binhedr"""
        return( int(self._binhdr[(byte-1):(byte-1)+4].encode('hex'),16)  )
    
    def bheader(self, byte, size): 
        """Returns the <size>-byte integer from <byte> position in self.binhedr"""
        return( int(self._binhdr[(byte-1):(byte-1)+size].encode('hex'),16)  )
    
    
    def __loadheader(self):
        """Load information from _binhdr into a dictionary for easy access. This dictionary uses the SEG-Y rev 1 standard"""
        tmp = {}
        
        tmp['trace_number_line']        = self.__i4(1)    # Trace sequence number within line
        tmp['trace_number_reel']        = self.__i4(5)    # Trace sequence number within reel
        tmp['FFID']                     = self.__i4(9)    # FFID - Original field record number
        tmp['trace_number_field']       = self.__i4(13)   # Trace number within field record
        tmp['SP']                       = self.__i4(17)   # SP - energy source point number

        tmp['ensemble']                 = self.__i4(21)   # Ensemble number (CDP, CMP, CRP, etc)
        tmp['CDP']                      = self.__i4(21)   # Ensemble number (CDP, CMP, CRP, etc)
        tmp['CMP']                      = self.__i4(21)   # Ensemble number (CDP, CMP, CRP, etc)
        tmp['CRP']                      = self.__i4(21)   # Ensemble number (CDP, CMP, CRP, etc)
        
        tmp['trace_number']             = self.__i4(25)   # Trace number within ensemble
        tmp['trace_id_code']            = self.__i2(29)   # Trace Identification code (0 = Unknown, 1= Seismic data, 2 = Dead, etc)
        tmp['n_vertically_summed']      = self.__i2(31)   # Number of vertically summed traces
        tmp['n_horizontally_stacked']   = self.__i2(33)   # Number of horizontally stacked traces
        tmp['data_use']                 = self.__i2(35)   # Data use (1=production, 2=test)
        tmp['offset']                   = self.__i4(37)   # Offset (distance from source point to receiver group)
        tmp['rcvr_grp_elevation']       = self.__i4(41)   # Receiver group elevation
        tmp['surface_elevation_src']    = self.__i4(45)   # Surface elevation at source
        tmp['src_depth']                = self.__i4(49)   # Source depth below surface (>0)
        tmp['datum_elevation_rcvr']     = self.__i4(53)   # Datum elevation at receiver group
        tmp['datum_elevation_src']      = self.__i4(57)   # Datum elevation at source
        tmp['water_depth_src']          = self.__i4(61)   # Water depth at source
        tmp['water_depth_grp']          = self.__i4(65)   # Water depth at group
        tmp['scalar_elevation_depths']  = self.__i2(69)   # Scalar to be applied to all elevations and depths
        tmp['scalar_coordinates']       = self.__i2(71)   # Scalar to be applied to all coordinates
        tmp['src_X_coord']              = self.__i4(73)   # Source X-coordinate
        tmp['src_Y_coord']              = self.__i4(77)   # Source Y-coordinate
        tmp['grp_X_coord']              = self.__i4(81)   # Group X-coordinate
        tmp['grp_Y_coord']              = self.__i4(85)   # Group Y-coordinate
        tmp['coord_units']              = self.__i2(89)   # Coordinate units
        tmp['weathering_vel']           = self.__i2(91)   # Weathering velocity
        tmp['subweathering_vel']        = self.__i2(93)   # Subweathering velocity
        tmp['uphole_time_src']          = self.__i2(95)   # Uphole time at source (ms)
        tmp['uphole_time_grp']          = self.__i2(97)   # Uphole time as group (ms)
        tmp['src_static']               = self.__i2(99)   # Source static corretion (ms)
        tmp['grp_static']               = self.__i2(101)  # Group static corretion (ms)
        tmp['total_static']             = self.__i2(103)  # Total static applied (ms)
        tmp['lag_time_A']               = self.__i2(105)  # Lag time A
        tmp['lag_time_B']               = self.__i2(107)  # Lag time B
        tmp['delay_recording_time']     = self.__i2(109)  # Delay recording time
        tmp['mute_start']               = self.__i2(111)  # Mute time - start
        tmp['mute_end']                 = self.__i2(113)  # Mute time - end
        
        tmp['number_of_samples']        = self.__i2(115)  # Number of samples in this trace
        tmp['nsamples']                 = self.__i2(115)  # Number of samples in this trace
        tmp['nsample']                  = self.__i2(115)  # Number of samples in this trace
        
        tmp['sample_interval']          = self.__i2(117)  # Sample interval
        tmp['gain_type']                = self.__i2(119)  # Gain type of field instruments
        tmp['inst_gain_constant']       = self.__i2(121)  # Instrument gain constant
        tmp['inst_ini_gain']            = self.__i2(123)  # Instrument initial or early gain
        tmp['correlated']               = self.__i2(125)  # Correlated (0=no, 1=yes)
        tmp['sweep_freq_start']         = self.__i2(127)  # Sweep frequency at start
        tmp['sweep_freq_end']           = self.__i2(129)  # Sweep frequency at end
        tmp['sweep_length']             = self.__i2(131)  # Sweep length
        tmp['sweep_type']               = self.__i2(133)  # Sweep type
        tmp['sweep_tr_tpr_len_start']   = self.__i2(135)  # Sweep trace taper length at start
        tmp['sweep_tr_tpr_len_end']     = self.__i2(137)  # Sweep trace taper length at end
        tmp['taper_type']               = self.__i2(139)  # Taper type
        tmp['alias_freq']               = self.__i2(141)  # Alias filter frequency
        tmp['alias_slope']              = self.__i2(143)  # Alias filter slope
        tmp['notch_freq']               = self.__i2(145)  # Notch filter frequency
        tmp['notch_slope']              = self.__i2(147)  # Notch filter slope
        tmp['lowcut_freq']              = self.__i2(149)  # Lowcut filter frequency
        tmp['highcut_freq']             = self.__i2(151)  # Highcut filter frequency
        tmp['lowcut_slope']             = self.__i2(153)  # Lowcut filter slope
        tmp['highcut_slope']            = self.__i2(155)  # Highcut filter slope
        tmp['year']                     = self.__i2(157)  # Year data recorded
        tmp['day']                      = self.__i2(159)  # Day of year 
        tmp['hour']                     = self.__i2(161)  # Hour of day
        tmp['minute']                   = self.__i2(163)  # Minute of hour
        tmp['second']                   = self.__i2(165)  # Second of minute
        tmp['time_basis']               = self.__i2(167)  # Time basis code
        tmp['weighting_factor']         = self.__i2(169)  # Trace weighting factor
        tmp['geophone_grp_n_roll_p_1']  = self.__i2(171)  # Geophone group number of roll switch position one
        tmp['geophone_grp_n_tr_1']      = self.__i2(173)  # Geophone group number of trace one
        tmp['geophone_grp_n_last_tr']   = self.__i2(175)  # Geophone group number of last trace
        tmp['gap_size']                 = self.__i2(177)  # Gap size (# of groups dropped)
        tmp['over_travel_taper']        = self.__i2(179)  # Over travel associated with taper at beginning or end of line
        tmp['CDP_X']                    = self.__i4(181)  # CDP X coordinate
        tmp['CDP_Y']                    = self.__i4(185)  # CDP Y coordinate
        
        tmp['inline']                   = self.__i4(189)  # Inline number
        tmp['iline']                    = self.__i4(189)  # Inline number
        tmp['crossline']                = self.__i4(193)  # Crossline number
        tmp['xline']                    = self.__i4(193)  # Crossline number
        
        tmp['shot_point_number']        = self.__i4(197)  # Shot point number
        tmp['shot_point_scalar']        = self.__i4(201)  # Shot point scalar
        tmp['trc_value_unit']           = self.__i2(203)  # Trace value measurement unit
        tmp['transduction_const_mant']  = self.__i4(205)  # Transduction constant (mantissa)
        tmp['transduction_const_pow']   = self.__i2(209)  # Transduction constant (power)
        tmp['transduction_unit']        = self.__i2(211)  # Transduction unit
        tmp['identifier']               = self.__i2(213)  # Device/trace identifier
        tmp['scalar_time']              = self.__i2(215)  # Scalar to be applied to bytes 95-114 (basically concerning times)
        tmp['source_tp_orient']         = self.__i2(217)  # Source type/orientation

        #tmp['source_energy_direction']  = self.__i4(219)  # Source energy direction
        
        tmp['source_measurement_mant']  = self.__i4(225)  # Source measurement (mantissa)
        tmp['source_measurement_pow']   = self.__i2(229)  # Source measurement (power)
        
        return tmp
        

def ibm2ieee(ibm_str):
    """
    Converts an IBM floating point number into IEEE format.
    :param: ibm - 32 bit unsigned integer: unpack('>L', f.read(4))
    """

    ibm = struct.unpack('>L', ibm_str)[0]
    sign = ibm >> 31 & 0x01
    exponent = ibm >> 24 & 0x7f
    mantissa = ibm & 0x00ffffff
    mantissa = (mantissa * 1.0) / pow(2, 24)
    ieee = (1 - 2 * sign) * mantissa * pow(16, exponent - 64)
    return ieee

def i4(buffer, byte): 
    """Retorna o valor inteiro de 4 bytes da posição byte, contido em buffer"""
    return( int(buffer[(byte-1):(byte-1)+4].encode('hex'),16)  )
    

def i2(buffer, byte): 
    """Retorna o valor inteiro de 2 bytes da posição byte, contido em buffer"""
    return( int(buffer[(byte-1):(byte-1)+2].encode('hex'),16)  )    

def gettrace(buffer):
    """Retorna traco de numero trace_n em forma de lista (ja convertido para float)"""
    
    traco = []
    for i in range(240, len(buffer), 4):
        #traco.append(  struct.unpack('f', buffer[i:i+4])[0]  )
        traco.append(ibm2ieee2(buffer[i:i+4]))
    
    return traco
    
def opensegy(filename):
    """Abre arquivo segy, retornando header textual, header binario, tracos com seus respectivos headers, numero de tracos, numero de amostras por traco e taxa de amostragem
    
    input:
        filename = string contendo nome do arquivo
        
    output:
    
        (texhdr, binhdr, data, ntrace, nsample, dt), onde:
        
        texhdr = header textual
        binhdr = header binario
        data = tracos com seus respectivos headers
        ntrace = numero de tracos
        nsample = numero de amostras
        dt = taxa de amostragem
    """
    
    f = open(filename, "rb")
    file = str()
    try:
        byte = f.read(1)
        while byte != "":
            file = file + str(byte)
            byte = f.read(1)
    finally:
        f.close()

    #convertendo de EBCDIC para ASCII
    textual = file[:3200].decode('EBCDIC-CP-BE')
    binhead = file[3200:3600]

    #corta header binario e ebcdic
    data = file[3600:]

    #informacoes do header
    nsamples = i2(binhead,21)     #no. de amostras
    dt = i2(binhead,17) / 1000    #dt em milisegundos

    ntrace = len(data) / ((nsamples*4) +240)
    
    return (textual, binhead, data, ntrace, nsamples, dt)

    
def __normalization_factors(traces, ntraces, mode):
    """Normalize traces' amplitude values.
    
    Input:
        traces - array/matrix containing the traces' samples
        mode   - what kind of normalization will be performed
    Output:
        returns a vector containing the normalization factors to be used ijn the normalization process
    
    
    Modes are:
        0 - no normalization (returns vector filled with 1's)
        1 - single trace normalization
        2 - all traces normalization
    
    Normalizing a trace means all of it's amplitudes will stay within the [-1,+1] interval. 
    The normalization process is basically devide all samples by the maximum absolute sample value.
    If the chosen mode is 2 (all traces normalization), the maximum value is picked from all traces, and all traces are normalized using this single value.
    In this particular case, the returned vector will contain the same value in all positions.

    """
    
    factors = []
    
    #no normalization
    if (mode == 0):
        for i in range(0, ntrace, 1):
            factors.append(1)
    
    #single trace normalization
    if (mode == 1):
    
        for i in range(0, ntrace, 1):
            factors.append(max(absolute(asarray(traces[i]))))
    
    #all traces normalization
    if (mode == 2):
    
        tmpmax = []
        for i in range(0, ntrace, 1):
            tmpmax.append(max(absolute(asarray(traces[i]))))
            factors.append(1)
        
    
    return factors
    
    
    
def displaywiggle(traces, ntraces, nsamples, dt, scale=1):
    """
    
    
    
    """

    #numero maximo de tracos por display
    maxtraces = 500
    if (ntraces < 500):
        maxtraces = ntraces
    
    t = range(0, nsample*dt, dt)

    plt.ylim(nsample*dt,0)
    plt.xlim(-1,maxtraces)
    

    plt.ylabel('t (ms)')
    t = asarray(t)
    
    f = __normalization_factors(traces, ntrace, 2)
    #f = np.ones(ntrace)
    
    for i in range(0, maxtraces, 1):
        tr = asarray(traces[i])

        tr = (tr / f[i]) * scale
        
#        plt.plot((i+1)+tr, t, 'k')
#        plt.fill_betweenx(t, (i+1), (i+1)+tr, tr>=0, color='k');
        plt.plot(i+tr, t, 'k')
        plt.fill_betweenx(t, i, i+tr, tr>=0, color='k');

    #plt.show()
    

def displaycontour(traces, ntraces, nsamples, dt):
    """
    
    
    
    """

    #numero maximo de tracos por display
    maxtraces = 5000
    if (ntraces < 5000):
        maxtraces = ntraces
    

    plt.ylim(nsamples*dt,0)
    plt.xlim(-1,maxtraces)
    
    maxval = -1.0;
    minval = 1.0;
    
    matx = zeros( (maxtraces,nsamples) )
    maty = zeros( (maxtraces,nsamples) )
    for i in linspace(0, maxtraces-1, maxtraces):
        for j in linspace(nsamples-1, 0, nsamples):

            i = int(i)
            j = int(j)
            #print i, j
            
            matx[i][j] = i
            maty[i][j] = j*dt
            
            if (traces[i][j] < minval):
                minval = traces[i][j]
            if (traces[i][j] > maxval):
                maxval = traces[i][j]
		
    print "shape x:", shape(matx)
    print "shape y:", shape(maty)
    print "shape traces:", shape(traces)
    plt.ylabel('t (ms)')
    
    disc = (maxval-minval)*0.05
    
    plt.contourf(matx, maty, traces, levels=linspace(minval+disc,maxval-disc,100), cmap=cm.Greys)


    #plt.show()
############################################################################


def clicked(event):
    print "x: ", event.xdata
    print "y: ", event.ydata
    
    plot_avo(event.ydata)
    

def motion(event):
    print "(x,y): ", event.xdata, event.ydata

    
def plot_avo(y):
    #plt.subplot(212)
    
    index = int(y/dt)
    print index
    #plt.cla()
    
    plt.plot(offsets, avo[index], 'ks')
    plt.plot(offsets, avo[index], '-')
    
    plt.xlim(min(offsets), max(offsets))
    
    
filename = str("sub_vol.sgy")

(texthdr, binhead, data, ntrace, nsample, dt) = opensegy(filename)


#trace size
tsize = ((nsample*4) +240)

format = i2(binhead,25)

tracos = []
offsets = []
for i in range(0, ntrace, 1):

    tracotmp = Trace(data[i * tsize: (i+1)*tsize], format)
    offsets.append(tracotmp.offset)
    trtmp = tracotmp.data
    tracos.append(trtmp)

    
gather = []
for i in range(0, ntrace, 1):
    gather.append(Trace(data[i * tsize: (i+1)*tsize], format))
    

#Escala de cinza
displaycontour(tracos, ntrace, nsample, dt)

#Wiggle
displaywiggle(tracos, ntrace, nsample, dt, 0.5)



plt.show()

