r"""
Segy file format manipulating utilities

* Uses the SEG-Y rev 1 standard (binary header 301-302)
* Data sample format code supporting: IEEE and IBM 4 bytes float-point (binary header 25-26)
* Only support writing segy files with IEEE 4 bytes float for data sample format 
* Not supporting different trace lengths (binary header 303-304)
* Not supporting Extended Textual File Header records following the Binary Header (binary header 305-307) 
* Not supporting Big Endian conversion

* using ctypes structs is expect to improve performance using also readinto()

"""

import numpy as np
import struct
import os



# it must be 3200 bytes in size (sample EBCDIC)
EBCDIC = 'This file was created using Fatiando a Terra (http://fatiando.org)'
EBCDIC += (3200-len(EBCDIC))*' '

class BinaryField(object):
    def __init__(self, name, byte, size, fmt, value):                
        self.name = name
        self.byte = byte
        self.size = size
        self.fmt = fmt
        self.value = value

class SegyBH(list):
    r"""
    Segy Binary Header Definition. 400 bytes 
    Uses the SEG-Y rev 1 standard
    """
    def __call__(self, key):
        for bf in self:
            if bf.name == key :
                return bf
    
    def __init__(self):
        super(SegyBH, self).__init__()
        self.append(BinaryField('job_id',                 1, 4, 'i',      0)) # Job identification number
        self.append(BinaryField('line_number',            5, 4, 'i',      0)) # Line number
        self.append(BinaryField('rell_number',            9, 4, 'i',      0)) # Reel number
        self.append(BinaryField('number_traces',         13, 2, 'u',      0)) # Number of data traces per ensemble
        self.append(BinaryField('aux_traces',            15, 2, 'u',      0)) # Number of auxiliary traces per ensemble
        self.append(BinaryField('sample_interval',       17, 2, 'u',   4000)) # @mandatory Sample interval in microseconds
        self.append(BinaryField('sample_interval_org',   19, 2, 'u',   4000)) # Sample interval in microseconds (orig)
        self.append(BinaryField('samples_trace',         21, 2, 'u',      0)) # @mandatory Number of samples per data trace
        self.append(BinaryField('samples_trace_org',     23, 2, 'u',      0)) # Number of samples per data trace (orig)
        self.append(BinaryField('data_format',           25, 2, 'u',      5)) # @mandatory Data sample format code (1 to 8)
        self.append(BinaryField('fold',                  27, 2, 'u',      0)) # Ensemble fold 
        self.append(BinaryField('trace_sort',            29, 2, 'u',      0)) # Trace sorting code
        self.append(BinaryField('vertical_scode',        31, 2, 'u',      0)) # Vertical sum code
        self.append(BinaryField('sweep_fs',              33, 2, 'u',      0)) # Sweep frequency at start (Hz)
        self.append(BinaryField('sweep_fe',              35, 2, 'u',      0)) # Sweep frequency at end
        self.append(BinaryField('sweep_len',             37, 2, 'u',      0)) # Sweep length (ms)
        self.append(BinaryField('sweep_typ',             39, 2, 'u',      0)) # Sweep type code
        self.append(BinaryField('sweep_chn',             41, 2, 'u',      0)) # Trace number sweep channel
        self.append(BinaryField('sweep_stas',            43, 2, 'u',      0)) # Sweep trace taper length in milliseconds at start
        self.append(BinaryField('sweep_stae',            45, 2, 'u',      0)) # Sweep trace taper length in milliseconds at end
        self.append(BinaryField('taper_typ',             47, 2, 'u',      0)) # Taper type
        self.append(BinaryField('corr_traces',           49, 2, 'u',      0)) # Correlated data traces
        self.append(BinaryField('binary_grcv',           51, 2, 'u',      0)) # Binary gain recovered
        self.append(BinaryField('amp_rcvm',              53, 2, 'u',      0)) # Amplitude recovery method
        self.append(BinaryField('meters_feet',           55, 2, 'u',      0)) # Measurement system
        self.append(BinaryField('polarity',              57, 2, 'u',      0)) # Impulse signal polarity
        self.append(BinaryField('vpolarity',             59, 2, 'u',      0)) # Vibratory polarity code
        self.append(BinaryField('unassigned',            61, 240, 'c',    0)) 
        self.append(BinaryField('sgy_revision',         301, 2, 'u',    256)) # SEG Y Format Revision Number 0100 hex
        self.append(BinaryField('fixed_length_traces',  303, 2, 'u',      1)) # Fixed length trace flag (1 or 0)
        self.append(BinaryField('extended_text_header', 305, 2, 'u',      0)) # Number of 3200-byte, Extended Textual File Header 
        # records following the  Binary Header 

class SegyTH(list):
    r"""
    Binary Trace Header Definition. 240 bytes 
    Uses the SEG-Y rev 1 standard.
    """
    def __call__(self, key):
        for bf in self:
            if bf.name == key:
                return bf
    
    def _size(self):
        r"""
        must be less than 240 bytes
        """
        sizebytes = 0
        for bf in self:
            sizebytes += bf.size

    def __init__(self):
        super(SegyTH, self).__init__()
        self.append(BinaryField('trace_number_line',       1,   4, 'i',     0))  # Trace sequence number within line
        self.append(BinaryField('trace_number_reel',       5,   4, 'i',     0))  # Trace sequence number within reel
        self.append(BinaryField('FFID',                    9,   4, 'i',     0))  # FFID - Original field record number
        self.append(BinaryField('trace_number_field',      13,  4, 'i',     0))  # Trace number within field record
        self.append(BinaryField('SP',                      17,  4, 'i',     0))  # SP - energy source point number
        self.append(BinaryField('ensemble_number',         21,  4, 'i',     0))  # Ensemble number (CDP, CMP, CRP, etc)
        self.append(BinaryField('trace_number',            25,  4, 'i',     0))  # Trace number within ensemble
        self.append(BinaryField('trace_id_code',           29,  2, 'u',     0))  # Trace Identification code (0 = Unknown, 1= Seismic data, 2 = Dead, etc)
        self.append(BinaryField('n_vertically_summed',     31,  2, 'u',     0))  # Number of vertically summed traces
        self.append(BinaryField('n_horizontally_stacked',  33,  2, 'u',     0))  # Number of horizontally stacked traces
        self.append(BinaryField('data_use',                35,  2, 'u',     0))  # Data use (1=production, 2=test)
        self.append(BinaryField('offset',                  37,  4, 'i',     0))  # Offset (distance from source point to receiver group)
        self.append(BinaryField('rcvr_grp_elevation',      41,  4, 'i',     0))  # Receiver group elevation
        self.append(BinaryField('surface_elevation_src',   45,  4, 'i',     0))  # Surface elevation at source
        self.append(BinaryField('src_depth',               49,  4, 'i',     0))  # Source depth below surface (>0)
        self.append(BinaryField('datum_elevation_rcvr',    53,  4, 'i',     0))  # Datum elevation at receiver group
        self.append(BinaryField('datum_elevation_src',     57,  4, 'i',     0))  # Datum elevation at source
        self.append(BinaryField('water_depth_src',         61,  4, 'i',     0))  # Water depth at source
        self.append(BinaryField('water_depth_grp',         65,  4, 'i',     0))  # Water depth at group
        self.append(BinaryField('scalar_elevation_depths', 69,  2, 'u',     0))  # Scalar to be applied to all elevations and depths
        self.append(BinaryField('scalar_coordinates',      71,  2, 'u',     0))  # Scalar to be applied to all coordinates
        self.append(BinaryField('src_X_coord',             73,  4, 'i',     0))  # Source X-coordinate
        self.append(BinaryField('src_Y_coord',             77,  4, 'i',     0))  # Source Y-coordinate
        self.append(BinaryField('grp_X_coord',             81,  4, 'i',     0))  # Group X-coordinate
        self.append(BinaryField('grp_Y_coord',             85,  4, 'i',     0))  # Group Y-coordinate
        self.append(BinaryField('coord_units',             89,  2, 'u',     0))  # Coordinate units
        self.append(BinaryField('weathering_vel',          91,  2, 'u',     0))  # Weathering velocity
        self.append(BinaryField('subweathering_vel',       93,  2, 'u',     0))  # Subweathering velocity
        self.append(BinaryField('uphole_time_src',         95,  2, 'u',     0))  # Uphole time at source (ms)
        self.append(BinaryField('uphole_time_grp',         97,  2, 'u',     0))  # Uphole time as group (ms)
        self.append(BinaryField('src_static',              99,  2, 'u',     0))  # Source static corretion (ms)
        self.append(BinaryField('grp_static',              101, 2, 'u',     0))  # Group static corretion (ms)
        self.append(BinaryField('total_static',            103, 2, 'u',     0))  # Total static applied (ms)
        self.append(BinaryField('lag_time_A',              105, 2, 'u',     0))  # Lag time A
        self.append(BinaryField('lag_time_B',              107, 2, 'u',     0))  # Lag time B
        self.append(BinaryField('delay_recording_time',    109, 2, 'u',     0))  # Delay recording time
        self.append(BinaryField('mute_start',              111, 2, 'u',     0))  # Mute time - start
        self.append(BinaryField('mute_end',                113, 2, 'u',     0))  # Mute time - end
        self.append(BinaryField('number_of_samples',       115, 2, 'u',     0))  # Number of samples in this trace
        self.append(BinaryField('sample_interval',         117, 2, 'u',  4000))  # Sample interval
        self.append(BinaryField('gain_type',               119, 2, 'u',     0))  # Gain type of field instruments
        self.append(BinaryField('inst_gain_constant',      121, 2, 'u',     0))  # Instrument gain constant
        self.append(BinaryField('inst_ini_gain',           123, 2, 'u',     0))  # Instrument initial or early gain
        self.append(BinaryField('correlated',              125, 2, 'u',     0))  # Correlated (0=no, 1=yes)
        self.append(BinaryField('sweep_freq_start',        127, 2, 'u',     0))  # Sweep frequency at start
        self.append(BinaryField('sweep_freq_end',          129, 2, 'u',     0))  # Sweep frequency at end
        self.append(BinaryField('sweep_length',            131, 2, 'u',     0))  # Sweep length
        self.append(BinaryField('sweep_type',              133, 2, 'u',     0))  # Sweep type
        self.append(BinaryField('sweep_tr_tpr_len_start',  135, 2, 'u',     0))  # Sweep trace taper length at start
        self.append(BinaryField('sweep_tr_tpr_len_end',    137, 2, 'u',     0))  # Sweep trace taper length at end
        self.append(BinaryField('taper_type',              139, 2, 'u',     0))  # Taper type
        self.append(BinaryField('alias_freq',              141, 2, 'u',     0))  # Alias filter frequency
        self.append(BinaryField('alias_slope',             143, 2, 'u',     0))  # Alias filter slope
        self.append(BinaryField('notch_freq',              145, 2, 'u',     0))  # Notch filter frequency
        self.append(BinaryField('notch_slope',             147, 2, 'u',     0))  # Notch filter slope
        self.append(BinaryField('lowcut_freq',             149, 2, 'u',     0))  # Lowcut filter frequency
        self.append(BinaryField('highcut_freq',            151, 2, 'u',     0))  # Highcut filter frequency
        self.append(BinaryField('lowcut_slope',            153, 2, 'u',     0))  # Lowcut filter slope
        self.append(BinaryField('highcut_slope',           155, 2, 'u',     0))  # Highcut filter slope
        self.append(BinaryField('year',                    157, 2, 'u',     0))  # Year data recorded
        self.append(BinaryField('day',                     159, 2, 'u',     0))  # Day of year 
        self.append(BinaryField('hour',                    161, 2, 'u',     0))  # Hour of day
        self.append(BinaryField('minute',                  163, 2, 'u',     0))  # Minute of hour
        self.append(BinaryField('second',                  165, 2, 'u',     0))  # Second of minute
        self.append(BinaryField('time_basis',              167, 2, 'u',     0))  # Time basis code
        self.append(BinaryField('weighting_factor',        169, 2, 'u',     0))  # Trace weighting factor
        self.append(BinaryField('geophone_grp_n_roll_p_1', 171, 2, 'u',     0))  # Geophone group number of roll switch position one
        self.append(BinaryField('geophone_grp_n_tr_1',     173, 2, 'u',     0))  # Geophone group number of trace one
        self.append(BinaryField('geophone_grp_n_last_tr',  175, 2, 'u',     0))  # Geophone group number of last trace
        self.append(BinaryField('gap_size',                177, 2, 'u',     0))  # Gap size (# of groups dropped)
        self.append(BinaryField('over_travel_taper',       179, 2, 'u',     0))  # Over travel associated with taper at beginning or end of line
        self.append(BinaryField('CDP_X',                   181, 4, 'i',     0))  # CDP X coordinate
        self.append(BinaryField('CDP_Y',                   185, 4, 'i',     0))  # CDP Y coordinate
        self.append(BinaryField('inline',                  189, 4, 'i',     0))  # Inline number
        self.append(BinaryField('crossline',               193, 4, 'i',     0))  # Crossline number
        self.append(BinaryField('shot_point_number',       197, 4, 'i',     0))  # Shot point number
        self.append(BinaryField('shot_point_scalar',       201, 4, 'i',     0))  # Shot point scalar
        self.append(BinaryField('trc_value_unit',          203, 2, 'u',     0))  # Trace value measurement unit
        self.append(BinaryField('transduction_const_mant', 205, 4, 'i',     0))  # Transduction constant (mantissa)
        self.append(BinaryField('transduction_const_pow',  209, 2, 'u',     0))  # Transduction constant (power)
        self.append(BinaryField('transduction_unit',       211, 2, 'u',     0))  # Transduction unit
        self.append(BinaryField('identifier',              213, 2, 'u',     0))  # Device/trace identifier
        self.append(BinaryField('scalar_time',             215, 2, 'u',     0))  # Scalar to be applied to bytes 95-114 (basically concerning times)
        self.append(BinaryField('source_tp_orient',        217, 2, 'u',     0))  # Source type/orientation
        self.append(BinaryField('source_energy_direction', 219, 4, 'i',     0))  # Source energy direction
        self.append(BinaryField('source_measurement_mant', 225, 4, 'i',     0))  # Source measurement (mantissa)
        self.append(BinaryField('source_measurement_pow',  229, 2, 'u',     0))  # Source measurement (power)


def _i2pack(integer):
    r"""
    pack the int (integer) value as an integer 2 (signed)
    """
    return struct.pack('h', integer)

def _i2(strpackedbin, byte):
    r"""
    Returns the 2-byte integer from <byte> position in strpackedbin
    * strpackedbin : string containing packed binary data
    """
    return( int(strpackedbin[(byte-1):(byte-1)+2].encode('hex'),16)  )

def _i4pack(integer):
    r"""
    pack the int (integer) value as an integer 4 (signed)
    """
    return struct.pack('i', integer)

def _i4(strpackedbin, byte): 
    r"""
    Returns the 4-byte integer from <byte> position in strpackedbin
    * strpackedbin : string containing packed binary data
    """
    return( int(strpackedbin[(byte-1):(byte-1)+4].encode('hex'),16)  )
    #return struct.unpack("i", strpackedbin[(byte-1):(byte-1)+4])  

def _f4(strpackedbin, byte):
    r"""
    Returns the 4-byte floating point from <byte> position in strpackedbin
    
    * strpackedbin : string containing packed binary data containing the float number
    """
    return struct.unpack("f", strpackedbin[(byte-1):(byte-1)+4])[0]

def _f4pack(_float):
    r"""
    pack the float32 (_float) value as an float 4 little endian
    """
    return struct.pack("f", _float)

def _f4ibm(strpackedbin, byte):
    r"""
    Converts an IBM floating point number (string containing packed binary data) into IEEE format
    (string containing packed binary data).
    
    * strpackedbin : string containing packed binary data containing the float number
    
    :param: ibm - 32 bit unsigned integer: unpack('>L', f.read(4))
    """

    strpackedbin = strpackedbin[(byte-1):(byte-1)+4]

    ibm = struct.unpack('>L', strpackedbin)[0]
    sign = ibm >> 31 & 0x01
    exponent = ibm >> 24 & 0x7f
    mantissa = ibm & 0x00ffffff
    mantissa = (mantissa * 1.0) / pow(2, 24)
    ieee = (1 - 2 * sign) * mantissa * pow(16, exponent - 64)
    return ieee

def _str2floats(sampleformat, strpackedbin):
    r"""
    Create an numpy array of floats (4 bytes)
    from a string strpackedbin using the correct format for reading 
    
    * sampleformat  : 1 (IBM 4 bytes float) or 5 (IEEE 4 bytes float)
    * strpackedbin : string containing packed binary data containing float32 values
    
    """
    #samples = np.empty(0,dtype=np.float32) the bellow is faster (list)
    samples = []

    if sampleformat == 1:  # IBM float
        for i in range(0, len(strpackedbin), 4):
            samples.append(_f4ibm(strpackedbin, i+1))
    elif sampleformat == 5:  # IEEE float
        for i in range(0, len(strpackedbin), 4):
            samples.append(_f4(strpackedbin, i+1))
    return samples


class Segy(object):
    r"""
    Segy file format utility class.
    
    Writing and reading support.    
    """
    def __init__(self, filename, traces=None, ebcdic=EBCDIC, bheader=SegyBH()):
        r"""
        Create a segy class. 
        First assign the path filename to the segy file.
        
        For reading a segy file, use as:

            with Segy('mysegyfile.sgy') as sgy:
                # do whatever you want with the contents
        
        For writing a segy file, use as:
        
            # create ebcdic string
            # create binary header (can use default SegyBH)
            traces = []
            # create list of traces
            Segy('mysegyfile.sgy', ebcdicstr, binheader, traces)
            Segy.write()
        
        * filename : segy file name path
        * traces   : list of Trace's object  
        * ebcdic   : ascii or ebcdic segy header 3200 bytes
        * bheader  : segy binary header object SegyBH
        """
        self.filename = filename

        if isinstance(traces, list):
            if isinstance(traces[0], Trace):
                self._traces = traces
                self.bheader = bheader
                self.ebcdic = ebcdic


    def __enter__(self):
        r"""
        Opens the file.
        The file will remain open for reading traces. 
        Only Binary and Ascii headers will be read.
        """
        self.file = open(self.filename, "rb")
        
        # read general segy headers
        filebytes = self.file.read(3200+400)
        
        # Ebcdic or text header 3200 bytes
        self.ebcdic = filebytes[:3200].decode('EBCDIC-CP-BE')
        # 400 bytes binary header
        self.bheader = self._getbinaryheader(filebytes[3200:3600])
        
        if not self.bheader('fixed_length_traces').value == 1:
            #raise Exception('file not supported')
            pass 
        if not self.bheader('sgy_revision').value == 256:
            #raise Exception('file not supported')
            pass 
        if not self.bheader('extended_text_header').value == 0:
            #raise Exception('file not supported')
            pass
        # IBM and IEEE support 5/1        
        if not ( self.bheader('data_format').value == 5 or self.bheader('data_format').value == 1):
            #raise Exception('file not supported')
            pass
        # for not extended text header and fixed length trace gets
        # the number of traces based on 4 byte sample data format
        self.filesize = os.stat(self.filename).st_size
        
        self.samplestrace = self.bheader('samples_trace').value
        self.numbertraces = (self.filesize-(3200+400))/(self.samplestrace*4+240)
        self.sampleformat = self.bheader('data_format').value
        
        self.samplerate = self.bheader('sample_interval').value
        
        return self

    def __exit__(self, eType, eValue, eTrace):
        r""" close the file """
        self.file.close()

    def _getbinaryheader(self, strpackedbin):
        r""" 
        Loads the binary header (string containing packed binary data) into a custom list for easy access. 
        """
        segybh = SegyBH() # custom list for storing binary header information
        for bf in segybh: # unpack based on information of field
            if bf.size == 2:
                bf.value = _i2(strpackedbin, bf.byte)
            elif bf.size == 4:
                bf.value = _i4(strpackedbin, bf.byte)
        
        return segybh

    def traces(self, key):
        r"""
        Return a especific trace from file (reading mode)
        
        Return a especific trace from list of traces (writing mode)
         
        Max traces defined by self.numbertraces
        
        * key : trace index 
        """
        if isinstance(self.file, file) and not self.file.closed:
            self.file.seek((3200+400)+(self.samplestrace*4+240)*key, 0) # 0 from begging of file
            tracestrpackedbin = self.file.read((self.samplestrace*4+240))
            return Trace(self.bheader('samples_trace').value, self.bheader('data_format').value, tracestrpackedbin)
        else:
            return self._traces[key]

    def _pack(self):
        r"""
        Pack this file as a string containing packed binary data
        """
        strpackedbin = str()
        
        # ebcdic packing
        strpackedbin += struct.pack('3200c', *self.ebcdic)
        
        # segy binary header
        for bf in self.bheader: # pack based on information of field
            if bf.size == 2:
                strvalue = _i2pack(bf.value)
            elif bf.size == 4:
                strvalue =_i4pack(bf.value)
            elif bf.size == 240: # empty block
                strvalue = struct.pack('240c', *(240*' '))
            strpackedbin += strvalue
        
        # binary header SegyBH contains information until byte 306,
        # padding 94 bytes more white spaces
        strpackedbin += struct.pack('94c', *(94*' '))
        
        # pack all traces inside
        for trace in self._traces:
            strpackedbin += trace._pack()
            
        return strpackedbin  
    
    def write(self):
        r"""
        open the file (write + binary) write in it and closes 
        """
        sfile = open(self.filename, "wb")
        sfile.write(self._pack())
        sfile.close()

class Trace(object):
    r"""
    Segy Trace utility class.
    """
    def __init__(self, numbersamples, sampleformat=5, tracestrpackedbin=None, theader=None):
        r"""
        Create a trace object
        
        * numbersamples : number of samples for this trace
        * sampleformat  : 1 (IBM 4 bytes float) or 5 (IEEE 4 bytes float)
        
        when writing a segy use:
        * theader   : a custom list containg valid fields for this trace header
        
        when reading a segy use:
        * tracestrpackedbin    : a string containing packed binary data containg every byte data from one segy trace
        loaded from a file (Segy class)
        """

        # reading a trace from file
        if isinstance(tracestrpackedbin, str): 
            # Convert every sample to a float array using correct format
            self.samples = _str2floats(sampleformat, tracestrpackedbin[240:])
            # get the trace header values
            self.theader = self._gettraceheader(tracestrpackedbin[:240])
        # creating a trace to write in file         
        elif isinstance(theader, str):
            self.theader = theader
            # all trace samples start with 0.0 float value
            #self.samples = np.zeros(self.theader['nsamples'].value,dtype=np.float32)
            self.samples = []
        else :
            raise Exception("see usage (docstring)")
        

# Consistency check?        
#         if (len(self.samples) != self.theader['nsamples']):
#             pass
#             #raise exception? npts from header differs from real npts
       
    def numbersamples(self):
        r"""
        return the real number of samples for this trace
        """
        return len(self.samples)
    
    def __getitem__(self, key):
        r"""
         return a sample value at trace index position key
         max defined by self.numbersamples
        """
        return self.samples[key]
    
    def __setitem__(self, key, item):
        r"""
         set the sample value at trace index position key as item
         max value can be changed
         
         * item : must be a np.float32
        """
        self.samples[key] = item


    def _gettraceheader(self, strpackedbin):
        r"""
        Load information from strpackedbin (string containing packed binary data) into a custom list for easy access. 
        """
        segyth = SegyTH() # custom list for storing binary header information
        for bf in segyth: # unpack based on information of field
            if bf.size == 2:
                bf.value = _i2(strpackedbin, bf.byte)
            elif bf.size == 4:
                bf.value = _i4(strpackedbin, bf.byte)
        
        return segyth

    def _pack(self):
        r"""
        Pack this trace as a string containing packed binary data
        """
        strpackedbin = str()
        
        for bf in self.theader: # pack based on information of field
            strvalue = _i2pack(bf.value) if bf.size == 2 else _i4pack(bf.value)
            strpackedbin += strvalue
        
        # trace header SegyTH contains information until byte 230,
        # padding 10 bytes more white space
        strpackedbin += struct.pack('10c', *(10*' '))
        
        # now pack the samples float 32 IEEE
        strpackedbin += struct.pack('%df' % len(self.samples), *self.samples)
            
        return strpackedbin

    def __repr__(self):
        tmp = "Trace " + str(self.theader('trace_number')) + "; "
        tmp += "Ensemble " + str(self.theader('ensemble')) + "; "
        tmp += "Offset " + str(self.theader('offset')) + "; "
        tmp += "Samples " + str(self.theader('nsample'))
        return tmp
    
    def summary(self):  #necessary?
        summary = \
        {"Number of samples": self.theader('nsamples'), \
        "Offset": self.theader('offset'), \
        "Ensemble": self.theader('CDP'), \
        "FFID": self.theader('FFID') }
        return (summary)
    

    
