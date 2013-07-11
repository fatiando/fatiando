r"""
Segy file format manipulating utilities

* Uses the SEG-Y rev 1 standard (binary header 301-302)
* Data sample format code supporting: IEEE and IBM 4 bytes float-point (binary header 25-26)
* Not supporting different trace lengths (binary header 303-304)
* Not supporting Extended Textual File Header records following the Binary Header (binary header 305-307) 
* Not supporting Big Endian

"""
import numpy as np
import struct
import os


def _i2(chars, byte):
    r"""Returns the 2-byte integer from <byte> position in chars"""
    return( int(chars[(byte-1):(byte-1)+2].encode('hex'),16)  )
        
def _i4(chars, byte): 
    r"""Returns the 4-byte integer from <byte> position in chars"""
    return( int(chars[(byte-1):(byte-1)+4].encode('hex'),16)  )
    #return struct.unpack("H", chars[(byte-1):(byte-1)+4])[0]    

def _ibm2ieee(ibm_chars):
    r"""
    Converts an IBM floating point number into IEEE format.
    :param: ibm - 32 bit unsigned integer: unpack('>L', f.read(4))
    """

    ibm = struct.unpack('>L', ibm_chars)[0]
    sign = ibm >> 31 & 0x01
    exponent = ibm >> 24 & 0x7f
    mantissa = ibm & 0x00ffffff
    mantissa = (mantissa * 1.0) / pow(2, 24)
    ieee = (1 - 2 * sign) * mantissa * pow(16, exponent - 64)
    return ieee

def _str2samples(sformat, chars):
    samples = []

    if (sformat == 1):  # IBM float
        for i in range(0, len(chars), 4):
            samples.append(_ibm2ieee(chars[i:i+4]))
        return samples
    elif (sformat == 5):  # IEEE float
        for i in range(0, len(chars), 4):
            samples.append(struct.unpack(">f",chars[i:i+4])[0])
        return samples


class Segy(object):
    r"""
    Segy file format utility class
    
    """
    def __init__(self, filename):
        r"""
        Create a segy class by assigning the path filename 
        to the segy file.
        Use as:
        
        with Segy('mysegyfile.sgy') as sgy:
            do whatever you want with the contents
        
         
        
        * filename : segy file name path    
        """
        self.filename = filename


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
        self.binaryheader = self._getbinaryheader(filebytes[3200:3600])
        
        if not self.binaryheader['fixed_length_traces'] == 1:
            #raise Exception('file not supported')
            pass 
        if not self.binaryheader['sgy_revision'] == 1:
            #raise Exception('file not supported')
            pass 
        if not self.binaryheader['extended_text_header'] == 0:
            #raise Exception('file not supported')
            pass
        # IBM and IEEE support 5/1        
        if not ( self.binaryheader['data_format'] == 5 or self.binaryheader['data_format'] == 1):
            #raise Exception('file not supported')
            pass
        # for not extended text header and fixed length trace gets
        # the number of traces based on 4 byte sample data format
        self.filesize = os.stat(self.filename).st_size
        
        self.samplestrace = self.binaryheader['samples_trace']
        self.numbertraces = (self.filesize-(3200+400))/(self.samplestrace*4+240)
        self.sample_format = self.binaryheader['data_format']
        
        self.samplerate = self.binaryheader['sample_interval']
        
        return self

    def __exit__(self, eType, eValue, eTrace):
        r""" close the file """
        self.file.close()

    def _getbinaryheader(self, chars):
        r""" 
        Loads the binary header into a dictionary for easy access. This dictionary uses the SEG-Y rev 1 standard.
        """
        self.binaryheader = {}
        
        self.binaryheader['job_id'] = _i4(chars, 1) # Job identification number
        self.binaryheader['line_number'] = _i4(chars, 5) # Line number
        self.binaryheader['rell_number'] = _i4(chars, 9) # Reel number
        self.binaryheader['number_traces'] = _i2(chars, 13) # Number of data traces per ensemble
        self.binaryheader['aux_traces'] = _i2(chars, 15) # Number of auxiliary traces per ensemble
        self.binaryheader['sample_interval'] = _i2(chars, 17) # @mandatory Sample interval in microseconds
        self.binaryheader['sample_interval_org'] = _i2(chars, 19) # Sample interval in microseconds (orig)
        self.binaryheader['samples_trace'] = _i2(chars, 21) # @mandatory Number of samples per data trace
        self.binaryheader['samples_trace_org'] = _i2(chars, 23) # Number of samples per data trace (orig)
        self.binaryheader['data_format'] = _i2(chars, 25) # @mandatory Data sample format code
        self.binaryheader['fold'] = _i2(chars, 27) # Ensemble fold 
        self.binaryheader['trace_sort'] = _i2(chars, 29) # Trace sorting code
        self.binaryheader['vertical_scode'] = _i2(chars, 31) # Vertical sum code
        self.binaryheader['sweep_fs'] = _i2(chars, 33) # Sweep frequency at start (Hz)
        self.binaryheader['sweep_fe'] = _i2(chars, 35) # Sweep frequency at end
        self.binaryheader['sweep_len'] = _i2(chars, 37) # Sweep length (ms)
        self.binaryheader['sweep_typ'] = _i2(chars, 39) # Sweep type code
        self.binaryheader['sweep_chn'] = _i2(chars, 41) # Trace number sweep channel
        self.binaryheader['sweep_stas'] = _i2(chars, 43) # Sweep trace taper length in milliseconds at start
        self.binaryheader['sweep_stae'] = _i2(chars, 45), # Sweep trace taper length in milliseconds at end
        self.binaryheader['taper_typ'] = _i2(chars, 47)# Taper type
        self.binaryheader['corr_traces'] = _i2(chars, 49) # Correlated data traces
        self.binaryheader['binary_grcv'] = _i2(chars, 51) # Binary gain recovered
        self.binaryheader['amp_rcvm'] = _i2(chars, 53) #Amplitude recovery method
        self.binaryheader['meters_feet'] = _i2(chars, 55) # Measurement system
        self.binaryheader['polarity'] = _i2(chars, 57) # Impulse signal polarity
        self.binaryheader['vpolarity'] = _i2(chars, 59) # Vibratory polarity code
        self.binaryheader['sgy_revision'] = _i2(chars,301) # SEG Y Format Revision Number 0100 hex
        self.binaryheader['fixed_length_traces'] = _i2(chars,303) # Fixed length trace flag (1 or 0)
        self.binaryheader['extended_text_header'] = _i2(chars,305) # Number of 3200-byte, Extended Textual File Header 
        # records following the  Binary Header
        return self.binaryheader

    def Traces(self, key):
        r"""
        return a especific trace from file 
        max traces defined by self.numbertraces
        """
        self.file.seek((3200+400)+(self.samplestrace*4+240)*key, 0) # 0 from begging of file
        tracechars = self.file.read((self.samplestrace*4+240))
        
        return Trace(self.binaryheader, tracechars)


class Trace(object):
    r"""Trace class."""

    def __init__(self, segybinaryheader, tracechars):
        """
        Initialization. Uses the raw binary trace <bintrace> to extract and populate both data and header members of the class
        """
        self.segybinaryheader = segybinaryheader        
        self.samplestrace = segybinaryheader['samples_trace']
        self.sample_format = segybinaryheader['data_format']
        # already convert every sample to the correct float format
        self.samples = _str2samples(self.sample_format, tracechars[240:])
        # get the trace header values
        self.traceheader = self._gettraceheader(tracechars[:240])
        
        if (len(self.samples) != self.traceheader['nsamples']):
            pass
            #raise exception? npts from header differs from real npts
       
    
    def __getitem__(self, key):
        r"""
         return a sample value at index position key
         max defined by self.samplestrace
        """
        return self.samples[key]

    def _gettraceheader(self, chars):
        """Load information from _binhdr into a dictionary for easy access. This dictionary uses the SEG-Y rev 1 standard"""
        self.traceheader = {}
        self.traceheader['trace_number_line']        = _i4(chars, 1)    # Trace sequence number within line
        self.traceheader['trace_number_reel']        = _i4(chars, 5)    # Trace sequence number within reel
        self.traceheader['FFID']                     = _i4(chars, 9)    # FFID - Original field record number
        self.traceheader['trace_number_field']       = _i4(chars, 13)   # Trace number within field record
        self.traceheader['SP']                       = _i4(chars, 17)   # SP - energy source point number
        self.traceheader['ensemble']                 = _i4(chars, 21)   # Ensemble number (CDP, CMP, CRP, etc)
        self.traceheader['CDP']                      = _i4(chars, 21)   # Ensemble number (CDP, CMP, CRP, etc)
        self.traceheader['CMP']                      = _i4(chars, 21)   # Ensemble number (CDP, CMP, CRP, etc)
        self.traceheader['CRP']                      = _i4(chars, 21)   # Ensemble number (CDP, CMP, CRP, etc)
        self.traceheader['trace_number']             = _i4(chars, 25)   # Trace number within ensemble
        self.traceheader['trace_id_code']            = _i2(chars, 29)   # Trace Identification code (0 = Unknown, 1= Seismic data, 2 = Dead, etc)
        self.traceheader['n_vertically_summed']      = _i2(chars, 31)   # Number of vertically summed traces
        self.traceheader['n_horizontally_stacked']   = _i2(chars, 33)   # Number of horizontally stacked traces
        self.traceheader['data_use']                 = _i2(chars, 35)   # Data use (1=production, 2=test)
        self.traceheader['offset']                   = _i4(chars, 37)   # Offset (distance from source point to receiver group)
        self.traceheader['rcvr_grp_elevation']       = _i4(chars, 41)   # Receiver group elevation
        self.traceheader['surface_elevation_src']    = _i4(chars, 45)   # Surface elevation at source
        self.traceheader['src_depth']                = _i4(chars, 49)   # Source depth below surface (>0)
        self.traceheader['datum_elevation_rcvr']     = _i4(chars, 53)   # Datum elevation at receiver group
        self.traceheader['datum_elevation_src']      = _i4(chars, 57)   # Datum elevation at source
        self.traceheader['water_depth_src']          = _i4(chars, 61)   # Water depth at source
        self.traceheader['water_depth_grp']          = _i4(chars, 65)   # Water depth at group
        self.traceheader['scalar_elevation_depths']  = _i2(chars, 69)   # Scalar to be applied to all elevations and depths
        self.traceheader['scalar_coordinates']       = _i2(chars, 71)   # Scalar to be applied to all coordinates
        self.traceheader['src_X_coord']              = _i4(chars, 73)   # Source X-coordinate
        self.traceheader['src_Y_coord']              = _i4(chars, 77)   # Source Y-coordinate
        self.traceheader['grp_X_coord']              = _i4(chars, 81)   # Group X-coordinate
        self.traceheader['grp_Y_coord']              = _i4(chars, 85)   # Group Y-coordinate
        self.traceheader['coord_units']              = _i2(chars, 89)   # Coordinate units
        self.traceheader['weathering_vel']           = _i2(chars, 91)   # Weathering velocity
        self.traceheader['subweathering_vel']        = _i2(chars, 93)   # Subweathering velocity
        self.traceheader['uphole_time_src']          = _i2(chars, 95)   # Uphole time at source (ms)
        self.traceheader['uphole_time_grp']          = _i2(chars, 97)   # Uphole time as group (ms)
        self.traceheader['src_static']               = _i2(chars, 99)   # Source static corretion (ms)
        self.traceheader['grp_static']               = _i2(chars, 101)  # Group static corretion (ms)
        self.traceheader['total_static']             = _i2(chars, 103)  # Total static applied (ms)
        self.traceheader['lag_time_A']               = _i2(chars, 105)  # Lag time A
        self.traceheader['lag_time_B']               = _i2(chars, 107)  # Lag time B
        self.traceheader['delay_recording_time']     = _i2(chars, 109)  # Delay recording time
        self.traceheader['mute_start']               = _i2(chars, 111)  # Mute time - start
        self.traceheader['mute_end']                 = _i2(chars, 113)  # Mute time - end
        self.traceheader['number_of_samples']        = _i2(chars, 115)  # Number of samples in this trace
        self.traceheader['nsamples']                 = _i2(chars, 115)  # Number of samples in this trace
        self.traceheader['nsample']                  = _i2(chars, 115)  # Number of samples in this trace
        self.traceheader['sample_interval']          = _i2(chars, 117)  # Sample interval
        self.traceheader['gain_type']                = _i2(chars, 119)  # Gain type of field instruments
        self.traceheader['inst_gain_constant']       = _i2(chars, 121)  # Instrument gain constant
        self.traceheader['inst_ini_gain']            = _i2(chars, 123)  # Instrument initial or early gain
        self.traceheader['correlated']               = _i2(chars, 125)  # Correlated (0=no, 1=yes)
        self.traceheader['sweep_freq_start']         = _i2(chars, 127)  # Sweep frequency at start
        self.traceheader['sweep_freq_end']           = _i2(chars, 129)  # Sweep frequency at end
        self.traceheader['sweep_length']             = _i2(chars, 131)  # Sweep length
        self.traceheader['sweep_type']               = _i2(chars, 133)  # Sweep type
        self.traceheader['sweep_tr_tpr_len_start']   = _i2(chars, 135)  # Sweep trace taper length at start
        self.traceheader['sweep_tr_tpr_len_end']     = _i2(chars, 137)  # Sweep trace taper length at end
        self.traceheader['taper_type']               = _i2(chars, 139)  # Taper type
        self.traceheader['alias_freq']               = _i2(chars, 141)  # Alias filter frequency
        self.traceheader['alias_slope']              = _i2(chars, 143)  # Alias filter slope
        self.traceheader['notch_freq']               = _i2(chars, 145)  # Notch filter frequency
        self.traceheader['notch_slope']              = _i2(chars, 147)  # Notch filter slope
        self.traceheader['lowcut_freq']              = _i2(chars, 149)  # Lowcut filter frequency
        self.traceheader['highcut_freq']             = _i2(chars, 151)  # Highcut filter frequency
        self.traceheader['lowcut_slope']             = _i2(chars, 153)  # Lowcut filter slope
        self.traceheader['highcut_slope']            = _i2(chars, 155)  # Highcut filter slope
        self.traceheader['year']                     = _i2(chars, 157)  # Year data recorded
        self.traceheader['day']                      = _i2(chars, 159)  # Day of year 
        self.traceheader['hour']                     = _i2(chars, 161)  # Hour of day
        self.traceheader['minute']                   = _i2(chars, 163)  # Minute of hour
        self.traceheader['second']                   = _i2(chars, 165)  # Second of minute
        self.traceheader['time_basis']               = _i2(chars, 167)  # Time basis code
        self.traceheader['weighting_factor']         = _i2(chars, 169)  # Trace weighting factor
        self.traceheader['geophone_grp_n_roll_p_1']  = _i2(chars, 171)  # Geophone group number of roll switch position one
        self.traceheader['geophone_grp_n_tr_1']      = _i2(chars, 173)  # Geophone group number of trace one
        self.traceheader['geophone_grp_n_last_tr']   = _i2(chars, 175)  # Geophone group number of last trace
        self.traceheader['gap_size']                 = _i2(chars, 177)  # Gap size (# of groups dropped)
        self.traceheader['over_travel_taper']        = _i2(chars, 179)  # Over travel associated with taper at beginning or end of line
        self.traceheader['CDP_X']                    = _i4(chars, 181)  # CDP X coordinate
        self.traceheader['CDP_Y']                    = _i4(chars, 185)  # CDP Y coordinate
        self.traceheader['inline']                   = _i4(chars, 189)  # Inline number
        self.traceheader['iline']                    = _i4(chars, 189)  # Inline number
        self.traceheader['crossline']                = _i4(chars, 193)  # Crossline number
        self.traceheader['xline']                    = _i4(chars, 193)  # Crossline number
        self.traceheader['shot_point_number']        = _i4(chars, 197)  # Shot point number
        self.traceheader['shot_point_scalar']        = _i4(chars, 201)  # Shot point scalar
        self.traceheader['trc_value_unit']           = _i2(chars, 203)  # Trace value measurement unit
        self.traceheader['transduction_const_mant']  = _i4(chars, 205)  # Transduction constant (mantissa)
        self.traceheader['transduction_const_pow']   = _i2(chars, 209)  # Transduction constant (power)
        self.traceheader['transduction_unit']        = _i2(chars, 211)  # Transduction unit
        self.traceheader['identifier']               = _i2(chars, 213)  # Device/trace identifier
        self.traceheader['scalar_time']              = _i2(chars, 215)  # Scalar to be applied to bytes 95-114 (basically concerning times)
        self.traceheader['source_tp_orient']         = _i2(chars, 217)  # Source type/orientation
        #self.traceheader['source_energy_direction']  = _i4(chars, 219)  # Source energy direction
        self.traceheader['source_measurement_mant']  = _i4(chars, 225)  # Source measurement (mantissa)
        self.traceheader['source_measurement_pow']   = _i2(chars, 229)  # Source measurement (power)
        
        return self.traceheader

    def __repr__(self):
        tmp = "Trace " + str(self.traceheader['trace_number']) + "; "
        tmp += "Ensemble " + str(self.traceheader['ensemble']) + "; "
        tmp += "Offset " + str(self.trace['offset']) + "; "
        tmp += "Samples " + str(self.traceheader['nsample'])
        return tmp
    
    def summary(self):  #necessary?
        summary = \
        {"Number of samples": self.traceheader['nsamples'], \
        "Offset": self.traceheader['offset'], \
        "Ensemble": self.traceheader['CDP'], \
        "FFID": self.traceheader['FFID'] }
        return (summary)
    

    
