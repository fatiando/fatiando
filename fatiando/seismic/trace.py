#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from numpy import *
import struct


class Trace:
    """Trace class."""

    def __init__(self, bintrace):
        """
        Initialization. Uses the raw binary trace <bintrace> to extract and populate both data and header members of the class
        """
        self._binhdr = bintrace[:240]
        self.data = self.__convertdata(bintrace[240:])
        self.header = self.__loadheader()

        self.offset =    self.__i4(37)
        self.cdp =       self.__i4(21)
        self.trcnumber = self.__i4(25)
        
        if (len(self.data) != self.__i2(115)):
            pass
            #raise exception? npts from header differs from real npts
        else:
            self.npts =  self.__i2(115)
        
        
       
    def __convertdata(self, trc):
        trctmp = []
        for i in range(0, len(trc), 4):
            trctmp.append(__ibm2ieee2(trc[i:i+4]))
        return trctmp
        
    def __ibm2Ieee(ibm_float):
        """
        ibm2Ieee(ibm_float)
        Used by permission
        (C) Secchi Angelo
        with thanks to Howard Lightstone and Anton Vredegoor. 
        """
        I = struct.unpack('>I',ibm_float)[0]
        sign = [1,-1][bool(i & 0x100000000L)]
        characteristic = ((i >> 24) & 0x7f) - 64
        fraction = (i & 0xffffff)/float(0x1000000L)
        return sign*16**characteristic*fraction



    def __ibm2ieee2(ibm_float):
        """
        ibm2ieee2(ibm_float)
        Used by permission
        (C) Secchi Angelo
        with thanks to Howard Lightstone and Anton Vredegoor. 
        """
        dividend=float(16**6)
        
        if ibm_float == 0:
            return 0.0
        istic,a,b,c=struct.unpack('>BBBB',ibm_float)
        if istic >= 128:
            sign= -1.0
            istic = istic - 128
        else:
            sign = 1.0
        mant= float(a<<16) + float(b<<8) +float(c)
        return sign* 16**(istic-64)*(mant/dividend)
        
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
        