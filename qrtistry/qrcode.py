# -*- coding: utf-8 -*-
import scipy.misc, numpy, numpy.linalg, math, reedsolo
import utils

from enum import Enum
from abc import abstractmethod, ABCMeta, abstractproperty

class Direction(Enum):
    TOP_TO_BOTTOM_LEFT_TO_RIGHT = 0
    LEFT_TO_RIGHT_TOP_TO_BOTTOM = 1

def format_xor(seq):
    return seq and tuple( map(lambda a: a[0]^a[1], zip(seq, FORMAT_XOR_MASK)) )

FORMAT_XOR_MASK = [e == "1" for e in reversed("101010000010010")]

## Source: https://github.com/davidshimjs/qrcodejs/blob/master/qrcode.js
## Licensed under MIT license
RS_BLOCK_TABLE= \
[[1, 26, 19],
 [1, 26, 16],
 [1, 26, 13],
 [1, 26, 9],
 [1, 44, 34],
 [1, 44, 28],
 [1, 44, 22],
 [1, 44, 16],
 [1, 70, 55],
 [1, 70, 44],
 [2, 35, 17],
 [2, 35, 13],
 [1, 100, 80],
 [2, 50, 32],
 [2, 50, 24],
 [4, 25, 9],
 [1, 134, 108],
 [2, 67, 43],
 [2, 33, 15, 2, 34, 16],
 [2, 33, 11, 2, 34, 12],
 [2, 86, 68],
 [4, 43, 27],
 [4, 43, 19],
 [4, 43, 15],
 [2, 98, 78],
 [4, 49, 31],
 [2, 32, 14, 4, 33, 15],
 [4, 39, 13, 1, 40, 14],
 [2, 121, 97],
 [2, 60, 38, 2, 61, 39],
 [4, 40, 18, 2, 41, 19],
 [4, 40, 14, 2, 41, 15],
 [2, 146, 116],
 [3, 58, 36, 2, 59, 37],
 [4, 36, 16, 4, 37, 17],
 [4, 36, 12, 4, 37, 13],
 [2, 86, 68, 2, 87, 69],
 [4, 69, 43, 1, 70, 44],
 [6, 43, 19, 2, 44, 20],
 [6, 43, 15, 2, 44, 16],
 [4, 101, 81],
 [1, 80, 50, 4, 81, 51],
 [4, 50, 22, 4, 51, 23],
 [3, 36, 12, 8, 37, 13],
 [2, 116, 92, 2, 117, 93],
 [6, 58, 36, 2, 59, 37],
 [4, 46, 20, 6, 47, 21],
 [7, 42, 14, 4, 43, 15],
 [4, 133, 107],
 [8, 59, 37, 1, 60, 38],
 [8, 44, 20, 4, 45, 21],
 [12, 33, 11, 4, 34, 12],
 [3, 145, 115, 1, 146, 116],
 [4, 64, 40, 5, 65, 41],
 [11, 36, 16, 5, 37, 17],
 [11, 36, 12, 5, 37, 13],
 [5, 109, 87, 1, 110, 88],
 [5, 65, 41, 5, 66, 42],
 [5, 54, 24, 7, 55, 25],
 [11, 36, 12],
 [5, 122, 98, 1, 123, 99],
 [7, 73, 45, 3, 74, 46],
 [15, 43, 19, 2, 44, 20],
 [3, 45, 15, 13, 46, 16],
 [1, 135, 107, 5, 136, 108],
 [10, 74, 46, 1, 75, 47],
 [1, 50, 22, 15, 51, 23],
 [2, 42, 14, 17, 43, 15],
 [5, 150, 120, 1, 151, 121],
 [9, 69, 43, 4, 70, 44],
 [17, 50, 22, 1, 51, 23],
 [2, 42, 14, 19, 43, 15],
 [3, 141, 113, 4, 142, 114],
 [3, 70, 44, 11, 71, 45],
 [17, 47, 21, 4, 48, 22],
 [9, 39, 13, 16, 40, 14],
 [3, 135, 107, 5, 136, 108],
 [3, 67, 41, 13, 68, 42],
 [15, 54, 24, 5, 55, 25],
 [15, 43, 15, 10, 44, 16],
 [4, 144, 116, 4, 145, 117],
 [17, 68, 42],
 [17, 50, 22, 6, 51, 23],
 [19, 46, 16, 6, 47, 17],
 [2, 139, 111, 7, 140, 112],
 [17, 74, 46],
 [7, 54, 24, 16, 55, 25],
 [34, 37, 13],
 [4, 151, 121, 5, 152, 122],
 [4, 75, 47, 14, 76, 48],
 [11, 54, 24, 14, 55, 25],
 [16, 45, 15, 14, 46, 16],
 [6, 147, 117, 4, 148, 118],
 [6, 73, 45, 14, 74, 46],
 [11, 54, 24, 16, 55, 25],
 [30, 46, 16, 2, 47, 17],
 [8, 132, 106, 4, 133, 107],
 [8, 75, 47, 13, 76, 48],
 [7, 54, 24, 22, 55, 25],
 [22, 45, 15, 13, 46, 16],
 [10, 142, 114, 2, 143, 115],
 [19, 74, 46, 4, 75, 47],
 [28, 50, 22, 6, 51, 23],
 [33, 46, 16, 4, 47, 17],
 [8, 152, 122, 4, 153, 123],
 [22, 73, 45, 3, 74, 46],
 [8, 53, 23, 26, 54, 24],
 [12, 45, 15, 28, 46, 16],
 [3, 147, 117, 10, 148, 118],
 [3, 73, 45, 23, 74, 46],
 [4, 54, 24, 31, 55, 25],
 [11, 45, 15, 31, 46, 16],
 [7, 146, 116, 7, 147, 117],
 [21, 73, 45, 7, 74, 46],
 [1, 53, 23, 37, 54, 24],
 [19, 45, 15, 26, 46, 16],
 [5, 145, 115, 10, 146, 116],
 [19, 75, 47, 10, 76, 48],
 [15, 54, 24, 25, 55, 25],
 [23, 45, 15, 25, 46, 16],
 [13, 145, 115, 3, 146, 116],
 [2, 74, 46, 29, 75, 47],
 [42, 54, 24, 1, 55, 25],
 [23, 45, 15, 28, 46, 16],
 [17, 145, 115],
 [10, 74, 46, 23, 75, 47],
 [10, 54, 24, 35, 55, 25],
 [19, 45, 15, 35, 46, 16],
 [17, 145, 115, 1, 146, 116],
 [14, 74, 46, 21, 75, 47],
 [29, 54, 24, 19, 55, 25],
 [11, 45, 15, 46, 46, 16],
 [13, 145, 115, 6, 146, 116],
 [14, 74, 46, 23, 75, 47],
 [44, 54, 24, 7, 55, 25],
 [59, 46, 16, 1, 47, 17],
 [12, 151, 121, 7, 152, 122],
 [12, 75, 47, 26, 76, 48],
 [39, 54, 24, 14, 55, 25],
 [22, 45, 15, 41, 46, 16],
 [6, 151, 121, 14, 152, 122],
 [6, 75, 47, 34, 76, 48],
 [46, 54, 24, 10, 55, 25],
 [2, 45, 15, 64, 46, 16],
 [17, 152, 122, 4, 153, 123],
 [29, 74, 46, 14, 75, 47],
 [49, 54, 24, 10, 55, 25],
 [24, 45, 15, 46, 46, 16],
 [4, 152, 122, 18, 153, 123],
 [13, 74, 46, 32, 75, 47],
 [48, 54, 24, 14, 55, 25],
 [42, 45, 15, 32, 46, 16],
 [20, 147, 117, 4, 148, 118],
 [40, 75, 47, 7, 76, 48],
 [43, 54, 24, 22, 55, 25],
 [10, 45, 15, 67, 46, 16],
 [19, 148, 118, 6, 149, 119],
 [18, 75, 47, 31, 76, 48],
 [34, 54, 24, 34, 55, 25],
 [20, 45, 15, 61, 46, 16]]

class Codeword(object):
    def __init__(self, num, value=None):
        self.num = num
        self.value = value
    def __repr__(self):
        return "%s(%s%s)" % (self.__class__.__name__, self.num, ",value=%s" % self.value if self.value is not None else "")
class DataCodeword(Codeword): pass
class ErrorCorrectionCodeword(Codeword): pass

class ErrorCorrectionSequence(object):
    def __init__(self, version, level):
        self.version = version
        self.level = level
        self.data_blocks = []
        self.error_correction_blocks = []
        
        mapping = {
            QRCode.ErrorCorrection.L: 0,
            QRCode.ErrorCorrection.M: 1,
            QRCode.ErrorCorrection.Q: 2,
            QRCode.ErrorCorrection.H: 3,
        }
        
        block_info = RS_BLOCK_TABLE[ (version-1)*4 + mapping[level]  ]
        
        last_data = 1
        last_error = 1
        
        while block_info:
            (number, total, data), block_info = block_info[:3], block_info[3:]
            for _ in range(number):
                block = []
                for _ in range(data):
                    block.append(DataCodeword(last_data))
                    last_data = last_data + 1
                self.data_blocks.append(block)
                
                block = []
                for _ in range(total-data):
                    block.append(ErrorCorrectionCodeword(last_error))
                    last_error = last_error + 1
                self.error_correction_blocks.append(block)
    
    def column_order(self):
        def column_order(blocks):
            for column in range(len(blocks[-1])):
                for row in range(len(blocks)):
                    if column < len(blocks[row]):
                        yield blocks[row][column]
        
        for b in column_order(self.data_blocks):
            yield b
        for b in column_order(self.error_correction_blocks):
            yield b
    
    def rows(self):
        for i in range(len(self.data_blocks)):
            yield self.data_blocks[i], self.error_correction_blocks[i]
    
    def data_codewords(self):
        for row_items in self.data_blocks:
            for cw in row_items:
                yield cw

class Block(object):
    def __init__(self, offset_i=0, offset_j=0):
        l = locals()
        del locals()["self"]
        
        for k, v in l.items():
            setattr(self, k, v)
        
        self.positions = []
    
    def __iter__(self):
        for i, j in self.positions:
            yield i + self.offset_i, j + self.offset_j
    
    def __repr__(self):
        return "Block.from_positions(positions=%r,offset_i=%i,offset_j=%i)" % (self.positions,self.offset_i,self.offset_j)

    @classmethod
    def rect(cls, width, height, direction=Direction.TOP_TO_BOTTOM_LEFT_TO_RIGHT,  **kwargs):
        i, j = 0, 0
         
        result = cls(**kwargs)
         
        for x in range(width * height):
            if i >= width:
                i = 0
                j = j + 1
            if j >= height:
                j = 0
                i = i + 1
             
            result.positions.append( (i, j) )
             
            if direction == Direction.TOP_TO_BOTTOM_LEFT_TO_RIGHT:
                j = j + 1
            else:
                i = i + 1
         
        return result
    
    @classmethod
    def format_left_upper(cls, **kwargs):
        result = cls(**kwargs)
        i = 8
        for j in range(8):
            if j > 5: j = j + 1
            result.positions.append( (i, j) )
        for i in range(7, 0, -1):
            if i < 7: i = i - 1
            result.positions.append( (i, j) )
        
        result.positions = list(reversed(result.positions))
        
        return result
    
    @classmethod
    def format_right_and_bottom(cls, code_size, **kwargs):
        result = cls(**kwargs)
        j = 8
        for i in range(code_size-1, code_size-9, -1):
            result.positions.append( (i, j) )
        i = 8
        for j in range(code_size-7, code_size):
            result.positions.append( (i, j) )
        
        result.positions = list(reversed(result.positions))
        
        return result

    @classmethod
    def from_positions(cls, positions, **kwargs):
        result = cls(**kwargs)
        result.positions = positions
        return result

class BCH(object):
    def __init__(self, code_bits, data_bits):
        self.code_bits = code_bits
        self.data_bits = data_bits
    
    def decode(self, data):
        if data is None: return data
        # FIXME Implement :)
        return tuple(reversed(data[-self.data_bits:]))

class BitstreamEvent(object):
    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)
    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, ",".join("%s=%r" % _ for _ in self.kwargs.items()))

class DecodeError(BitstreamEvent): pass
class CharacterCountIndicator(BitstreamEvent): pass

class QRCode(object):
    class ErrorCorrection(Enum):
        L = 1
        M = 0
        Q = 3
        H = 2
    class MaskPattern(Enum):
        MaskPattern_000 = 0
        MaskPattern_001 = 1
        MaskPattern_010 = 2
        MaskPattern_011 = 3
        MaskPattern_100 = 4
        MaskPattern_101 = 5
        MaskPattern_110 = 6
        MaskPattern_111 = 7
        

    FUNCTION_MODULE_FINDER = 1
    FUNCTION_MODULE_TIMING = 2
    FUNCTION_MODULE_FORMAT = 4
    FUNCTION_MODULE_VERSION = 8
    FUNCTION_MODULE_ALIGNMENT = 16
    FUNCTION_MODULE_FIXED = 32
    FUNCTION_MODULE_ALL = FUNCTION_MODULE_FINDER | FUNCTION_MODULE_TIMING | \
        FUNCTION_MODULE_FORMAT | FUNCTION_MODULE_VERSION | FUNCTION_MODULE_ALIGNMENT | \
        FUNCTION_MODULE_FIXED

    def __init__(self, qr_image=None, qr_image_offset=numpy.array([0, 0]), 
            background_image=None, background_image_offset=numpy.array([0, 0]), 
            canvas_size=None, 
            ul=None, # Upper left
            rv=None, # Right vector
            dv=None, # Down vector
            version=None, 
            code_size=None, 
            error_correction=None, 
            mask=None, 
            blocks=None,
        ):
        l = locals()
        del locals()["self"]
        
        for k, v in l.items():
            setattr(self, k, v)
        
        if self.canvas_size is None:
            minsize = (0, 0)
            for i in "qr", "background":
                image = l["%s_image" % i]
                offset = l["%s_image_offset" % i]
                if image is None: continue
                
                minsize = map(
                    max, 
                    zip(
                        map(
                            sum, 
                            zip(image.shape[:2], offset)
                        ), 
                        minsize
                    )
                )
            self.canvas_size = minsize
        
        self.error_correction_sequence = None
        
        self.debug_color = None
        self.debug_image = numpy.array([[[0, 0, 0]]*self.canvas_size[0]]*self.canvas_size[1])
        for x in range(self.canvas_size[0]):
            for y in range(self.canvas_size[1]):
                self.debug_image[y, x] = numpy.array([1, 1, 1]) * self.get_pixel( numpy.array([x, y]) )
        self.debug_color = [255, 0, 0]
    
    def __repr__(self):
        attribs = ["version", "code_size", "mask", "error_correction"]
        return "<QRCode(%s)>" % ", ".join("%s=%s" % (k, getattr(self, k)) for k in attribs)
    
    def encode_tty(self, apply_mask=True, mark_special=True, prefix="  "):
        for i in range(self.code_size):
            yield prefix + "".join(map(lambda j: 
                { 0: ["  ", "██"][self.get_module(i, j, apply_mask=apply_mask)], 
                    self.FUNCTION_MODULE_FINDER: "Ｏ", 
                    self.FUNCTION_MODULE_TIMING: "Ｔ", 
                    self.FUNCTION_MODULE_FORMAT: "Ｆ", 
                    self.FUNCTION_MODULE_VERSION: "Ｖ", 
                    self.FUNCTION_MODULE_ALIGNMENT: "Ａ", 
                    self.FUNCTION_MODULE_FIXED: "Ｘ", 
                }[mark_special and self.is_function_module(i, j)], range(self.code_size)))


    def find_points(self):
        pf = utils.PointFinderZXing()
        
        points = pf.find_image(self.qr_image)
        assert len(points) >= 3
        
        # Left bottom, left top, left right points
        self.dl, self.ul, self.ur =  map(lambda a: numpy.array(a) + self.qr_image_offset,  points[:3] )
        
        # normalize down and right vectors for 1 pixel
        self.rv = self.ur - self.ul; self.rv /= numpy.linalg.norm(self.rv)
        self.dv = self.dl - self.ul; self.dv /= numpy.linalg.norm(self.dv)
    
    def read_version(self):
        # Starting at the base point, go in all 4 directions and find the mandatory black-white-black-white transitions
        #  determine the thickness between them (should be 1 module) and use the median of those number as the module size
        # Use the module size and distance between finders to determine provisional version,
        #  then try to read real version from data, if necessary
        
        sizes = [[], []]
        module_sizes = [0, 0]
        STEPS = 1000
        STEP_SIZE = 0.1
        
        for base in self.ul, self.dl, self.ur:
            for axis, directions in enumerate([ (self.rv, -self.rv),  (self.dv, -self.dv) ]):
                for direction in directions:
                    last_v = self.get_pixel( base )
                    last_i = None
                    count = 0  # At count=0, expect black to white, at count=1: white to black, etc. end at count=3
                    
                    for i in range(STEPS):
                        try:
                            v = self.get_pixel( base +  i * STEP_SIZE * direction )
                        except IndexError:
                            break
                        
                        # FIXME Smarter detection
                        v = [0, 255][v >= 128]
                        
                        difference = v - last_v
                        last_v = v
                        
                        if (difference > 0 and count%2 == 0) or (difference < 0 and count%2 == 1):
                            if last_i is not None:
                                sizes[axis].append( (i - last_i) * STEP_SIZE )
                            last_i = i
                            count = count + 1
                        
                        if count >= 3:
                            break
        
        for axis in range(2):
            module_sizes[axis] = sum(sizes[axis])/float(len(sizes[axis]))
        
        print module_sizes
        
        # normalize down and right vectors for 1 module
        self.rv = self.rv / numpy.linalg.norm(self.rv) * module_sizes[0]
        self.dv = self.dv / numpy.linalg.norm(self.dv) * module_sizes[1]
        
        self.version = int( (numpy.linalg.norm(self.ur - self.ul) / numpy.linalg.norm(self.rv) - 10)/4 )
        self.code_size = 17 + self.version*4
        
        if self.version > 6:
            # Need to decode version
            version1_loc = Block.rect(6, 3, Direction.TOP_TO_BOTTOM_LEFT_TO_RIGHT, offset_j = self.code_size - 11)
            version2_loc = Block.rect(3, 6, Direction.LEFT_TO_RIGHT_TOP_TO_BOTTOM, offset_i = self.code_size - 11)
            
            version1 = BCH(18, 6).decode( self.get_modules(version1_loc) )
            version2 = BCH(18, 6).decode( self.get_modules(version2_loc) )
            
            version1, version2 = map(utils.bin2long_be, (version1, version2))
            
            if version1 == version2:
                if version1 is not None:
                    self.version = version1
            else:
                if version1 is None:
                    self.version = version2
                elif version2 is None:
                    self.version = version1
                else:
                    version1 = version1
                    version2 = version2
                    
                    # if version information doesn't match, use the one that's closest to our provisional version
                    if abs(version1 - self.version) < abs(version2 - self.version):
                        self.version = version1
                    else:
                        self.version = version2
            
            # Need to update code_size
            self.code_size = 17 + self.version*4
    
    def determine_alignment_pattern_positions(self):
        coords = []
        
        if self.version > 1:
            coords.append(6)
            coords.append( 18 + 4*(self.version - 2) )
            
            coords_len = 1 + (self.version+7)/7
            coords_diff = int(math.ceil( ( float(coords[1] - coords[0]) / (coords_len - 1) ) / 2.0 )*2)
            
            while len(coords) < coords_len:
                coords.append( coords[-1] - coords_diff )
        
        self.alignment_patterns = []
        
        for i in coords:
            for j in coords:
                if not (self.is_function_module(i, j) & self.FUNCTION_MODULE_FINDER):
                    self.alignment_patterns.append( (i, j) )
    
    def lay_out_blocks(self):
        self.blocks = []
        
        def module_order():
            direction = -1
            
            i = self.code_size - 1 # Row from top
            j = self.code_size - 1 # Column from left
            
            ## Kludge kludge: Column 6 is solely for timing and finder patterns.
            ## To make operations easier, we will ignore that it exists, and only
            ## operate on code_size-1 columns. Then, when returning something,
            ## we add 1 if the column is greater than 5, effectively only ever
            ## returning 0-5 and 7-(code_size-1)
            j = j - 1
            def fix_return(i,j):
                if j > 5: return (i,j+1)
                else: return (i,j) 
            
            while i >= 0 and j >= 0:
                yield fix_return(i,j)
                if j > 0:
                    j = j - 1
                    yield fix_return(i,j)
                    j = j + 1
                else:
                    yield fix_return(i,j)
                
                if (direction is -1 and i > 0) or (direction is 1 and i < self.code_size - 1):
                    i = i + direction
                else:
                    direction = direction * -1
                    j = j - 2
        
        positions = []
        for (i,j) in module_order():
            if self.is_function_module(i, j):
                continue
            positions.append( (i,j) )
            if len(positions) >= 8:
                self.blocks.append(Block.from_positions(list(reversed(positions))))
                positions = []
        
        print self.blocks
                    
        
    
    def read_format(self):
        format1_loc = Block.format_left_upper()
        format2_loc = Block.format_right_and_bottom(self.code_size)
        
        format1 = BCH(15, 5).decode( format_xor( self.get_modules(format1_loc) ) )
        format2 = BCH(15, 5).decode( format_xor( self.get_modules(format2_loc) ) )
        
        if format1 == format2:
            if format1 is None:
                raise Exception("Format decoding failed")
            else:
                format = format1
        elif format1 is None:
            format = format2
        elif format2 is None:
            format = format1
        else:
            # Note: This means that no error correction error was detected, but the values still differ
            #  Arbitrarily chose format2, though it's probably wrong. Should abort
            format = format2
        
        self.error_correction = self.ErrorCorrection( utils.bin2long_be(format[:2]) )
        self.mask = utils.bin2long_be(format[2:])

    
    def get_pixel(self, coords, interpolate=True, bw=True):
        # FIXME handle layers, interpolation
        
        qr_coords = coords - self.qr_image_offset
        
        # WARNING WARNING WARNING
        # In numpy, the first coordinate goes up-down, the second one left-right
        # This is of course in contrast to the rest of the world, where the first one
        # is left-right x and the second one is up-down y
        # We have to access with swapped coordinates here and in every reference to
        # the image contents
        qr_coords_px = ( int(math.floor(qr_coords[1])), int(math.floor(qr_coords[0])) )
        v = self.qr_image[ qr_coords_px  ]
        
        if self.debug_color is not None: 
            self.debug_image[ qr_coords_px ] += self.debug_color
            self.debug_image[ qr_coords_px ] /= 2
        
        if isinstance(v, (bool,numpy.bool_)):
            v = 255 * int(v)
        
        if bw:
            try:
                v = v[:3]
                v = sum(v) / len(v)
            except TypeError:
                pass
        else:
            try:
                v[:3]
            except TypeError:
                v = [v]*3
        
        return v
    
    OVERSAMPLING = 1
    
    def get_module(self, i, j, apply_mask=True):
        # Subtract midpoint of the finder pattern, then use basepoint and right/down vectors
        coords = self.ul + (i - 3.5) * self.dv + (j - 3.5) * self.rv
        
        # Sample the middle of the module
        coords = coords + 0.5 * self.dv + 0.5 * self.rv
        
        samples = [self.get_pixel(coords, bw=True)]
        for oversample_i in range(0, self.OVERSAMPLING):
            for oversample_j in range(0, self.OVERSAMPLING):
                if oversample_i == 0 and oversample_j == 0: continue
                oi = oversample_i/float(2*self.OVERSAMPLING)
                oj = oversample_j/float(2*self.OVERSAMPLING)
                
                samples.append( self.get_pixel(coords + oi * self.dv + oj * self.rv, bw=True) )
                samples.append( self.get_pixel(coords + oi * self.dv - oj * self.rv, bw=True) )
                samples.append( self.get_pixel(coords - oi * self.dv + oj * self.rv, bw=True) )
                samples.append( self.get_pixel(coords - oi * self.dv - oj * self.rv, bw=True) )
    
        v = sum(samples)/float(len(samples))
        module =  v < 128
        
        if apply_mask:
            module = module ^ self.get_mask(i, j)
        
        return module
    
    __getitem__ = get_pixel
    
    def get_modules(self, block, **kwargs):
        try:
            return [self.get_module(i, j, **kwargs) for i, j in block]
        except IndexError:
            return None
    
    # WARNING: Bits in codewords are included in the bitstream most-significant bit
    #  first (but least-numbered codeword first). So the bottom right module is
    #  bit 7 of the first codeword, and is also the first bit in the bitstream (where
    #  it generally is the leftmost bit of the mode indicator)
    def read_codewords(self):
        for block, cw in zip(self.blocks, self.error_correction_sequence.column_order()):
            value = utils.bin2long_be( e for e in reversed( self.get_modules(block)) )
            cw.value = value
    
    def fix_data_errors(self):
        for data_row, error_correction_row in self.error_correction_sequence.rows():
            values = [cw.value for cw in data_row + error_correction_row]
            rs = reedsolo.RSCodec(len(error_correction_row))
            corrected = rs.decode(values)
            
            for cw, value in zip(data_row, corrected):
                cw.value = value
    
    def get_data_bitstream(self):
        for cw in self.error_correction_sequence.data_codewords():
            value = cw.value
            for i in range(7,-1,-1):
                yield bool( (value >> i) & 1 )
    
    def parse_bitstream(self, bitstream):
        bitstream = iter(bitstream)
        while True:
            # Get mode indicator
            mode = Mode.resolve_mode( utils.bin2long_be( bitstream.next() 
                    for _ in range(4)) )
            yield mode
            
            if mode is SpecialMode.TERMINATOR:
                break
            
            if not issubclass(mode, Mode):
                yield DecodeError()
                break
            
            character_count = utils.bin2long_be( bitstream.next() 
                    for _ in range(mode.character_count_bitsize(self.version)) )
            yield CharacterCountIndicator(character_count=character_count)
            
            for character in mode.read(bitstream, character_count):
                yield character            
    
    def assign_error_correction_sequence(self):
        self.error_correction_sequence = ErrorCorrectionSequence(self.version, self.error_correction)
    
    def is_function_module(self, i, j):
        if i <= 6 and j <= 6: # Upper left finder
            return self.FUNCTION_MODULE_FINDER
        
        if j <= 6 and i >= self.code_size - 7: # Bottom left finder
            return self.FUNCTION_MODULE_FINDER
        
        if i <= 6 and j >= self.code_size - 7: # Upper right finder
            return self.FUNCTION_MODULE_FINDER
        
        for pat_i, pat_j in self.alignment_patterns:
            if abs(pat_i - i) <= 2 and abs(pat_j - j) <= 2:
                return self.FUNCTION_MODULE_ALIGNMENT
        
        if (i <= 7 and j == 7) or (i == 7 and j <= 7): # Blank upper left
            return self.FUNCTION_MODULE_FIXED
        
        if (i  >= self.code_size-8 and j== 7) or (i == self.code_size-8 and j <= 7): # Blank bottom left
            return self.FUNCTION_MODULE_FIXED

        if (i == 7 and j >= self.code_size-8) or (i <= 7 and j == self.code_size-8): # Blank upper right
            return self.FUNCTION_MODULE_FIXED
        
        if i == self.code_size - 8 and j == 8: # Fixed module bottom left    FIXME only for Model 2,  Model 1 is completely unhandled
            return self.FUNCTION_MODULE_FIXED
        
        if i == 6 or j == 6: # Timing
            return self.FUNCTION_MODULE_TIMING
        
        if i <= 8 and j <= 8: # Upper left format
            return self.FUNCTION_MODULE_FORMAT
        
        if i >= self.code_size - 7 and j == 8 :  # Bottom left format
            return self.FUNCTION_MODULE_FORMAT
        
        if i == 8 and j >= self.code_size - 8:  # Upper right format
            return self.FUNCTION_MODULE_FORMAT
        
        if self.version >= 7:
            if j <= 6 and i >= self.code_size - 11: # Bottom left version
                return self.FUNCTION_MODULE_VERSION
            
            if i <= 6 and j >= self.code_size - 11: # Upper right version
                return self.FUNCTION_MODULE_VERSION
        
        
        return 0
    
    def get_mask(self, i, j):
        # Mask information unavailable
        if self.mask is None:
            return False
        
        # Function modules
        if self.is_function_module(i, j):
            return False
        
        if self.mask == 0:
            return (i+j)%2 == 0
        elif self.mask == 1:
            return i%2 == 0
        elif self.mask == 2:
            return j%3 == 0
        elif self.mask == 3:
            return (i+j)%3 == 0
        elif self.mask == 4:
            return ((i/2)+(j/3))%2 == 0
        elif self.mask == 5:
            return (i * j) %2 + (i * j) %3 == 0
        elif self.mask == 6:
            return ((i * j) % 2 + (i * j) % 3) % 2 == 0
        elif self.mask == 7:
            return ((i * j) % 3 + (i+j) % 2) % 2 == 0
        
        return False
    
    @classmethod
    def encode_from_string(cls, s):
        raise NotImplementedError
    
    @classmethod
    def decode_from_image_file(cls, image_file):
        image = scipy.misc.imread(image_file)
        return cls.decode_from_image(image)

    @classmethod
    def decode_from_image(cls, image):
        qr = cls(qr_image = image)
        
        qr.find_points()
        qr.debug_color = [0, 255, 0]
        qr.read_version()
        qr.debug_color = [0, 0, 255]
        qr.read_format()
        qr.debug_color = [0, 255, 255]
        qr.determine_alignment_pattern_positions()
        qr.debug_color = [255, 255, 0]
        qr.lay_out_blocks()
        qr.assign_error_correction_sequence()
        qr.read_codewords()
        qr.fix_data_errors()
        
        return qr

class SpecialMode(Enum):
    ECI = 7
    STRUCTURED_APPEND = 3
    FNC1_FIRST = 5
    FNC1_ADDITIONAL = 9
    TERMINATOR = 0

class Mode(object):
    __metaclass__ = ABCMeta
    
    def __init__(self,value):
        self.value = value
    
    @classmethod
    def resolve_mode(cls, mode):
        for candidate in cls.__subclasses__():
            if candidate.MODE_INDICATOR == mode:
                return candidate
        return SpecialMode(mode)
    
    @classmethod
    @abstractmethod
    def estimate_encoded_size(cls, value):
        pass
    
    @classmethod
    def select_shortest(cls, value):
        candidates = []
        for candidate in cls.__subclasses__():
            size = candidate.estimate_encoded_size(value)
            if size:
                candidates.append( (size, candidate) )
        if not candidates:
            raise ValueError("Couldn't find a class to encode %r" % value)
        candidates.sort()
        return candidates[0][1](value)
    
    @abstractproperty
    def MODE_INDICATOR(self): pass
    
    @classmethod
    @abstractmethod
    def character_count_bitsize(cls, version): pass
    
    @classmethod
    @abstractmethod
    def read(cls, bitstream, character_count): pass

class NumericMode(Mode):
    CHARACTERS = "0123456789"
    MODE_INDICATOR = 1

    @classmethod
    def estimate_encoded_size(cls, value):
        if not isinstance(value, basestring):
            value = str(value)
        if any(e not in cls.CHARACTERS for e in value):
            return None
        return 10*( (len(value)+2)/3 )
    
    @classmethod
    def character_count_bitsize(cls, version):
        if version < 10: return 10
        if version < 27: return 12
        return 14
    
    @classmethod
    def read(cls, bitstream, character_count):
        for _ in range(character_count/3):
            int_value = utils.bin2long_be( bitstream.next() for _ in range(10) )
            value = "%03i" % int_value
            for char in value: yield char
        
        additional = 0
        if character_count % 3 == 1:
            additional = 4
            format_ = "%01i"
        elif character_count % 3 == 2:
            additional = 7
            format_ = "%02i"
        
        if additional:
            int_value = utils.bin2long_be( bitstream.next() for _ in range(additional) )
            value = format_ % int_value
            for char in value: yield char

class AlphaNumericMode(Mode):
    CHARACTERS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:"
    MODE_INDICATOR = 2
    
    @classmethod
    def estimate_encoded_size(cls, value):
        if not isinstance(value, basestring):
            value = str(value)
        if any(e not in cls.CHARACTERS for e in value):
            return None
        return 11*( (len(value)+1)/2 )

    @classmethod
    def character_count_bitsize(cls, version):
        if version < 10: return 9
        if version < 27: return 11
        return 13

    @classmethod
    def read(cls, bitstream, character_count):
        for _ in range(character_count / 2):
            int_value = utils.bin2long_be( bitstream.next() for _ in range(11) )
            yield cls.CHARACTERS[int_value / 45]
            yield cls.CHARACTERS[int_value % 45]
        if character_count % 2 != 0:
            int_value = utils.bin2long_be( bitstream.next() for _ in range(6) )
            yield cls.CHARACTERS[int_value]

class EightBitByteMode(Mode):
    MODE_INDICATOR = 4

    @classmethod
    def character_count_bitsize(cls, version):
        if version < 10: return 8
        if version < 27: return 8
        return 16
    
    @classmethod
    def read(cls, bitstream, character_count):
        for _ in range(character_count):
            value = utils.bin2long_be( bitstream.next() for _ in range(8) )
            yield chr(value)

class KanjiMode(Mode):
    MODE_INDICATOR = 8

    @classmethod
    def character_count_bitsize(cls, version):
        if version < 10: return 8
        if version < 27: return 10
        return 12
    

class Encoder(object):
    def __init__(self):
        self.contents = []
    
    def add_value(self, v):
        self.contents.append(Mode.select_shortest(v))
    
    @classmethod
    def from_string(cls, s):
        retval = cls()
        retval.add_value(s)
        return retval
