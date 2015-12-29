#!/usr/bin/env python
# -*- coding: utf-8 -*-

import qrtistry, sys, string

outfp = file("debug.html","w")

if __name__ == "__main__":
    qr = qrtistry.QRCode.decode_from_image_file(sys.argv[1])
    print qr
    
    if True:
        print
        print "\n".join(qr.encode_tty(apply_mask=False, mark_special=False))
        print

        print
        print "\n".join(qr.encode_tty(apply_mask=True, mark_special=True))
        print
    
    if False:
        #identifiers = map(lambda a: unichr(0xff00 + ord(a) - 0x20), string.digits + string.ascii_letters)
        colors = ["%02X%02X%02X" % (r,g,b) for r in [0,128,255] for g in [0,128,255] for b in [0,128,255] if r+g+b != 3*255]
        identifiers = [u"<span style='background: #%s; width: 5px; height: 5px; display: inline-block'></span>" % e for e in colors]
        blockmap = [ [u"<span style='width: 5px; height: 5px; display: inline-block'></span>"]*(qr.code_size) for i in range(qr.code_size) ]
        for index,block in enumerate(qr.blocks):
            for i,j in block:
                blockmap[i][j] = identifiers[index % len(identifiers) ]
        
        for line in blockmap:
            print >>outfp, u"".join(line).encode("utf-8"),"<br>"

    if False:
        #identifiers = map(lambda a: unichr(0xff00 + ord(a) - 0x20), string.digits + string.ascii_letters)
        colors = ["%02X%02X%02X" % (g,g,g) for g in [0, 128]]
        identifiers = [u"<span style='background: #%s; width: 5px; height: 5px; display: inline-block'></span>" % e for e in colors]
        blockmap = [ [u"<span style='width: 5px; height: 5px; display: inline-block'></span>"]*(qr.code_size) for i in range(qr.code_size) ]
        for index,(block,cw) in enumerate(zip(qr.blocks,qr.error_correction_sequence.column_order())):
            for i,j in block:
                blockmap[i][j] = identifiers[isinstance(cw,qrtistry.qrcode.DataCodeword) ]
        
        for line in blockmap:
            print >>outfp, u"".join(line).encode("utf-8"),"<br>"

    if False:
        #identifiers = map(lambda a: unichr(0xff00 + ord(a) - 0x20), string.digits + string.ascii_letters)
        colors = ["%02X%02X%02X" % (g,g,g) for g in range(0,254,2)]
        identifiers = [u"<span style='background: #%s; width: 5px; height: 5px; display: inline-block'></span>" % e for e in colors]
        blockmap = [ [u"<span style='width: 5px; height: 5px; display: inline-block'></span>"]*(qr.code_size) for i in range(qr.code_size) ]
        for index,(block,cw) in enumerate(zip(qr.blocks,qr.error_correction_sequence.column_order())):
            for i,j in block:
                blockmap[i][j] = identifiers[cw.num % len(identifiers) ]
        
        for line in blockmap:
            print >>outfp, u"".join(line).encode("utf-8"),"<br>"
    
    
    if False:
        from matplotlib import pyplot as plt
        plt.imshow(qr.debug_image / 255.0, interpolation='nearest')
        plt.show()
    
    if True:
        for bit in qr.get_data_bitstream():
            print int(bit),
        print
    
    if True:
        for cw in qr.error_correction_sequence.column_order():
            print cw
    
    if True:
        for data_row, error_correction_row in qr.error_correction_sequence.rows():
            print [cw.value for cw in data_row], \
                [cw.value for cw in error_correction_row]
    
    if True:
        for item in qr.parse_bitstream(qr.get_data_bitstream()):
            print item

    