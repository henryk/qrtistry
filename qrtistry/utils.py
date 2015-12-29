#!/usr/bin/env python

import subprocess, os, sys, re, tempfile, scipy.misc

def bin2long_be(seq):
    if seq is None: return seq
    result = 0
    for i in seq:
        result = result * 2
        if i:
            result = result + 1
    return result


class PointFinderZXing(object):
    """Can use ZXing library command line tool to find points"""
    LIBRARIES = ["javase/target/javase-3.2.2-SNAPSHOT.jar","core/target/core-3.2.2-SNAPSHOT.jar","zxingorg/target/zxingorg-3.2.2-SNAPSHOT/WEB-INF/lib/jcommander-1.48.jar"]
    POINT_RE = re.compile(r'^\s*Point\s+(?P<id>\d+)\s*:\s+\((?P<x>[0-9.]+)\s*,\s*(?P<y>[0-9.]+)\s*\)')
    
    def __init__(self, zxing_base=None):
        self.zxing_base = zxing_base or os.environ.get("ZXING_LIBRARY", ".")
    
    def find_file(self, image_url):
        cmd = ["java", "-cp", 
            ":".join(os.path.join(self.zxing_base, e) for e in self.LIBRARIES), 
            "com.google.zxing.client.j2se.CommandLineRunner", image_url]
        
        output = subprocess.check_output(cmd)
        
        result = []
        for line in output.splitlines():
            m = self.POINT_RE.match(line)
            if not m: continue
            result.append( map(float, (m.group("x"), m.group("y"))) )
        
        return result
    
    def find_image(self, image):
        with tempfile.NamedTemporaryFile() as t:
            scipy.misc.imsave(t, image, "PNG")
            t.flush()
            return self.find_file(t.name)

if __name__ == "__main__":
    pf = PointFinder()
    print pf.find(sys.argv[1])
