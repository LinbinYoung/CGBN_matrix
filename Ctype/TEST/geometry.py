import ctypes as C
import gmpy2
clib = C.CDLL('./libgeom.so')
clib.area.restype = C.c_float
clib.area.argtypes = [C.Structure]

class Rectangle(C.Structure):
    _fields_ = [
        ("width", C.c_float),
        ("height", C.c_float)
    ]
    def __init__ (self, width, height):
        self.width = width
        self.height = height
    
    def area(self):
        return clib.area(self)

class CGBN_MEM(C.Structure):
    _fields_ = [
        ("length", C.c_int64),
        ("val", C.c_char_p)
    ]
    def __init__ (self, len, num):
        if isinstance(num, mpz):
            self.val = C.c_char_p(int(num).to_bytes(len//8, 'little'))
        elif isinstance(num, int):
            self.val = C.c_char_p(num.to_bytes(len//8, 'little'))
        self.length = len