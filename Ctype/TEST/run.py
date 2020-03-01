from geometry import *

r = Rectangle(3, 4)

print(r.area())
r.width = 10
print(r.area())

import gmpy2

m = gmpy2.mpz(12347981237489237489326798)

data = CGBN_MEM(2048, m);
print (data.length)
print (data.val)