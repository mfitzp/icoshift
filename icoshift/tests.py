import icoshift
import numpy as np

test = np.array([
[0,0,0,2,3,2,0,0,2,1,2,np.nan,3,4,2,3,2,0,2,3,2],
[0,0,2,3,2,0,0,2,3,1,2,np.nan,3,4,3,2,0,2,3,2,0],
[0,2,3,2,0,0,2,3,2,1,2,np.nan,3,4,2,0,2,3,2,0,1],
])
xCS,ints,ind,target=icoshift.icoshift('average',test)
print '********** PASS 1: average **********'

test = np.array([
[0,0,0,2,3,2,0,0,2,3,2,np.nan,0,0,2,3,2,0,2,3,2],
[0,0,2,3,2,0,0,2,3,2,0,np.nan,0,2,3,2,0,2,3,2,0],
[0,2,3,2,0,0,2,3,2,0,0,np.nan,2,3,2,0,2,3,2,0,1],
])

xCS,ints,ind,target=icoshift.icoshift('median',test)
print '********** PASS 2: median **********'


