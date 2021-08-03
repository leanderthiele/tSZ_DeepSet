import ctypes as ct

_libprtfinder = ct.CDLL('./libprtfinder.so')

# the main function
prtfinder = _libprtfinder.prtfinder

# returns pointer to the particle indices
prtfinder.restype = ct.POINTER(ct.c_uint64)

# arguments (some are return values)
prtfinder.argtypes = [np.ctypeslib.ndpointer(dtype=ct.c_float, ndim=1, flags='C_CONTIGUOUS'), # TNG coordinates
                      ct.c_float, # radius of the sphere
                      np.ctypeslib.ndpointer(dtype=ct.c_float, ndim=2, flags='C_CONTIGUOUS'), # DM coordinates
                      ct.c_uint64, # number of particles
                      np.ctypeslib.ndpointer(dtype=ct.c_float, ndim=1, flags='C_CONTIGUOUS'), # ul_corner
                      ct.c_float, # extent
                      np.ctypeslib.ndpointer(dtype=ct.c_uint64, ndim=1, flags='C_CONTIGUOUS'), # offsets
                      ct.POINTER(ct.c_uint64), # length of the returned array
                      ct.POINTER(ct.c_int), # error flag
                     ]

# to free memory
myfree = _libprtfinder.myfree
myfree.restype = None
myfree.argtypes = [ct.POINTER(ct.c_uint64), ]
