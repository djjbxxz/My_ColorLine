import numpy as np
import numpy.ctypeslib as npct
from numpy.lib.npyio import load

arr_1 = np.array([1, 2, 3, 4], dtype=np.int)
arr_2 = np.ndarray((4,), dtype=np.int)


def load_PathfindingDLL():
    lib = npct.load_library("PathFindingDllforPython",
                            r"DLL")
    lib.Estimate.argtypes = [npct.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
                             npct.ndpointer(dtype=np.int, ndim=2, flags="C_CONTIGUOUS", shape=(9, 9))]
    print("Pathfinding_DLL已加载！")
    return lib.Estimate

if __name__=='__main__':
    load_PathfindingDLL()