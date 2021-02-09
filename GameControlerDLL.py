import numpy as np
import numpy.ctypeslib as npct

def load_GameControlerDLL():
    lib = npct.load_library("GameControlerDLL",
                            r"DLL")
    lib.judge.argtypes = [npct.ndpointer(dtype=np.int, ndim=2, flags="C_CONTIGUOUS", shape=(9, 9)),
                             npct.ndpointer(dtype=np.int, ndim=1, flags="C_CONTIGUOUS"),
                             npct.ndpointer(dtype=np.int, ndim=2, flags="C_CONTIGUOUS",shape=(2, 2))]
    lib.judge.restypes=[np.int]
    print("Controler_DLL已加载！")
    return lib.judge


if __name__== '__main__':
    arr_1 = np.array([1,2,3], dtype=np.int)
    arr_2 = np.array([[8,8],[0,5]], dtype=np.int)
    arr_3 = np.zeros(shape = (9,9),dtype=np.int)
    for i in range(4):
        arr_3[0][i] = 1
    arr_3[8][8] = 1
    judge = load_GameControlerDLL()
    print(judge(arr_3,arr_1,arr_2))
    print(arr_3)
