__author__ = 'me'

class I4_State:
    def __init__(self):
        self.initalized = True
        self.log_max = 30
        self.atmost = 2**self.log_max - 1
        self.dim_max = 40
        self.dim_num_save = -1
        self.seed_save = -1

    maxcol = None
    v = None
    poly = None
    recipd = None




state = I4_State()
a = 1