import os

DT_ABSPATH = os.path.join(os.path.sep.join(os.path.dirname(os.path.abspath(__file__)).split(os.path.sep)[:-1]), '.datasets')

try:
    os.makedirs(DT_ABSPATH)
except OSError, e:
    if e.errno != 17: # 17 = file exists
        raise
