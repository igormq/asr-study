from common.utils import safe_mkdirs
import os

DT_ABSPATH = os.path.join(os.path.sep.join(os.path.dirname(os.path.abspath(__file__)).split(os.path.sep)[:-1]), '.datasets')

safe_makedirs(DT_ABSPATH)
