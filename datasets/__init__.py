from __future__ import absolute_import, division, print_function

from utils.generic_utils import safe_mkdirs
import os

DT_ABSPATH = os.path.join(os.path.sep.join(os.path.dirname(os.path.abspath(__file__)).split(os.path.sep)[:-1]), '.datasets')
safe_mkdirs(DT_ABSPATH)

from datasets.dataset_parser import DatasetParser
from datasets.sid import Sid
from datasets.lapsbm import LapsBM
from datasets.voxforge import VoxForge
from datasets.cslu import CSLU
from datasets.dummy import Dummy
from datasets.brsd import BRSD
