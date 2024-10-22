import pandas as pd

from cate.utils import PathLinker
from cate.dataset import Dataset

pathlinker = PathLinker()


df = pd.read_csv(pathlinker.data.lenta.origin)

df