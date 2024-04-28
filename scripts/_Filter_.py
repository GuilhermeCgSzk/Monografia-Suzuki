import pandas as pd
from abc import ABC, abstractmethod

class Filter(ABC):
	@abstractmethod
	def filter(self, df: pd.DataFrame) -> pd.DataFrame:
		pass
