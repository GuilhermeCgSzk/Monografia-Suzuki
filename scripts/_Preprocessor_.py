from ._Parser_ import Parser
from ._Names_ import NoProjection

class Preprocessor:
	def __init__(self, dataframe):
		self.df = Preprocessor.process(dataframe.copy())
		
	def get(self):
		return self.df.copy()
		
	def process(df):  	
	
		def f(x, idx):
			return Parser.parse_study_name(x)[idx]
	    		
		df['model'] = 		df['study_name'].apply(lambda x: f(x,0))
		df['projection'] =	df['study_name'].apply(lambda x: f(x,1))
		
		df['cohen_kappa_score'] -= -1
		df['cohen_kappa_score'] /= 2
		
		
		df['projection'] = df['projection'].fillna(NoProjection().name())
		
		return df
