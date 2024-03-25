from ._Parser_ import Parser

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
		
		return df
