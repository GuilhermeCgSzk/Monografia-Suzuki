import pandas as pd
import os
from ._Names_ import Names
from ._Generator_ import Generator

class MemoryTableGenerator(Generator):
	def __init__(self, df):
		self.df = df.copy()
		
	def generate(self, path):
		df = self.df.copy()		
		
		df = df[['model','projection','memory_size (bytes)']].groupby(['model','projection'], as_index=False).max()
		df['memory_size (Megabytes)'] = df['memory_size (bytes)'].apply(lambda x: x//10**6)
		
		df['use_mix'] = df['projection'] == 'ProjectionMix_V2'
		
		
		df = df[['model','use_mix','memory_size (Megabytes)']].groupby(['model','use_mix'], as_index=False).max()
		
		df['use_mix'] = df['use_mix'].apply(lambda x: {False:'Usual', True:'Proposed'}[x])
		df['model'] = df['model'].apply(Names.get_model_mappings_function())
		
		df = df.drop('use_mix', axis=1)
		
		def agg_func(x):
			values = []
			for value in x:
				values.append(value)
			return values	
			
		df = df.groupby('model', as_index=False).agg(agg_func)
		
		df['memory_size (Megabytes)'] = df['memory_size (Megabytes)'].apply(lambda x: x[0])
		
		df = df.rename(
			{
				'model':'Neural Network', 
				'memory_size (Megabytes)': 'Memory Size (MB)',
			},
			axis=1
		)
		
		fraction = len(df)//3
		split_index1,split_index2 = fraction, 2*fraction
		df1,df2,df3 = df[:split_index1],df[split_index1:split_index2],df[split_index2:]
		
		kwargs = {'index':False}
		
		def make_path(tail):
			return os.path.join(path,tail)
		
		df1.to_latex(make_path('memory_table_1.tex'), **kwargs)
		df2.to_latex(make_path('memory_table_2.tex'), **kwargs)
		df3.to_latex(make_path('memory_table_3.tex'), **kwargs)
	
	
	
	
