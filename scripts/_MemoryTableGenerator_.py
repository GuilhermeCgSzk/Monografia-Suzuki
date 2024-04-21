import pandas as pd
import os
from ._Names_ import Names
from ._Generator_ import Generator
from ._DataframeSplitter_ import DataframeSplitter

class MemoryTableGenerator(Generator):
	def __init__(self, df):
		self.df = df.copy()
		
	def generate(self, path):
		for group in Names.get_group_list():
			self.generate_per_group(group, path)
	
	def generate_per_group(self, group, path):
		df = self.df.copy()		
		
		df = df[df['model'].isin(group.mappings())]
		
		df = df[['model','projection','memory_size (bytes)']].groupby(['model','projection'], as_index=False).max()
		df['memory_size (Megabytes)'] = df['memory_size (bytes)'].apply(lambda x: x/10**6)
		
		df['use_mix'] = df['projection'] == 'ProjectionMix_V2'
		
		
		df = df[['model','use_mix','memory_size (Megabytes)']].groupby(['model','use_mix'], as_index=False).max()
		
		df['use_mix'] = df['use_mix'].apply(lambda x: {False:'Usual', True:'Proposed'}[x])
		df['model'] = df['model'].apply(lambda x: group.mappings()[x])
		
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
				'memory_size (Megabytes)': '\\begin{tabular}{c}Memory \\\\Size (MB)\\end{tabular}',
			},
			axis=1
		)
		
		kwargs = {
			'index':False,
			'na_rep':'',	
		}
				
		
		for i,dfi in enumerate(DataframeSplitter(3).split(df)):
			directory = os.path.join(path,group.name())
			os.makedirs(directory, exist_ok=True)
			full_path = os.path.join(directory,f'memory_table_{i+1}.tex')
			dfi.to_latex(full_path, **kwargs)
	
	
	
	
