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

	def generate_per_pair_group(self, path, pair_group):
		self.generate_per_group(pair_group.get_group(), path, filter_obj=pair_group)
	
	def generate_per_group(self, group, path, filter_obj=None):
		df = self.df.copy()		
		
		if filter_obj is not None:
			df = filter_obj.filter(df)
		
		df = df[df['model'].isin(group.mappings())]
		df['model'] = df['model'].apply(lambda x: group.mappings()[x])
		
		df = df[['model','memory_size (bytes)']].groupby('model', as_index=False).max()
		
		df['memory_size (Megabytes)'] = df['memory_size (bytes)'].apply(lambda x: x/10**6)
		df = df.drop('memory_size (bytes)', axis=1)
		
		
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
	
	
	
	
