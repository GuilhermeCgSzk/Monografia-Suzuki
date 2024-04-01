import pandas as pd
import os

from ._Names_ import Names
from ._Generator_ import Generator

metrics = ["accuracy_score","f1_score", "precision_score","recall_score"]

class AveragesTableGenerator(Generator):
	def __init__(self, df):
		df = df.copy()
	
		df = df[['model','projection']+metrics].groupby(['model','projection'], as_index=False, dropna=False).mean()
		df['model'] = df['model'].apply(Names.get_model_mappings_function())
		df['projection'] = df['projection'].apply(Names.get_projection_mappings_function())

		self.df = df
		
	def generate(self, path):
		df = self.df.copy()
	
		for metric in metrics:
			maximum = df[metric].max()
			for i in range(len(df)):
				loc = df.columns.get_loc(metric)
		    		
				string = f'{df.iloc[i,loc]:.3f}'
		    		
				if df.iloc[i,loc] == maximum:
					df.iloc[i,loc] = '\\textbf'+'{'+string+'}'	
				else:
					df.iloc[i,loc] = string
	    		    	
		df = df.rename(columns=Names.metric_mappings)
	    
		for i in range(len(df)):
			if i%4!=0:
				df.iat[i,0] = ''
	    		
		for i in range(len(df)):
			if i%4==2:
				df.iloc[i],df.iloc[i+1]= df.iloc[i+1].copy(),df.iloc[i].copy()

		dfs = [None for i in range(1)]
		fraction = ((len(df)//4)//len(dfs))*4
		for i in range(0,len(dfs)):
			dfs[i] = df.iloc[fraction*i:fraction*(i+1)]
	    
		kwargs = {
			'index':False,
			'position':'t',
			'longtable':True
		}
	    
		def to_latex(x):
			y = x.to_latex(**kwargs)
			y = y.replace('\\begin{table}[t]','')
			y = y.replace('\\end{table}','')
			y = y.replace('\\begin{tabular}','\\begin{tabular}[t]')
			return y
	    
		for i,dfx in enumerate(dfs):
			with open(os.path.join(path,f'average{i+1}.tex'),'w') as f:
				f.write(to_latex(dfx))
