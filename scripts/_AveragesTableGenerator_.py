import pandas as pd
import os

from ._Names_ import Names
from ._Generator_ import Generator

metrics = ["accuracy_score","f1_score", "precision_score","recall_score"]

def aggregator(series):
	return (series.mean(),series.std())

class AveragesTableGenerator(Generator):
	def __init__(self, df):
		df = df.copy()
	
		df = df[['model','projection']+metrics].groupby(['model','projection'], as_index=False, dropna=False).agg(aggregator)
	
		df['projection'] = df['projection'].apply(Names.get_projection_mappings_function())

		self.df = df
		
	def generate(self, path):
		for group in Names.get_group_list():
			self.generate_for_group(group, path)
		
	def generate_for_group(self, group, path):
		df = self.df.copy()
		
		df = df[df['model'].isin(group.mappings())]
		df['model'] = df['model'].apply(lambda x: group.mappings()[x])
	
		for metric in metrics:
			maximum = df[metric].max()[0]
			for i in range(len(df)):
				loc = df.columns.get_loc(metric)
		    		
				string = f'{df.iloc[i,loc][0]:.3f} $\\pm$ {df.iloc[i,loc][1]:.3f}'
		    		
				if df.iloc[i,loc][0] == maximum:
					df.iloc[i,loc] = '\\textbf'+'{'+string+'}'	
				else:
					df.iloc[i,loc] = string
	    		    	
		df = df.rename(columns=Names.get_metric_mappings_function())
	    
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
			'longtable':True,
			'caption':f'All folds averages for each model in {group.name()}'
		}
	    
		def to_latex(x):
			y = x.to_latex(**kwargs)
			y = y.replace('\\begin{table}[t]','')
			y = y.replace('\\end{table}','')
			y = y.replace('\\begin{tabular}','\\begin{tabular}[t]')
			return y
	    
		for i,dfx in enumerate(dfs):
			with open(os.path.join(path,f'Averages_of_{group.name()}.tex'),'w') as f:
				f.write(to_latex(dfx))
