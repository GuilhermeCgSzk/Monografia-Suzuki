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

		half = ((len(df)//4+1)//2)*4
		df1,df2 = df.iloc[:half],df.iloc[half:]
	    
		kwargs = {
			'index':False,
			'position':'t'
		}
	    
		def to_latex(x):
			y = x.to_latex(**kwargs)
			y = y.replace('\\begin{table}[t]','')
			y = y.replace('\\end{table}','')
			y = y.replace('\\begin{tabular}','\\begin{tabular}[t]')
			return y
	    
		with open(os.path.join(path,'average1.tex'),'w') as f:
			f.write(to_latex(df1))
	    	
		with open(os.path.join(path,'average2.tex'),'w') as f:
			f.write(to_latex(df2))
