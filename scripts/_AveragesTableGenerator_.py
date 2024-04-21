import pandas as pd
import os
import math

from ._Names_ import Names
from ._Generator_ import Generator

metrics = ["cohen_kappa_score","f1_score","precision_score"] # ,"recall_score""accuracy_score",

class AveragesTableGenerator(Generator):
	def __init__(self, df):
		df = df.copy()
	
		df['projection'] = df['projection'].apply(Names.get_projection_mappings_function())	
			
		df_grouped = df.groupby(['model','projection'],as_index=False).max()
		
		for metric in metrics:
			df_grouped[self.get_mean_key(metric)] = df[['model','projection',metric]].groupby(['model','projection'],as_index=False).mean()[metric]	
			
		for metric in metrics:
			df_grouped[self.get_std_key(metric)] = df[['model','projection',metric]].groupby(['model','projection'],as_index=False).std()[metric]
			
		self.df = df_grouped
		
	def get_std_key(self, metric_key):
		return f'{metric_key}_std'
		
	def get_mean_key(self, metric_key):
		return f'{metric_key}_mean'
		
	def generate(self, path):
		for model in Names.get_model_list():
			self.generate_for_group(model, path)
		
	def generate_for_group(self, group, path):
		df = self.df.copy()
		
		df = df[df['model'].isin(group.mappings())]
		df['model'] = df['model'].apply(lambda x: group.mappings()[x])
	
		for metric in metrics:
			minimum_mean,maximum_mean = df[self.get_mean_key(metric)].min(),df[self.get_mean_key(metric)].max()
			minimum_std,maximum_std = df[self.get_std_key(metric)].min(),df[self.get_std_key(metric)].max()
			
			loc = df.columns.get_loc(metric)
			loc_mean = df.columns.get_loc(self.get_mean_key(metric))
			loc_std = df.columns.get_loc(self.get_std_key(metric))
			
			for i in range(len(df)):    						
				mean,std = df.iloc[i,loc_mean],df.iloc[i,loc_std]
		
				blue_factor = 0 
				
				mean_scale = (mean-minimum_mean)/(1 if math.isclose(maximum_mean,minimum_mean) else maximum_mean-minimum_mean  )
				string_mean = (
					'\\textcolor[rgb]{'+f'{max(1-mean_scale,0):.10f},{min(mean_scale,0.5):.10f},{blue_factor}'+'}{' +
						f'{mean:.3f}' + 
					'}' 
				)
				
				std_scale = (std-minimum_std)/(1 if math.isclose(maximum_std,minimum_std) else maximum_std-minimum_std  )
				string_std = (
					'\\textcolor[rgb]{'+f'{max(std_scale,0):.10f},{min(1-std_scale,0.5):.10f},{blue_factor}'+'}{' +
						 f'{std:.3f}' +
					'}' 					
		    		)
		    		
		    		
				if mean == maximum_mean:
					df.iloc[i,loc_mean] = '\\textbf'+'{'+string_mean+'}'	
				else:
					df.iloc[i,loc_mean] = string_mean
				
				if mean >= 0.9:
					df.iloc[i,loc_mean] = '\\underline{'+df.iloc[i,loc_mean]+'}'
				
					
				if df.iloc[i,loc_std] == minimum_std:
					df.iloc[i,loc_std] = '\\textbf'+'{'+string_std+'}'	
				else:
					df.iloc[i,loc_std] = string_std
					
				
					
				df.iloc[i,loc] = f'{df.iloc[i,loc_mean]} $\\pm$ {df.iloc[i,loc_std]}'
	    		    	
			
		df = df[['model','projection']+metrics]

	    		    	
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
			'longtable':False,
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
