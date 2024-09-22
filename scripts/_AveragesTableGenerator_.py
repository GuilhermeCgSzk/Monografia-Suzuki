import pandas as pd
import numpy as np
import os
import math

from ._Names_ import Names,NoProjection
from ._Generator_ import Generator

metrics = ["cohen_kappa_score","f1_score","precision_score"] # ,"recall_score""accuracy_score",

class AveragesTableGenerator(Generator):
	def __init__(self, df):
		df = df.copy()	
			
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
			self.generate_for_name_obj(model, path)
		
	def generate_per_pair_group(self, path, pair_group):
		self.generate_for_name_obj(pair_group.get_group(), path, filter_obj=pair_group, delete_repeated=False)
		
	def generate_for_name_obj(self, name_obj, path, *, filter_obj=None, delete_repeated=True):
		df = self.df.copy()			
		
		if filter_obj is not None:
			df = filter_obj.filter(df)
			
		df = df[df['model'].isin(name_obj.mappings())]
		df['model'] = df['model'].apply(lambda x: name_obj.mappings()[x])
				
		old_df = df.copy()	
	
		for metric in metrics:			
			def get_color(value ,*, get_key_function, inverted=False):
				colors = {
					'green': {'r':0.0,'g':0.7,'b':0.0},
					'red': {'r':1.0,'g':0.2,'b':0.2},
					'neutral': {'r':0.3,'g':0.3,'b':0.5},
				}
				
				if value <= old_df[get_key_function(metric)].quantile(0.25):
					if inverted:
						return colors['green']
					else: 
						return colors['red']
				elif value >= old_df[get_key_function(metric)].quantile(0.75):
					if inverted:
						return colors['red']
					else:
						return colors['green']
				else:
					return colors['neutral']
			
			loc = old_df.columns.get_loc(metric)
			loc_mean = old_df.columns.get_loc(self.get_mean_key(metric))
			loc_std = old_df.columns.get_loc(self.get_std_key(metric))
			
			for i in range(len(df)):    						
				mean,std = old_df.iloc[i,loc_mean],old_df.iloc[i,loc_std]
				
				mean_color = get_color(mean, get_key_function=self.get_mean_key)
					
				string_mean = (
					'\\textcolor[rgb]{'+f'{mean_color["r"]:.10f},{mean_color["g"]:.10f},{mean_color["b"]}'+'}{' +
						f'{mean:.3f}' + 
					'}' 
				)
				
				std_color = get_color(std, get_key_function=self.get_std_key,inverted=True)
				
				string_std = (
					'\\textcolor[rgb]{'+f'{std_color["r"]:.10f},{std_color["g"]:.10f},{std_color["b"]}'+'}{' +
						 f'{std:.3f}' +
					'}' 					
		    		)
		    		
		    		
				if mean == old_df[self.get_mean_key(metric)].max():
					df.iloc[i,loc_mean] = '\\textbf'+'{'+string_mean+'}'	
				else:
					df.iloc[i,loc_mean] = string_mean				
					
				if std	 == old_df[self.get_std_key(metric)].min():
					df.iloc[i,loc_std] = '\\textbf'+'{'+string_std+'}'	
				else:
					df.iloc[i,loc_std] = string_std
					
				
					
				df.iloc[i,loc] = f'{df.iloc[i,loc_mean]} $\\pm$ {df.iloc[i,loc_std]}'
				
		df = df[['model','projection']+metrics]
	    
		projections = df['projection'].unique()
		
		df['projection'] = df['projection'].apply(Names.get_projection_mappings_function())
		
		if len(projections)==1 and projections[0]==NoProjection().name():
			df = df.drop('projection',axis=1)
	    		    	
		df = df.rename(columns=Names.get_metric_mappings_function())
	    
		if delete_repeated:
			for i in range(len(df)):
				if i%4!=0:
					df.iat[i,0] = ''
		    		
			for i in range(len(df)):
				if i%4==2:
					df.iloc[i],df.iloc[i+1]= df.iloc[i+1].copy(),df.iloc[i].copy()
	    	
	    		    
		kwargs = {
			'index':False,
			'position':'t',
			'longtable':False,
		}
	    
		def to_latex(x):
			y = x.to_latex(**kwargs)
			y = y.replace('\\begin{table}[t]','')
			y = y.replace('\\end{table}','')
			return y
	    
		with open(os.path.join(path,f'Averages_of_{name_obj.name()}.tex'),'w') as f:
			f.write(to_latex(df))
