import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from ._Names_ import Names
from ._Names_ import *
from ._Generator_ import Generator
from ._DataframeSplitter_ import DataframeSplitter

class TimeBoxplotsGenerator(Generator):
	def __init__(self, df):
		self.df = df.copy()
		
		self.df = self.df.sort_values(by='model')
		
	def generate(self, path):	
		for model in Names.get_model_list():
			self.generate_for_name_obj(model, path)

	def generate_per_pair_group(self, path, pair_group):
		self.generate_for_name_obj(pair_group.get_group(), path, filter_obj=pair_group)
		
	def generate_for_name_obj(self, name_obj, path, filter_obj=None):	
		plt.rcParams["font.family"] = "monospace"
		plt.rcParams["font.monospace"] = ["FreeMono"]
		fontsize=24
		
		df = self.df.copy()
		
		if filter_obj is not None:
			df = filter_obj.filter(df)		
		
		df = df[df['model'].isin(name_obj.mappings())]
		
		df['projection'] = df['projection'].apply(Names.get_projection_mappings_function())
		
		df['model'] = df['model'].apply(lambda x: f'{name_obj.mappings()[x]:>30}')
		
		
		projections = df['projection'].unique()
	
		h = max(0.25*(len(name_obj.mappings())*max(2,len(projections))),2)
		plt.figure(figsize=(10,h))
		
		hue_dict = {
			GAF().final_name(): 'firebrick',
			MTF().final_name(): 'green',
			RP().final_name(): '#0c5fef',
			Mix().final_name(): '#8c5fef', 
			NoProjection().final_name(): 'black',
		}
		
		ax = sns.boxplot(
			df, x='inference_time', y='model', hue='projection', 
	    		hue_order=[p for p in hue_dict if p in projections], 		
	    		palette=[hue_dict[p] for p in hue_dict if p in projections],
	    		showfliers=False,
	    		orient='h',
	    		fill=False,
	    		gap=0.1,
	    		notch=True,
		)
		plt.title(name_obj.name(), fontsize=fontsize)
	    	
		#yticks = [0,10,20,30,40,50]
	    	
		#plt.yticks(yticks,[str(i) for i in yticks],fontsize=20)
		#ax.set_xlim(0,100)
	    	
	    	
		plt.legend(fontsize=fontsize/1.5, ncols=1, loc='upper right')
			
		plt.xticks(fontsize=fontsize)
		plt.xlabel('Inference Time (ms)', fontsize=fontsize)
		plt.yticks(ha='right', fontsize=fontsize)	
		plt.ylabel('')
	    	
		plt.savefig(os.path.join(path,f'benchmark_of_{name_obj.name()}.pdf'), bbox_inches='tight', dpi=500)
		
		plt.close()
