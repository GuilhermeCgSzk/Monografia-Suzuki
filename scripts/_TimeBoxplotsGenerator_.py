import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from ._Names_ import Names
from ._Generator_ import Generator
from ._DataframeSplitter_ import DataframeSplitter

class TimeBoxplotsGenerator(Generator):
	def __init__(self, df):
		self.df = df.copy()
		self.df = self.df.sort_values(by='model')
		
	def generate(self, path):	
		for model in Names.get_model_list():
			self.generate_for_model(model, path)
		
	def generate_for_model(self, model, path):	
		plt.rcParams["font.family"] = "monospace"
		plt.rcParams["font.monospace"] = ["FreeMono"]
		fontsize=24
	
		h = max(1*len(model.mappings()),2)
		plt.figure(figsize=(10,h))
		
		df = self.df.copy()
		df = df[df['model'].isin(model.mappings())]
		
		df['projection'] = df['projection'].apply(Names.get_projection_mappings_function())
		
		df['model'] = df['model'].apply(lambda x: f'{model.mappings()[x]:>30}')
		
		ax = sns.boxplot(
			df, x='inference_time', y='model', hue='projection', 
	    		hue_order=["GAF", "MTF", "RP", "Mix"], 		
	    		palette=['firebrick','green','#0c5fef','#8c5fef'],
	    		showfliers=False,
	    		orient='h',
	    		fill=False,
	    		gap=0.1,
	    		notch=True,
		)
		plt.title(model.name(), fontsize=fontsize)
	    	
		#yticks = [0,10,20,30,40,50]
	    	
		#plt.yticks(yticks,[str(i) for i in yticks],fontsize=20)
		ax.set_xlim(0,100)
	    	
	    	
		plt.legend(fontsize=fontsize/1.5, ncols=1, loc='upper right')
			
		plt.xticks(fontsize=fontsize)
		plt.xlabel('Inference Time (ms)', fontsize=fontsize)
		plt.yticks(ha='right', fontsize=fontsize)	
		plt.ylabel('')
	    	
		plt.savefig(os.path.join(path,f'benchmark_of_{model.name()}.pdf'), bbox_inches='tight', dpi=500)
		
		plt.close()
