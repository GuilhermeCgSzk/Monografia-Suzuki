import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from ._Names_ import Names
from ._Generator_ import Generator

class TimeBoxplotsGenerator(Generator):
	def __init__(self, df):
		self.df = df.copy()
		
	def generate(self, path):				
		df = self.df.copy()
		
		plt.figure(figsize=(18,4))
		
		df['projection'] = df['projection'].apply(lambda x: Names.projection_mappings[x])
		df['model'] = df['model'].apply(lambda x: Names.model_mappings[x])
		
		ax = sns.boxplot(
			df, x='model', y='inference_time', hue='projection', 
	    		hue_order=["GAF", "MTF", "RP", "Mix"], 		
	    		palette=['firebrick','green','#0c5fef','#8c5fef'],
	    		showfliers=False,
	    		fill=False,
	    		gap=0.1,
	    	)
	    	
		yticks = [0,10,20,30,40,50]
	    	
		plt.yticks(yticks,[str(i) for i in yticks],fontsize=20)
	    	
		plt.xlabel('', fontsize=20)
		plt.ylabel('Inference Time (ms)', fontsize=20)
		plt.title(f'', fontsize=24)
		plt.xticks(rotation=30, ha='right', fontsize=20)
		plt.legend(loc='upper right', fontsize=20, ncols=4)
	    	
		plt.savefig(os.path.join(path,'benchmark.pdf'), bbox_inches='tight', dpi=1000)
