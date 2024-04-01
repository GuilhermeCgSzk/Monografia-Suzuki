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
		
	def generate(self, path):				
		df = self.df.copy()
		
		df = df.sort_values(by='model')
		
		for i,dfi in enumerate(DataframeSplitter(5).split(df)):
			self.generate_fraction(path,dfi,i+1)
		
	def generate_fraction(self, path, df, number):	
		plt.figure(figsize=(5,10))
		
		df['projection'] = df['projection'].apply(Names.get_projection_mappings_function())
		df['model'] = df['model'].apply(Names.get_model_mappings_function())
		
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
		plt.title('')
	    	
		#yticks = [0,10,20,30,40,50]
	    	
		#plt.yticks(yticks,[str(i) for i in yticks],fontsize=20)
		ax.set_xlim(0,100)
	    	
		plt.legend(fontsize=12, ncols=1)
			
		plt.xlabel('Inference Time (ms)', fontsize=12)
		plt.yticks(rotation=45, ha='right', fontsize=12)	
		plt.ylabel('')
	    	
		plt.savefig(os.path.join(path,f'benchmark{number}.pdf'), bbox_inches='tight', dpi=1000)
		plt.close()
