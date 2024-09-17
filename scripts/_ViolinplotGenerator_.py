import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd
import functools

from ._Names_ import Names
from ._Generator_ import Generator
from ._DataframeSplitter_ import DataframeSplitter

class ViolinplotGenerator(Generator):
	def __init__(self, df):
		df = df.copy()

		df = df.sort_values(by="model")
		df['projection'] = df['projection'].apply(Names.get_projection_mappings_function())
		self.df = df

	def generate_plot_per_metric(self, df, metric_name, metric, title, path, show_yticks=False, show_xticks=False):	
		# Set up the figure and axes
		scalable_size=len(df[['model','projection']].groupby(['model','projection']).any())*0.2
		plt.figure(figsize=(2*(1+scalable_size),4))

		# Create the boxplot using seaborn
		ax = sns.violinplot(
			x='model', y=metric, hue='projection', data=df,  density_norm='count', showmeans=True,
			hue_order=["GAF", "MTF", "RP", "PMix"], 		
			palette=['firebrick','green','#0c5fef','#8c5fef'],
			orient='v',
			#meanprops={"marker":"D","markerfacecolor":"orange", "markeredgecolor":"black", 'markersize':4},
			#flierprops={'markersize':3}
		)
		
		#ax.set_ylim(-0.35, 1.35)
		if show_yticks:
			ticks = [i/10 for i in range(2,11,2)]
			ax.set_yticks(ticks,[f'{i:.1f}' for i in ticks],fontsize=12,rotation=45)
			plt.ylabel(metric_name, fontsize=20)
		else:
			plt.ylabel('')
			ax.set_yticks([])

		#ax.set_ylim((-0.3 ,1.3))

		# Set plot labels and title
		plt.xlabel('', fontsize=16)
		#plt.ylabel(f'{title}', fontsize=20)
		plt.title(title, fontsize=24)
		if show_xticks:
			plt.xticks(rotation=30, ha='right', fontsize=20)
		else:
			plt.xticks([])
		#plt.legend(loc='lower left', fontsize=20, ncols=1, framealpha=0.5, bbox_to_anchor=(1,0))
		plt.legend().remove()

		# Save the figure to a file
		path = os.path.join(path,metric)
		os.makedirs(path, exist_ok=True)
		plt.savefig(os.path.join(path,f'violinplot_{metric}.png'), bbox_inches='tight', dpi=500)
		plt.close()
		
	def generate(self, path):
		metrics = [
			("cohen_kappa_score", "Cohen Kappa\nScore"),
			("f1_score", "\nF1-Score"),
			("precision_score", "Precision\nScore"),
		]
		
		for metrics_i,(column, metric) in enumerate(metrics):
			dfs = []
		
			for i,model in enumerate(Names.get_model_list()):
				dfi = self.df.copy()
				dfi = dfi[dfi['model'].isin(model.mappings())]
				dfi['model'] = dfi['model'].apply(lambda x: model.mappings()[x])
				
				dfagg = dfi[['model','projection',column]].groupby(['model','projection'])
				dfagg = dfagg.agg(['mean','std']).reset_index()
				
				combinations = []
				
				for projection in ["GAF", "MTF", "RP", "PMix"]:
					dfagg_proj = dfagg[dfagg['projection']==projection].copy()
					dfagg_proj.to_csv('k.csv')
					dfagg_proj = dfagg_proj.sort_values(by=[(column,'mean'),(column,'std')],ascending=[False,True])
					dfagg_proj.to_csv('kk.csv')
					model_str = dfagg_proj['model'].iloc[0]
					combinations.append((model_str,projection))
					
				dfi = dfi[dfi[['model','projection']].apply(tuple, axis=1).isin(combinations)]
				dfi['model'] = model.name()
				dfs.append(dfi)
				
			df = pd.concat(dfs)
			
			show_xticks = (metrics_i==len(metrics)-1)
			self.generate_plot_per_metric(df, metric, column, '', path, show_yticks=True, show_xticks=show_xticks)	
