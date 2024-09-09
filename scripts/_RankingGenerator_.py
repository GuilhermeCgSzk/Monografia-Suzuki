import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

from ._Names_ import Names
from ._Generator_ import Generator
from ._DataframeSplitter_ import DataframeSplitter

from ._Model_Names_ import Aeon_Group,Model_Names
from ._Model_Names_  import *


def arrange_columns(df):
	df['projection'] = df['projection'].apply(Names.get_projection_mappings_function())
		
	mappings = {}	
	for name_obj in Model_Names.models_list() + Aeon_Group.model_list:	
		mappings = mappings | name_obj.mappings()
	df = df[df['model'].isin(mappings)].copy()	
	
	df['combination'] = np.where(
		df['projection']!='Not projected', df['model'] + '+' + df['projection'], df['model']
	)
	
	return df

class RankingGenerator(Generator):
	def __init__(self, df, benchmark_df):
		self.df = arrange_columns(df.copy())
		self.benchmark_df = arrange_columns(benchmark_df.copy())


	def generate_plot_per_metric(self, df, name, metric, title, path, dim, limit=None, show_xticks=False, ascending=True):	
		# Set up the figure and axes
		plt.figure(figsize=dim)

		average_df = df[['combination',metric]].groupby('combination').mean().reset_index().sort_values(
			metric, ascending=ascending
		)
		
		if limit is None:
			order = average_df['combination']
		else:
			order = average_df['combination'][-limit:]
		
		sns.barplot(
			data=df, x='combination', y=metric, hue='projection', order=order,
			errorbar=('sd', 1),
			err_kws={'linewidth':0.5},
			estimator='mean',
			palette={
				'RP':(0.5,0.5,1,1), 
				'GAF':(1,0.5,0.5,1), 
				'MTF':(0.5,1,0.5,1), 
				'PMix':(1,0.7,1,1),
				'Not projected':(0.5,0.5,0.5,1)
			},
		)
		
		#ax.set_ylim(-0.35, 1.35)
		#ticks = [i/10 for i in range(0,11,2)]
		#ax.set_xticks(ticks,[f'{i:.1f}' for i in ticks],fontsize=16,rotation=45)

		#ax.set_xlim((-0.3 ,1.3))

		if not show_xticks:
			plt.xticks([])
		else:
			plt.xticks(rotation=60, ha='right', fontsize=16)

		# Set plot labels and title
		#plt.ylabel('', fontsize=20)
		#plt.xlabel(f'{title}', fontsize=20)
		#plt.title(name, fontsize=24)
		#plt.yticks(rotation=30, ha='right', fontsize=20)
		plt.ylabel(title)
		plt.xlabel('')
		plt.legend(loc='lower right',  ncols=5)


		# Save the figure to a file
		plt.savefig(os.path.join(path,f'rankingplot_{metric}_{name}.png'), bbox_inches='tight', dpi=1000)
		plt.close()
		
	def generate(self, path):
		metrics = [
			("accuracy_score", "Accuracy Score"),
			("cohen_kappa_score", "Cohen Kappa Score"),
			("precision_score", "Precision Score"),
			("recall_score", "Recall Score"),
			("f1_score", "F1-Score"),
		]
		
		for column, metric in metrics:
			#for projection in ['GAF','RP','MTF','PMix','Not projected']:
			#	df = self.df.copy()
			#	df = df[df['projection']==projection]
			#	self.generate_plot_per_metric(df, projection, column, metric, path)
			df = self.df.copy()		
			self.generate_plot_per_metric(df, 'all', column, metric, path, dim=(15,2))
			self.generate_plot_per_metric(df, 'top10', column, metric, path, limit=10, show_xticks=True, dim=(7,5))
			
		metrics = [
			("inference_time", "Inference Time"),
			("memory_size (bytes)", "Memory Size (Bytes)"),
		]
		
		for column, metric in metrics:
			#for projection in ['GAF','RP','MTF','PMix','Not projected']:
			#	df = self.df.copy()
			#	df = df[df['projection']==projection]
			#	self.generate_plot_per_metric(df, projection, column, metric, path)
			df = self.benchmark_df.copy()			
			self.generate_plot_per_metric(df, 'all', column, metric, path, dim=(15,2), ascending=False)
			self.generate_plot_per_metric(df, 'top10', column, metric, path, limit=10, show_xticks=True, dim=(7,5), ascending=False)
		
