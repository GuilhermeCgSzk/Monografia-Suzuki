import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

from ._Names_ import *
from ._Names_ import Names
from ._Generator_ import Generator
from ._DataframeSplitter_ import DataframeSplitter

from ._Model_Names_ import Aeon_Group,Model_Names
from ._Model_Names_  import *


def arrange_columns(df):
	df['projection'] = df['projection'].apply(Names.get_projection_mappings_function())
	
	df['combination'] = np.where(
		df['projection']!='Not projected', df['model'] + '+' + df['projection'], df['model']
	)
	
	return df

class RankingGenerator(Generator):
	def __init__(self, df, benchmark_df):
		self.df = arrange_columns(df.copy())
		self.benchmark_df = arrange_columns(benchmark_df.copy())
		self.benchmark_df['memory_size (MegaBytes)'] = self.benchmark_df['memory_size (bytes)']/(10**6)
		self.benchmark_df['memory_size (KiloBytes)'] = self.benchmark_df['memory_size (bytes)']/(10**3)

	def generate_plot_per_metric(self, df, name, metric, title, path, dim, order, limit=None, show_xticks=False, ascending=True, ylim=None, loc='lower right', estimator='mean', errorbar=True):	
		# Set up the figure and axes
		plt.figure(figsize=dim)
		
		order = order.sort_values(metric, ascending=ascending)['combination']
		
		if limit is not None:
			order = order[-limit:]
		
		if errorbar:
			errorbar = ('sd', 1)
		else:	
			errorbar = None
		
		
		
		sns.barplot(
			data=df, x='combination', y=metric, hue='projection', order=order,
			errorbar=errorbar,
			err_kws={'linewidth':0.5},
			estimator=estimator,
			palette={
				RP().final_name():(0.5,0.5,1,1), 
				GAF().final_name():(1,0.5,0.5,1), 
				MTF().final_name():(0.5,1,0.5,1), 
				Mix().final_name():(1,0.7,1,1),
				NoProjection().final_name():(0.5,0.5,0.5,1)
			},
		)
		
		#ax.set_ylim(-0.35, 1.35)
		#ticks = [i/10 for i in range(0,11,2)]
		#ax.set_xticks(ticks,[f'{i:.1f}' for i in ticks],fontsize=16,rotation=45)

		if ylim is not None:
			plt.ylim(ylim)

		if not show_xticks:
			plt.xticks([])
		else:
			plt.xticks(rotation=60, ha='right',fontsize=16)

		# Set plot labels and title
		#plt.ylabel('', fontsize=20)
		#plt.xlabel(f'{title}', fontsize=20)
		#plt.title(name, fontsize=24)
		#plt.yticks(rotation=30, ha='right', fontsize=20)
		plt.ylabel(title,fontsize=16)
		plt.xlabel('')
		plt.legend(loc=loc,  ncols=5)


		# Save the figure to a file
		plt.savefig(os.path.join(path,f'rankingplot_{metric}_{name}.png'), bbox_inches='tight', dpi=1000)
		plt.close()
		
	def generate(self, path):
		metrics = [
			("cohen_kappa_score", "Cohen Kappa\nScore"),
			("precision_score", "Precision Score"),
			("f1_score", "F1-Score"),
		]
		
		for column, metric in metrics:
			df = self.df.copy()	
			
			average_df = df[['combination',column]].groupby('combination').mean().reset_index()
			std_df = df[['combination',column]].groupby('combination').std().reset_index()
				
			self.generate_plot_per_metric(
				df, 'all', column, metric, path, dim=(15,2), order=average_df
			)
			self.generate_plot_per_metric(
				df, 'top10', column, metric, path, order=average_df, limit=10, show_xticks=True, dim=(7,5)
			)
			
		metrics = [
			("inference_time", "Inference Time\n(seconds)"),
			("memory_size (MegaBytes)", "Memory Size\n(MegaBytes)"),
			("memory_size (KiloBytes)", "Memory Size\n(KiloBytes)"),
		]
		
		for column, metric in metrics:
			df = self.benchmark_df.copy()			
			
			average_df = df[['combination',column]].groupby('combination').mean().reset_index()
			
			self.generate_plot_per_metric(df, 'all', column, metric, path, order=average_df, dim=(15,2), ascending=False, ylim=(-5,70), loc='lower left')
			self.generate_plot_per_metric(df, 'top10', column, metric, path, order=average_df, limit=10, show_xticks=True, dim=(7,5), ascending=False, loc='upper right')
		
