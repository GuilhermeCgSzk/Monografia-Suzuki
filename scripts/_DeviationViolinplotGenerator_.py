import seaborn as sns
import matplotlib.pyplot as plt
import os

from ._Names_ import Names
from ._Generator_ import Generator
from ._DataframeSplitter_ import DataframeSplitter

class DeviationViolinplotGenerator(Generator):
	def __init__(self, df):
		df = df.copy()
		
		df = df.sort_values(by="model")
		df['projection'] = df['projection'].apply(Names.get_projection_mappings_function())
		
		self.df = df

	def generate_plot_per_metric(self, df, name, metric, title, path, q):	
		# Set up the figure and axes
		plt.figure(figsize=(4,7))

		# Create the boxplot using seaborn
		ax = sns.boxplot(
			y=metric, x='projection', hue='projection', data=df, #density_norm='area', 
			order=["GAF", "MTF", "RP", "PMix"],
			hue_order=["GAF", "MTF", "RP", "PMix", "Not projected"], 		
			palette=['firebrick','green','#0c5fef','pink','grey'],
			orient='v'
		)
		# Set plot labels and title
		
		#plt.ylim((-0.1,0.5))
		plt.xlabel('', fontsize=20)
		
		if q > 0:
			title = f'{title}\nStandard Deviation (Quantile {q})'
		else:
			title = f'{title}\nStandard Deviation'
		
		plt.ylabel(title, fontsize=20)
		plt.title(f'', fontsize=24)
		plt.xticks(rotation=30, ha='right', fontsize=20)
		#plt.legend(loc='lower left', fontsize=20, ncols=1, framealpha=0.5, bbox_to_anchor=(1,0))


		# Save the figure to a file
		plt.savefig(os.path.join(path,f'violinplot_{metric}_{name}.png'), bbox_inches='tight', dpi=1000)
		plt.close()
		
	def generate(self, path):
		metrics = [
			("cohen_kappa_score", "Cohen Kappa Score"),
			("precision_score", "Precision Score"),
			("f1_score", "F1-Score"),
		]
		
		for column, metric in metrics:
			for quantile in [0, 0.5]:
				df = self.df.copy()
				df = df[['model','projection',column]].groupby(['model','projection']).mean()
				quantile_value = df.quantile(q=quantile)[column]
				
				above_median_combinations = df[df[column]>=quantile_value].reset_index()[['model','projection']]
				
				df = self.df.copy()
				df = df[df[['model','projection']].apply(tuple, axis=1).isin(above_median_combinations.apply(tuple, axis=1))]
				
				std_df = df[['model','projection',column]].groupby(['model','projection']).std().reset_index()
		
				self.generate_plot_per_metric(std_df, f'std_({quantile})', column, metric, path, q=quantile)
