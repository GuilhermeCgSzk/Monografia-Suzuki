import seaborn as sns
import matplotlib.pyplot as plt
import os

from ._Names_ import Names
from ._Generator_ import Generator
from ._DataframeSplitter_ import DataframeSplitter

class ViolinplotGenerator(Generator):
	def __init__(self, df):
		df = df.copy()

		df = df.sort_values(by="model")
		df['projection'] = df['projection'].apply(Names.get_projection_mappings_function())
		self.df = df

	def generate_plot_per_metric(self, df, name, metric, title, path):
	
		
	
		# Set up the figure and axes
		scalable_size=len(df[['model','projection']].groupby(['model','projection']).any())*0.5
		plt.figure(figsize=(5,1+scalable_size))

		# Create the boxplot using seaborn
		ax = sns.violinplot(
			x=metric, y='model', hue='projection', data=df, density_norm='count', 
			hue_order=["GAF", "MTF", "RP", "Mix"], 		
			palette=['firebrick','green','#0c5fef','#8c5fef'],
			orient='h'
		)
		#ax.set_ylim(-0.35, 1.35)
		ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
		ax.set_xticks(ticks,[f'{i:.1f}' for i in ticks],fontsize=16,rotation=45)

		ax.set_xlim((-0.3 ,1.3))

		# Set plot labels and title
		plt.ylabel('', fontsize=20)
		plt.xlabel(title, fontsize=20)
		plt.title(name, fontsize=24)
		plt.yticks(rotation=30, ha='right', fontsize=20)
		plt.legend(loc='lower left', fontsize=20, ncols=1, framealpha=0.5, bbox_to_anchor=(1,0))


		# Save the figure to a file
		plt.savefig(os.path.join(path,f'violinplot_{metric}_{name}.pdf'), bbox_inches='tight', dpi=1000)
		plt.close()
		
	def generate(self, path):
		metrics = [
			("accuracy_score", "Accuracy Score"),
			("precision_score", "Precision Score"),
			("recall_score", "Recall Score"),
			("f1_score", "F1-Score"),
		]
		
		for column, metric in metrics:
			for group in Names.get_group_list():
				dfi = self.df.copy()
				dfi = dfi[dfi['model'].isin(group.mappings())]
				dfi['model'] = dfi['model'].apply(lambda x: group.mappings()[x])
				
				self.generate_plot_per_metric(dfi, group.name(), column, metric, path)
