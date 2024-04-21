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
		scalable_size=len(df[['model','projection']].groupby(['model','projection']).any())*0.3
		plt.figure(figsize=(5,1+scalable_size))

		# Create the boxplot using seaborn
		ax = sns.violinplot(
			x=metric, y='model', hue='projection', data=df, density_norm='count', 
			hue_order=["GAF", "MTF", "RP", "Mix"], 		
			palette=['firebrick','green','#0c5fef','#8c5fef'],
			orient='h'
		)
		#ax.set_ylim(-0.35, 1.35)
		ticks = [i/10 for i in range(0,11,2)]
		ax.set_xticks(ticks,[f'{i:.1f}' for i in ticks],fontsize=16,rotation=45)

		ax.set_xlim((-0.3 ,1.3))

		# Set plot labels and title
		plt.ylabel('', fontsize=20)
		plt.xlabel(f'{title}', fontsize=20)
		plt.title(name, fontsize=24)
		plt.yticks(rotation=30, ha='right', fontsize=20)
		plt.legend(loc='lower left', fontsize=20, ncols=1, framealpha=0.5, bbox_to_anchor=(1,0))


		# Save the figure to a file
		plt.savefig(os.path.join(path,f'violinplot_{metric}_{name}.pdf'), bbox_inches='tight', dpi=1000)
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
			for model in Names.get_model_list():
				dfi = self.df.copy()
				dfi = dfi[dfi['model'].isin(model.mappings())]
				dfi['model'] = dfi['model'].apply(lambda x: model.mappings()[x])
				
				self.generate_plot_per_metric(dfi, model.name(), column, metric, path)
