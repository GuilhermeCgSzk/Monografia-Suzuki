import seaborn as sns
import matplotlib.pyplot as plt
import os

from ._Names_ import Names
from ._Generator_ import Generator

class ViolinplotGenerator(Generator):
	def __init__(self, df):
		df = df.copy()
		valid_models = Names.model_mappings.keys()
		df = df[df['model'].str.contains('|'.join(valid_models))]
		for substring in valid_models:
			df['model'] = df.apply(lambda row: Names.model_mappings[substring] if substring in row['model'] else row['model'], axis=1)

		df = df.sort_values(by="model")

		df['feature'] = df['study_name'].apply(lambda x: x.split(',')[-1])
		# List of unique features

		accepted_substrings = Names.projection_mappings.keys()
		# Filter the DataFrame based on substrings in the "feature" column
		df = df[df['feature'].str.contains('|'.join(accepted_substrings))]
		for substring in accepted_substrings:
			df['feature'] = df.apply(lambda row: Names.projection_mappings[substring] if substring in row['feature'] else row['feature'], axis=1)
		
		self.df = df

	def generate_plot_per_metric(self, metric, title, path):
		# Set up the figure and axes
		plt.figure(figsize=(18, 4))

		# Create the boxplot using seaborn
		ax = sns.violinplot(
			x='model', y=metric, hue='feature', data=self.df, density_norm='count', 
			hue_order=["GAF", "MTF", "RP", "Mix"], 		
			palette=['firebrick','green','#0c5fef','#8c5fef']
		)
		#ax.set_ylim(-0.35, 1.35)
		ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
		ax.set_yticks(ticks,[f'{i:.1f}' for i in ticks],fontsize=16)


		# Set plot labels and title
		plt.xlabel('', fontsize=20)
		plt.ylabel(title, fontsize=20)
		plt.title(f'', fontsize=24)
		plt.xticks(rotation=30, ha='right', fontsize=20)
		plt.legend(loc='lower right', fontsize=20, ncols=4)


		# Save the figure to a file
		plt.savefig(os.path.join(path,f'violinplot_{metric}.pdf'), bbox_inches='tight', dpi=1000)
		plt.clf()
		
	def generate(self, path):
		metrics = [
			("accuracy_score", "Accuracy Score"),
			("precision_score", "Precision Score"),
			("recall_score", "Recall Score"),
			("f1_score", "F1-Score"),
		]
		for column, metric in metrics:
			self.generate_plot_per_metric(column, metric, path)
