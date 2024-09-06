import seaborn as sns
import matplotlib.pyplot as plt
import os

from ._Names_ import Names
from ._Generator_ import Generator
from ._DataframeSplitter_ import DataframeSplitter

class RankingGenerator(Generator):
	def __init__(self, df):
		df = df.copy()

		df = df.sort_values(by="model")
		df['projection'] = df['projection'].apply(Names.get_projection_mappings_function())
		
		self.df = df

	def generate_plot_per_metric(self, df, name, metric, title, path):	
		# Set up the figure and axes
		plt.figure(figsize=(15,5))

			
		df['combination'] = df['model'] + '+' + df['projection']
		average_df = df[['combination',metric]].groupby('combination').mean().reset_index().sort_values(metric)
		
		sns.barplot(
			data=df, x='combination', y=metric, hue='projection', order=average_df['combination'][-1000:],
			errorbar=('sd', 1),
			err_kws={'linewidth':0.5},
			estimator='mean',
			palette={'RP':'lightblue', 'GAF':'pink', 'MTF':'lightgreen', 'PMix':'gold'},
		)
		
		#ax.set_ylim(-0.35, 1.35)
		#ticks = [i/10 for i in range(0,11,2)]
		#ax.set_xticks(ticks,[f'{i:.1f}' for i in ticks],fontsize=16,rotation=45)

		#ax.set_xlim((-0.3 ,1.3))

		plt.xticks([])

		# Set plot labels and title
		#plt.ylabel('', fontsize=20)
		#plt.xlabel(f'{title}', fontsize=20)
		#plt.title(name, fontsize=24)
		#plt.yticks(rotation=30, ha='right', fontsize=20)
		#plt.legend(loc='lower left', fontsize=20, ncols=1, framealpha=0.5, bbox_to_anchor=(1,0))


		# Save the figure to a file
		plt.savefig(os.path.join(path,f'scatterplot_{metric}_{name}.pdf'), bbox_inches='tight', dpi=1000)
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
			df = df[df['projection']!='Not projected']			
			self.generate_plot_per_metric(df, 'teste', column, metric, path)
