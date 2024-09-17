import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os

from ._Names_ import Names
from ._Generator_ import Generator
from ._DataframeSplitter_ import DataframeSplitter

class BUTPPGInfoGenerator(Generator):
	def __init__(self, path):
		df = pd.read_csv(os.path.join(path,'subject-info.csv'))
		df['subject_id'] = (df['ID']//1000).apply(str)
		
		
		self.subject_info_df = df[['subject_id','Gender','Age','Weight']].groupby('subject_id').first().reset_index()
		
		motion_info_df = df[['ID','Motion','subject_id','Gender']]
		
		quality_df = pd.read_csv(os.path.join(path,'quality-hr-ann.csv'))
		
		self.signal_info_df = quality_df.merge(motion_info_df, on='ID').reset_index()

	def generate_gender_plot(self, path):
		
		plt.figure(figsize=(5,5))
		
		ax = sns.countplot(
			data=self.subject_info_df, x='Gender', y=None, hue="Gender",
		)
		
		plt.ylabel('')
		plt.yticks(fontsize=20)
		plt.xlabel('Gender',fontsize=20)
		plt.xticks(fontsize=20)
		
		plt.savefig(os.path.join(path,f'gender_plot.png'), bbox_inches='tight', dpi=1000)
		plt.close()
		

	def generate_subject_plot(self, attribute, path):
		
		plt.figure(figsize=(10,5))
		
		ax = sns.catplot(
			data=self.subject_info_df, x=attribute, y=None, hue="Gender",
			kind="violin", bw_adjust=.5, split=True, inner='stick', legend_out=False
		)
		
		plt.legend(fontsize=20, title='Gender', title_fontsize=20, bbox_to_anchor=(1.1, 1.05))
		
		plt.xticks(fontsize=20)
		plt.xlabel(attribute,fontsize=20)
		plt.savefig(os.path.join(path,f'subject_info_{attribute}.png'), bbox_inches='tight', dpi=1000)
		plt.close()
		
	def generate_signal_plot(self, attribute, path):
		
		plt.figure(figsize=(10,5))
		
		plt.pie(
			self.signal_info_df[attribute].value_counts(), explode=(0.1,0), labels=('Good','Bad'), 
			textprops={'fontsize': 20}, colors=((0.5,0.7,0.5), (0.7,0.5,0.5))
		)
			
		plt.savefig(os.path.join(path,f'motion_plot.png'), bbox_inches='tight', dpi=1000)
		plt.close()
		
	def generate_hr_plot(self, path):
		
		plt.figure(figsize=(10,5))
		
		
		sns.boxplot(
			data=self.signal_info_df, x='HR', y='subject_id', hue='Gender'
		)
			
		plt.legend(fontsize=20, title_fontsize=20)
		plt.xticks(fontsize=20)
		plt.xlabel('HR', fontsize=20)
		plt.yticks([])
		plt.ylabel('')
			
		plt.savefig(os.path.join(path,f'hr_plot.png'), bbox_inches='tight', dpi=1000)
		plt.close()
		
	def generate(self, path):
		self.generate_gender_plot(path)
		self.generate_subject_plot('Age',   path)
		self.generate_subject_plot('Weight',path)
		self.generate_signal_plot('Quality',path)
		self.generate_hr_plot(path)
