import pandas as pd

from ._Names_ import NoProjection
from ._Model_Names_ import Aeon_Group,Model_Names

class Selector:
	def select(df):
		has_new_projection_mix = df[df['projection']=='ProjectionMix_V2']['model']
		df = df[df['model'].isin(has_new_projection_mix) | (df['projection']==NoProjection().name())]
		df = df[df['projection']!='ProjectionMix']

		selection = df[['model','projection','timestamp']].groupby(['model','projection'], as_index=False, dropna=False).max()

		df = df.merge(selection,on=['model','projection','timestamp'])
		
		mappings = {}	
		for name_obj in Model_Names.models_list() + Aeon_Group.model_list:	
			mappings = mappings | name_obj.mappings()
		df = df[df['model'].isin(mappings)].copy()	

		return df
		
	def select_benchmark(df):		
		timestamps = df[['model','timestamp']].groupby('model').max().timestamp
		
		df = df[df['timestamp'].isin(timestamps)]
		
		return df
		

