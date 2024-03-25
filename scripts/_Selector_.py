import pandas as pd

class Selector:
	def select(df):
		has_new_projection_mix = df[df['projection']=='ProjectionMix_V2']['model']
		df = df[df['model'].isin(has_new_projection_mix)|pd.isna(df['projection'])]
		df = df[df['projection']!='ProjectionMix']

		selection = df[['model','projection','timestamp']].groupby(['model','projection'], as_index=False, dropna=False).max()

		df = df.merge(selection,on=['model','projection','timestamp'])

		return df
		
	def select_benchmark(df):
		timestamps = df[['model','timestamp']].groupby('model').max().timestamp
		
		df = df[df['timestamp'].isin(timestamps)]
		
		return df
		

