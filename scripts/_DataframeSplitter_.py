import pandas as pd

class DataframeSplitter:
	def __init__(self, n):
		self.n = n	
	
	def split(self, df):	
		length = (len(df)+self.n-1)//self.n
				
		for i in range(0,self.n):
			l = length*i
			r = l+length
			
			new_df = df[l:r].copy()
			new_df = new_df.reset_index(drop=True)
			new_df = new_df.reindex(range(length), fill_value=None)
			
			yield new_df
