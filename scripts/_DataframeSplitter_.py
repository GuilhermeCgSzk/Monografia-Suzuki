import pandas as pd

class DataframeSplitter:
	def __init__(self, n):
		self.n = n	
	
	def split(self, df):
		length = (len(df)+self.n-1)//self.n
				
		for i in range(0,self.n):
			l = length*i
			r = l+length
			
			new_df = df[l:r].copy().reindex(range(l,r),fill_value=-1)
			
			for i in range(length):
				for j in range(len(df.columns)):
					if new_df.iloc[i,j]==-1:
						new_df.iloc[i,j] = None	
			
			yield new_df
