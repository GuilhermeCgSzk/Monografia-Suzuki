class DataframeSplitter:
	def __init__(self, n):
		self.n = n	
	
	def split(self, df):
		length = len(df)//self.n
		base = (len(df)+(self.n-1))//self.n
				
		yield df[:base].copy()
				
		for i in range(1,self.n):
			l = base+(i-1)*length
			r = l+length
			yield df[l:r].copy()
