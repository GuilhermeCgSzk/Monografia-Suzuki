import pandas as pd

from scripts._Model_Names_ import Group
from scripts._Model_Names_ import *

if __name__=='__main__':
	from scripts import *
	
	df = pd.read_csv('data/results67+70.csv')
	df = Preprocessor(df).get()
	df = Selector.select(df)
	
	benchmark_df = pd.read_csv('data/benchmark67+70.csv')
	benchmark_df = Selector.select_benchmark(benchmark_df)
	
	
	#ViolinplotGenerator(df).generate('img/resultados/violinplots')
	AveragesTableGenerator(df).generate('tex/tabelas/resultados/averages/models')
	
	TimeBoxplotsGenerator(benchmark_df).generate('img/resultados/boxplots')
	
	memory_table_generator = MemoryTableGenerator(benchmark_df)
	memory_table_generator.generate('tex/tabelas/resultados/memory')
