import pandas as pd


if __name__=='__main__':
	from scripts import *
	
	df = pd.read_csv('data/results67+70.csv')
	df = Preprocessor(df).get()
	df = Selector.select(df)
	
	ViolinplotGenerator(df).generate('img/resultados/violinplots')
	AveragesTableGenerator(df).generate('tex/tabelas/resultados/averages')
	quit()
	
	benchmark_df = pd.read_csv('data/benchmark67+70.csv')
	benchmark_df = Selector.select_benchmark(benchmark_df)
	
	TimeBoxplotsGenerator(benchmark_df).generate('img/resultados/boxplots')
	MemoryTableGenerator(benchmark_df).generate('tex/tabelas/resultados/memory')
