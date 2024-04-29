import pandas as pd

from scripts._Names_ import Pair_Group, Pair
from scripts._Names_ import *
from scripts._Model_Names_ import SpecificModel,Group,Aeon_Group
from scripts._Model_Names_  import *

if __name__=='__main__':
	from scripts import *
	
	df = pd.read_csv('data/results67+70.csv')
	df = Preprocessor(df).get()
	df = Selector.select(df)
	
	benchmark_df = pd.read_csv('data/benchmark67+70.csv')
	benchmark_df = Selector.select_benchmark(benchmark_df)
	benchmark_df = benchmark_df.fillna('None')
	
	pair_group = Pair_Group(
		'Selected Models', [
			Pair(SpecificModel('VisionTransformer_L_16',VisionTransformer()),RP()),
			Pair(SpecificModel('WideResNet101_2',WideResNet()),Mix()),
			Pair(SpecificModel('AlexNet',AlexNet()),Mix()),
			Pair(SpecificModel('MNASNet_1_0',MNASNet()),Mix()),
			Pair(SpecificModel('RegNetY_400MF',RegNet()),RP()),
			Pair(SpecificModel('VGG16',VGG()),RP()),
			Pair(SpecificModel('SqueezeNet_1_1',SqueezeNet()),Mix()),
		]
	)
	
	#ViolinplotGenerator(df).generate('img/resultados/violinplots')
	averages_table_generator = AveragesTableGenerator(df)
	averages_table_generator.generate('tex/tabelas/resultados/averages/models')
	averages_table_generator.generate_for_name_obj(Aeon_Group,'tex/tabelas/resultados/averages/models',delete_repeated=False)
	averages_table_generator.generate_per_pair_group('tex/tabelas/resultados/averages/models',pair_group)
	
	time_box_plot_generator = TimeBoxplotsGenerator(benchmark_df)
	time_box_plot_generator.generate('img/resultados/boxplots')
	time_box_plot_generator.generate_for_name_obj(Aeon_Group,'img/resultados/boxplots')
	time_box_plot_generator.generate_per_pair_group('img/resultados/boxplots',pair_group)
	
	memory_table_generator = MemoryTableGenerator(benchmark_df)
	memory_table_generator.generate('tex/tabelas/resultados/memory')
	memory_table_generator.generate_per_name_obj(Aeon_Group,'tex/tabelas/resultados/memory')
	memory_table_generator.generate_per_pair_group('tex/tabelas/resultados/memory', pair_group)
