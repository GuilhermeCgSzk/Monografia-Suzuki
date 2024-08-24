import pandas as pd
import os

from scripts._Names_ import Pair_Group, Pair
from scripts._Names_ import *
from scripts._Model_Names_ import SpecificModel,Group,Aeon_Group,SimpleModel
from scripts._Model_Names_  import *

if __name__=='__main__':
	from scripts import *
	
	df = pd.read_csv('data/results67+70.csv')
	df = Preprocessor(df).get()
	df = Selector.select(df)
	
	benchmark_df = pd.read_csv('data/benchmark67+70.csv')
	benchmark_df = Selector.select_benchmark(benchmark_df)
	benchmark_df = benchmark_df.fillna(NoProjection().name())
	
	pair_group = Pair_Group(
		'Best Models', [
			Pair(SpecificModel('SwinTransformerV2_S',SwinTransformerV2()),Mix()),
			Pair(SpecificModel('WideResNet101_2',WideResNet()),Mix()),
			Pair(SpecificModel('MNASNet_1_0',MNASNet()),Mix()),
			Pair(SpecificModel('SqueezeNet_1_1',SqueezeNet()),Mix()),
			Pair(SpecificModel('ShuffleNetV2_x0_5',ShuffleNetV2()),Mix()),
			Pair(SpecificModel('RegNetY_400MF',RegNet()),RP()),
			Pair(SimpleModel('TemporalDictionaryEnsemble'),NoProjection()),
		]
		
	)
	
	#ViolinplotGenerator(df).generate('img/resultados/violinplots')
	
	averages_table_dir = 'tex/tabelas/resultados/averages/models'
	os.makedirs(averages_table_dir, exist_ok=True)
	
	averages_table_generator = AveragesTableGenerator(df)
	averages_table_generator.generate(averages_table_dir)
	averages_table_generator.generate_for_name_obj(Aeon_Group,averages_table_dir, delete_repeated=False)
	averages_table_generator.generate_per_pair_group(averages_table_dir, pair_group)
	
	time_box_plot_dir = 'img/resultados/boxplots'
	os.makedirs(time_box_plot_dir, exist_ok=True)
	
	time_box_plot_generator = TimeBoxplotsGenerator(benchmark_df)
	time_box_plot_generator.generate(time_box_plot_dir, lim=100)
	time_box_plot_generator.generate_for_name_obj(Aeon_Group,time_box_plot_dir, lim=11)
	time_box_plot_generator.generate_per_pair_group(time_box_plot_dir, pair_group)
	
	
	memory_table_dir = 'tex/tabelas/resultados/memory'
	os.makedirs(memory_table_dir, exist_ok=True)
	
	memory_table_generator = MemoryTableGenerator(benchmark_df)
	memory_table_generator.generate(memory_table_dir)
	memory_table_generator.generate_per_name_obj(Aeon_Group, memory_table_dir)
	memory_table_generator.generate_per_pair_group(memory_table_dir, pair_group)
	
	methodology_dir = 'img/methodology'
	os.makedirs(methodology_dir, exist_ok=True)
	MethodologyGenerator().generate(methodology_dir)
