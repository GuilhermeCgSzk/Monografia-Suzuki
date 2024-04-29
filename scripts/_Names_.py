import pandas as pd

from ._Model_Names_ import Model_Names,Group
from ._Filter_ import Filter

from ._Name_ import Name

__all__ = [
	'GAF','MTF','RP','Mix'
]

class Projection_Name(Name):
	pass
		
class GAF(Projection_Name):
	def name(self):
		return 'GramianAngularField'
	def mappings(self):
		return {
			self.name(): "GAF"
		}

class MTF(Projection_Name):
	def name(self):
		return 'MarkovTransitionField'
	def mappings(self):
		return {
			self.name(): "MTF"
		}

class RP(Projection_Name):
	def name(self):
		return 'RecurrencePlot'
	def mappings(self):
		return {
			self.name(): "RP"
		}
		
		
class Mix(Projection_Name):
	def name(self):
		return 'ProjectionMix_V2'
	def mappings(self):
		return {
			self.name(): "Mix"
		}
		
class Pair:
	def __init__(self, model, projection):
		self.model = model
		self.projection = projection

class Pair_Group(Filter, Group):
	def __init__(self, name, pairs):
		self.name = name
		self.pairs = pairs
		
	def get_tuples(self):
		return [(pair.model.name(), pair.projection.name()) for pair in self.pairs]
		
	def get_models(self):
		return [pair.model for pair in self.pairs]
		
	def get_group(self):
		return Group(
			self.name,
			self.get_models(),
		)
		
	def filter(self, df):
		new_df = df[df[['model','projection']].apply(tuple,axis=1).isin(self.get_tuples())]
		return new_df.copy()
	 	

class Mapping:
	def __init__(self, mappings):
		self.mappings = mappings
		
	def __call__(self, name):
		if name in self.mappings:
			return self.mappings[name]
		else:
			if not pd.isna(name) and '_' in name:
				name = name.replace('_',' ')
		
			return name
class Names:
	def get_metric_mappings_function():
		return Mapping({
			"model": "Model",
			"projection": "Projection",
			"accuracy_score": "Accuracy",
			"f1_score": "F1 Score", 
			"precision_score" : "Precision",
			"recall_score": "Recall",
			"cohen_kappa_score": "Cohen Kappa"
		})

	def get_group_list():
		return Model_Names.group_list()

	def get_model_list():
		return Model_Names.models_list() 

	def get_projection_mappings_function():
		return Mapping({
			"ProjectionMix_V2" : "Mix",
			"RecurrencePlot" : "RP",
			"GramianAngularField" : "GAF",
			"MarkovTransitionField" : "MTF"
		})
		
