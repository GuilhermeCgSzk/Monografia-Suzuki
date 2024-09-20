import pandas as pd

from abc import ABC,abstractmethod

from ._Model_Names_ import Model_Names,Group
from ._Filter_ import Filter

from ._Name_ import Name

__all__ = [
	'GAF','MTF','RP','Mix','NoProjection'
]

class Projection_Name(Name, ABC):		
	def mappings(self):
		return {
			self.name(): self.final_name()
		}
		
class GAF(Projection_Name):
	def name(self):
		return 'GramianAngularField'
	def final_name(self):
		return 'GAF'

class MTF(Projection_Name):
	def name(self):
		return 'MarkovTransitionField'
	def final_name(self):
		return 'MTF'

class RP(Projection_Name):
	def name(self):
		return 'RecurrencePlot'
	def final_name(self):
		return 'RP'
		
class Mix(Projection_Name):
	def name(self):
		return 'ProjectionMix_V2'
	def final_name(self):
		return 'PMix (ours)'
		
class NoProjection(Projection_Name):
	def name(self):
		return 'None'
	def final_name(self):
		return 'Not projected'
		
class Pair:
	def __init__(self, model, projection):
		self.model = model
		self.projection = projection

class Pair_Group(Filter, Group):
	def __init__(self, name, pairs, final_name=None):
		self.name = name
		self.pairs = pairs
		self.final_name = final_name
		
	def get_tuples(self):
		return [(pair.model.name(), pair.projection.name()) for pair in self.pairs]
		
	def get_models(self):
		return [pair.model for pair in self.pairs]
		
	def get_group(self):
		return Group(
			self.name,
			self.get_models(),
			self.final_name
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
		
	def get_2d_projections_names(use_final_name=True):
		return [c().final_name() if use_final_name else c().name() for c in [GAF, MTF, RP, Mix]]

	def get_projection_mappings_function():
		mappings = {}
		for projection_class in [GAF,RP,MTF,Mix,NoProjection]:
			mappings |= projection_class().mappings()
		return Mapping(mappings)
		
