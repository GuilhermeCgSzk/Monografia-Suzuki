import pandas as pd

from ._Model_Names_ import Model_Names

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
		
