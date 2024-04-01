import pandas as pd

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
	metric_mappings = {
		"model": "Neural model",
		"projection": "Projection",
		"accuracy_score": "Accuracy",
		"f1_score": "F1 Score", 
		"precision_score" : "Precision",
		"recall_score": "Recall"
	}
	def get_metric_mappings_function():
		return Mapping(Names.metric_mappings)
		
	model_mappings = {
		"AlexNet" : "AlexNet",
		"DenseNet121" : "DenseNet",
		"EfficientNet_B0" : "EfficientNet",
		"MNASNet_0_5" : "MNASNet",
		"MobileNet_V3_Large" : "MobileNet",
	        "MobileNet_V3_Small" : "MobileNet V3",
		"RegNetX_400MF" : "RegNetX",
		"RegNetY_400MF" : "RegNetY",
		"ResNet18" : "ResNet",
		"SqueezeNet_1_1" : "SqueezeNet",
		"SwinTransformer_T" : "SwinTransformer",
		"VGG11" : "VGG11",
		"VisionTransformer_B_32" : "ViT"
	}
	def get_model_mappings_function():
		return Mapping(Names.model_mappings)
	    
	projection_mappings = {
		"ProjectionMix_V2" : "Mix",
		"RecurrencePlot" : "RP",
		"GramianAngularField" : "GAF",
		"MarkovTransitionField" : "MTF"
	}
	def get_projection_mappings_function():
		return Mapping(Names.projection_mappings)
		
