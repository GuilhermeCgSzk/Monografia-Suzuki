import pandas as pd

from ._Name_ import Name

__all__ = [
	'AlexNet',
	'ConvNeXt',
	'DenseNet',				
	'EfficientNet',					
	'EfficientNetV2',
	'MaxVit',
	'MNASNet',
	'MobileNetV2',
	'MobileNetV3',
	'RegNet',
	'ResNet',
	'ResNeXt',
	'ShuffleNetV2',
	'SqueezeNet',
	'SwinTransformer',
	'VGG',
	'VisionTransformer',
	'WideResNet'
]

class Model_Name(Name):
	pass
	
class SimpleModel(Model_Name):
	def __init__(self, _name, _final_name=None):
		self._name = _name
		
		if _final_name is None:
			self._final_name = self._name
		else:
			self._final_name = _final_name
		
	def name(self):
		return self._final_name
		
	def mappings(self):
		return {self._name: self._final_name}
	
class SpecificModel(Model_Name):
	def __init__(self, _name, _group):
		self._name = _name
		self._group=_group
		
	def mappings(self):
		return {self.name(): self._group.mappings()[self.name()]}

	def name(self):
		return self._name
	
class AlexNet(Model_Name):
	def name(self):
		return 'AlexNet'
		
class ConvNeXt(Model_Name):
	def name(self):
		return 'ConvNeXt'
	def mappings(self):
		return {
			'ConvNeXt_Tiny'	: 'ConvNeXt: Tiny',
			'ConvNeXt_Small': 'ConvNeXt: Small',
			'ConvNeXt_Base'	: 'ConvNeXt: Base',
			'ConvNeXt_Large': 'ConvNeXt: Large',
		}

class DenseNet(Model_Name):
	def name(self):
		return 'DenseNet'
	def mappings(self):
		return {
			'DenseNet121' : 'DenseNet: 121',
			'DenseNet161' : 'DenseNet: 161',
			'DenseNet169' : 'DenseNet: 169',
			'DenseNet201' : 'DenseNet: 201',
		}
		
class EfficientNet(Model_Name):
	def name(self):
		return 'EfficientNet'
	def mappings(self):
		return {
			'EfficientNet_B0' : 'EfficientNet: B0',
			'EfficientNet_B1' : 'EfficientNet: B1',
			'EfficientNet_B2' : 'EfficientNet: B2',
			'EfficientNet_B3' : 'EfficientNet: B3',
			'EfficientNet_B4' : 'EfficientNet: B4',
		}
		
class EfficientNetV2(Model_Name):
	def name(self):
		return 'EfficientNetV2'
	def mappings(self):
		return {
			'EfficientNetV2_S' : 'EfficientNet V2'
		}
		
class MaxVit(Model_Name):
	def name(self):
		return 'MaxVit'
	def mappings(self):
		return {
			'MaxVit_T' : 'MaxVit'
		}
		
class MNASNet(Model_Name):
	def name(self):		
		return 'MNASNet'
	def mappings(self):
		return {
			'MNASNet_0_5' : 'MNASNet: 0.5',
			'MNASNet_0_75' : 'MNASNet: 0.75',
			'MNASNet_1_0' : 'MNASNet: 1.0',
			'MNASNet_1_3' : 'MNASNet: 1.3',
		}
		
class MobileNetV2(Model_Name):
	def name(self):
		return 'MobileNet V2'
	def mappings(self):
		return {
			'MobileNet_V2': 'MobileNet V2'
		}
		
class MobileNetV3(Model_Name):
	def name(self):
		return 'MobileNet V3'
	def mappings(self):	
		return  {
			'MobileNet_V3_Small': 'MobileNet V3: Small',
			'MobileNet_V3_Large': 'MobileNet V3: Large'
		}
		
class RegNet(Model_Name):
	def name(self):	
		return 'RegNet'
	def mappings(self):
		return {
			'RegNetX_400MF'	: 'RegNet: X; 400 MF',
			'RegNetX_800MF'	: 'RegNet: X; 800 MF',
			'RegNetX_1_6FG'	: 'RegNet: X; 1.6 GF',
			'RegNetX_3_2FG'	: 'RegNet: X; 3.2 GF',
			'RegNetX_8FG'	: 'RegNet: X; 8 GF',
			'RegNetX_16FG'	: 'RegNet: X; 16 GF',
			'RegNetX_32FG'	: 'RegNet: X; 32 GF',
			
			'RegNetY_400MF'	: 'RegNet: Y; 400 MF',
			'RegNetY_800MF'	: 'RegNet: Y; 800 MF',
			'RegNetY_1_6FG'	: 'RegNet: Y; 1.6 GF',
			'RegNetY_3_2FG'	: 'RegNet: Y; 3.2 GF',
			'RegNetY_8FG'	: 'RegNet: Y; 8 GF',
			'RegNetY_16FG'	: 'RegNet: Y; 16 GF',
			'RegNetY_32FG'	: 'RegNet: Y; 32 GF',
		}
		
class ResNet(Model_Name):
	def name(self):
		return 'ResNet'
	def mappings(self):
		return {
			'ResNet18' : 'ResNet: 18',
			'ResNet34' : 'ResNet: 34',
			'ResNet50' : 'ResNet: 50',
			'ResNet101' : 'ResNet: 101',
			'ResNet152' : 'ResNet: 152',
		}
		
class ResNeXt(Model_Name):
	def name(self):
		return 'ResNeXt'
	def mappings(self):
		return {
			'ResNeXt50_32x4d' : 'ResNeXt: 50; 32x4d',
			'ResNeXt101_32x8d' : 'ResNeXt: 101; 32x8d',
			'ResNeXt101_64x4d' : 'ResNeXt: 101; 64x4d',
		}
		
class ShuffleNetV2(Model_Name):
	def name(self):
		return 'ShuffleNet V2'
	def mappings(self):
		return {
			'ShuffleNetV2_x0_5' : 'ShuffleNet V2: x0.5',
			'ShuffleNetV2_x1_0' : 'ShuffleNet V2: x1.0',
			'ShuffleNetV2_x1_5' : 'ShuffleNet V2: x1.5',
			'ShuffleNetV2_x2_0' : 'ShuffleNet V2: x2.0',
		}
		
class SqueezeNet(Model_Name):
	def name(self):
		return 'SqueezeNet'
	def mappings(self):
		return {
			'SqueezeNet_1_0' : 'SqueezeNet: 1.0',
			'SqueezeNet_1_1' : 'SqueezeNet: 1.1',	
		}
		
class SwinTransformer(Model_Name):
	def name(self):
		return 'SwinTransformer'
	def mappings(self):
		return {
			'SwinTransformer_T' : 'Swin Transformer: T',
			'SwinTransformer_S' : 'Swin Transformer: S',
			'SwinTransformer_B' : 'Swin Transformer: B',
		}
		
class SwinTransformerV2(Model_Name):
	def name(self):
		return 'SwinTransformer V2'
	def mappings(self):
		return {
			'SwinTransformerV2_T' : 'Swin Transformer V2: T',
			'SwinTransformerV2_S' : 'Swin Transformer V2: S',
			'SwinTransformerV2_B' : 'Swin Transformer V2: B',
		}
		
class VGG(Model_Name):
	def name(self):
		return 'VGG'
	def mappings(self):
		return {
			'VGG11'	  : 'VGG: 11',
			'VGG11_BN': 'VGG: 11 BN',
			'VGG13'	  : 'VGG: 13',
			'VGG13_BN': 'VGG: 13 BN',
			'VGG16'	  : 'VGG: 16',
			'VGG16_BN': 'VGG: 16 BN',
			'VGG19'	  : 'VGG: 19',
			'VGG19_BN': 'VGG: 19 BN',
		}
		
class VisionTransformer(Model_Name):
	def name(self):
		return 'VisionTransformer'
	def mappings(self):
		return {
			'VisionTransformer_L_16': 'Vision Transformer: L 16',
			'VisionTransformer_L_32': 'Vision Transformer: L 32',
			'VisionTransformer_H_14': 'Vision Transformer: H 14',
		}
		
class WideResNet(Model_Name):
	def name(self):
		return 'Wide ResNet'
	def mappings(self):
		return {
			'WideResNet50_2' : 'Wide ResNet: 50-2',
			'WideResNet101_2' : 'Wide ResNet: 101-2'
		}
	
class Group:
	def __init__(self, name, model_list):
		self._name = name
		self.model_list = model_list
	
	def name(self):
		return self._name
		
	def mappings(self):
		mapping_dict = {}
		for model in self.model_list:
			mapping_dict |= model.mappings()
		
		return mapping_dict


class Model_Names:	
	def models_list():
		return [
			AlexNet(),
			ConvNeXt(),
			DenseNet(),				
			EfficientNet(),					
			EfficientNetV2(),
			MaxVit(),
			MNASNet(),
			MobileNetV2(),
			MobileNetV3(),
			RegNet(),
			ResNet(),
			ResNeXt(),
			ShuffleNetV2(),
			SqueezeNet(),
			SwinTransformer(),
			VGG(),
			VisionTransformer(),
			WideResNet()
		]
	
	def group_list():
		return [
			Group(
				'Transformers', [
					VisionTransformer(),
					SwinTransformer(),
					MaxVit(),
				]
			),
			Group(
				'RegNet', [
					RegNet(),
				]
			),
			Group(
				'ResNet based', [
					ResNet(),
					ResNeXt(),
					WideResNet(),
				]
			),
			Group(
				'Mobile nets', [ 
					MNASNet(),
					MobileNetV2(),
					MobileNetV3(),
				]
			),
			
			Group(
				'Extreme models', [
					DenseNet(),
					VGG(),
				]
			),
			Group(
				'Purely convolutional', [  
					ConvNeXt(),
					AlexNet(),
				]
			),
			Group(
				'Diverse', [ 
					ShuffleNetV2(),  
					SqueezeNet(), 
					EfficientNet(),
					EfficientNetV2(),
				]
			),
		]
	         

        
Aeon_Group = Group(
	'Non CV', [
		SimpleModel('Arsenal'),
		SimpleModel('RocketClassifier','Rocket Classifier'),
		SimpleModel('CNNClassifier','CNN Classifier'),
		SimpleModel('FCNClassifier','FCN Classifier'),
		SimpleModel('MLPClassifier','MLP Classifier'),
		SimpleModel('InceptionTimeClassifier','Inception Time Classifier'),
		SimpleModel('IndividualInceptionClassifier','Individual Inception Classifier'),
		SimpleModel('TapNetClassifier','TapNet Classifier'),
		SimpleModel('EncoderClassifier','Encoder Classifier'),
		SimpleModel('LITETimeClassifier','LITE Time Classifier'),
		SimpleModel('BOSSEnsemble','BOSS Ensemble'),
		SimpleModel('ContractableBOSS','Contractable BOSS'),
		SimpleModel('IndividualBOSS','Individual BOSS'),
		SimpleModel('IndividualTDE','Individual TDE'),
		SimpleModel('MUSE'),
		SimpleModel('TemporalDictionaryEnsemble','Temporal Dictionary Ensemble'),
		SimpleModel('WEASEL'),
		SimpleModel('WEASEL_V2','WEASEL V2'),
		SimpleModel('REDCOMETS'),
		SimpleModel('ElasticEnsemble','Elastic Ensemble'),
		SimpleModel('KNeighborsTimeSeriesClassifier','K-Neighbors Time Series Classifier'),
		SimpleModel('ShapeDTW','Shape DTW'),
		SimpleModel('Catch22Classifier','Catch 22 Classifier'),
		SimpleModel('FreshPRINCEClassifier','Fresh PRINCE Classifier'),
		SimpleModel('MatrixProfileClassifier','Matrix Profile Classifier'),
		SimpleModel('SignatureClassifier','Signature Classifier'),
		SimpleModel('SummaryClassifier','Summary Classifier'),
		SimpleModel('TSFreshClassifier','TS Fresh Classifier'),
		SimpleModel('HIVECOTEV1','HIVECOTE V1'),
		SimpleModel('HIVECOTEV2','HIVECOTE V2'),
		SimpleModel('CanonicalIntervalForestClassifier','Canonical Interval Forest Classifier'),
		SimpleModel('DrCIFClassifier','DrCIF Classifier'),
		SimpleModel('RandomIntervalSpectralEnsembleClassifier','Random Interval Spectral Ensemble Classifier'),
		SimpleModel('SupervisedTimeSeriesForest','Supervised Time Series Forest'),
		SimpleModel('TimeSeriesForestClassifier','Time Series Forest Classifier'),
		SimpleModel('RandomIntervalClassifier','Random Interval Classifier'),
		SimpleModel('ShapeletTransformClassifier','Shapelet Transform Classifier'),
		SimpleModel('MrSQMClassifier','MrSQM Classifier'),
		SimpleModel('RDSTClassifier','RDST Classifier'),
		SimpleModel('ContinuousIntervalTree','Continuous Interval Tree'),
		SimpleModel('RotationForestClassifier','Rotation Forest Classifier'),
		SimpleModel('ProbabilityThresholdEarlyClassifier','Probability Threshold Early Classifier'),
		SimpleModel('TEASER'),
		SimpleModel('IndividualOrdinalTDE','Individual Ordinal TDE'),
		SimpleModel('OrdinalTDE','Ordinal TDE'),
	]
)


         

         


        
		
