import matplotlib.pyplot as plt
import os
import math
import pyts.image as proj
import numpy
import numpy as np
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import gaussian

from ._Generator_ import Generator

def numpy_rescale(numpy_arr, *, src_min_val=None, src_max_val=None, dst_min_val=0, dst_max_val=1, axis):
	if src_min_val is None:
		src_min_val = numpy.min(numpy_arr, axis=axis)
	if src_max_val is None:
		src_max_val = numpy.max(numpy_arr, axis=axis)

	rescaled_arr = (numpy_arr - src_min_val)/(src_max_val - src_min_val)
	rescaled_arr = rescaled_arr*(dst_max_val - dst_min_val) + dst_min_val
	rescaled_arr = numpy.maximum(rescaled_arr, numpy.full(rescaled_arr.shape,dst_min_val))
	rescaled_arr = numpy.minimum(rescaled_arr, numpy.full(rescaled_arr.shape,dst_max_val))
	return rescaled_arr

def save(projected_signal, projection_name, color, path):
	full_path = os.path.join(path,f'{projection_name}.png')
	os.makedirs(os.path.split(full_path)[0], exist_ok=True)
	plt.imsave(
		full_path,
		projected_signal,
		dpi=300,
		cmap=color
	)


class PoincatePlotLogarithmGrid:
	def __init__(self, dimension: int):	
		self.dimension = dimension
		
	def project(self, signal):			
		coordinates = [(signal[i],signal[i+1]) for i in range(len(signal)-1)]
		
		projection = np.zeros((self.dimension,self.dimension),)
		
		for x,y in coordinates:
			i = min(int(x*self.dimension),self.dimension-1)
			j = min(int(y*self.dimension),self.dimension-1)
			projection[i][j] += 1
			
		for i in range(len(projection)):
			for j in range(len(projection[i])):
				if not math.isclose(projection[i][j],0):
					projection[i][j] = math.log(projection[i][j])
				
		return projection
			

class ProjectionsGenerator(Generator):
	def generate(self, path):
		N=100
		signal = np.array([math.sin(10*np.pi*i/N)/2 for i in range(N)])
		signal = numpy_rescale(signal,dst_min_val=0,dst_max_val=1, axis=0)
		
		# Matrix Embedding
		
		projections = [
			(proj.GramianAngularField(sample_range=(0,1)),'GramianAngularFieldSummation','pink'),
			(proj.GramianAngularField(sample_range=(0,1), method='difference'),'GramianAngularFieldDifference','autumn'), 
			(proj.MarkovTransitionField(n_bins=4),'MarkovTransitionField','Greens'), 
			(proj.RecurrencePlot(dimension=1),'RecurrencePlot','Blues')
		]
		for projection,projection_name,color in projections:
			reshaped_signal = signal.reshape(tuple(reversed(signal.shape)))
			matrix = projection.transform([reshaped_signal])[0]
			save(matrix, projection_name, color, path)
		
		# Short Term Fourier Transform Spectogram
		
		windows = gaussian(50, std=12, sym=True)
		stfft = ShortTimeFFT(windows, 5, 10)
		stfft_projection = stfft.spectrogram(signal)
		
		save(stfft_projection, 'ShortTimeFFT', 'summer', path)
		
		# Poincaré plot density map
		
		pplg_projection = PoincatePlotLogarithmGrid(20).project(signal)
		save(pplg_projection,'PoincatePlotLogarithmGrid','jet',path)
		
		
		# Multiscale Markov Transition Field
		velocity_signal = np.gradient(signal)
		acceleration_signal = np.gradient(velocity_signal)
		
		multiscale_signal = np.concatenate((signal, velocity_signal, acceleration_signal), axis=0)
		reshaped_multiscale_signal = multiscale_signal.reshape(tuple(reversed(multiscale_signal.shape)))
		matrix = proj.MarkovTransitionField(n_bins=4).transform([reshaped_multiscale_signal])[0]
		save(matrix, 'MultiscaleMarkovTransitionField', 'magma', path)
