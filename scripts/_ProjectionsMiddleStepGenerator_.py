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


def time_delay_phase_space(signal, path):			
	delay=10

	plt.figure(figsize=(5,5))
	plt.plot([signal[t] for t in range(len(signal)-delay)],[signal[t+delay] for t in range(len(signal)-delay)])
	plt.xlabel('$x_{t}$', fontsize=16)
	plt.ylabel('$x_{t+'+str(delay)+'}$', fontsize=16)
	plt.savefig(os.path.join(path,'delay_phase_space.pdf'))
	plt.close()
			

class ProjectionsMiddleStepGenerator(Generator):
	def generate(self, path):
		N=1000
		signal = np.array([math.sin(10*np.pi*i/N)/2 for i in range(N)])
		signal = numpy_rescale(signal,dst_min_val=0,dst_max_val=1, axis=0)
		
		# Signal Cartesian Plot
		
		plt.figure(figsize=(6,3))
		plt.plot(range(len(signal)),signal)
		with open(os.path.join(path,'signal.png'), 'wb') as f:
			plt.savefig(f)
		plt.close()
			
		#
		time_delay_phase_space(signal, path)
		
		
		
