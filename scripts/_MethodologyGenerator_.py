import matplotlib.pyplot as plt
import os
import math
import pyts.image as proj
import numpy
import itertools
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
		dpi=1000,
		cmap=color
	)
	plt.close()

class RecurrencePlotFigures:
	def __init__(self, delay, threshold):
		self.delay=delay
		self.threshold=threshold
		self.color='blue'

	def time_delay_phase_space(self, signal):
		x = [signal[t] for t in range(len(signal)-self.delay)]
		y = [signal[t+self.delay] for t in range(len(signal)-self.delay)]
		return x,y		
		
	def time_delay_phase_space_figure(self, signal, path, *, show_recurrences=False):	
		recurrence_color = 'red'
	
		x,y = self.time_delay_phase_space(signal)	

		plt.figure(figsize=(5,5))
		
		if show_recurrences:
			for (x1,y1),(x2,y2) in itertools.product(zip(x,y),zip(x,y)):
				dx = x1-x2
				dy = y1-y2
				distance = dx*dx+dy*dy
				if 0 < distance and distance <= self.threshold*self.threshold:
					plt.plot([x1,x2],[y1,y2],color=recurrence_color)
					plt.scatter([x1,x2],[y1,y2],color=recurrence_color,s=0.75)
				
		plt.plot(x,y,color=self.color)
		plt.scatter(x,y,color=self.color,s=0.75)
		plt.xlabel('$x_{t}$', fontsize=16)
		plt.ylabel('$x_{t+'+str(self.delay)+'}$', fontsize=16)
		
		if show_recurrences:
			name = 'delay_phase_space_recurrences.pdf'
		else:
			name = 'delay_phase_space.pdf'
		
		plt.savefig(os.path.join(path,name), dpi=300)
		plt.close()
		
	def projection_figure(self, signal, path, *, thresholded=True):
		if thresholded:
			threshold=self.threshold
		else:
			threshold=None
	
	
		reshaped_signal = signal.reshape(tuple(reversed(signal.shape)))
		matrix = proj.RecurrencePlot(
			dimension=2, 
			threshold=threshold, 
			time_delay=self.delay
		).transform([reshaped_signal])[0]
		
		if thresholded:
			name = "RecurrencePlot"
		else:
			name = "RecurrencePlotUnthresholded"
		
		
		save(matrix, name, 'Blues', path)
			
	

class MethodologyGenerator(Generator):
	def generate(self, path):
		N=200
		signal = np.array([math.sin(15*np.pi*i/N) + i/N for i in range(N)])
		signal = numpy_rescale(signal,dst_min_val=0,dst_max_val=1, axis=0)
		
		# Signal Cartesian Plot
		
		plt.figure(figsize=(6,3))
		plt.plot(range(len(signal)),signal,color='black')
		with open(os.path.join(path,'signal.png'), 'wb') as f:
			plt.savefig(f, dpi=300)
		plt.close()
			
		# Recurrence
		rp_figures = RecurrencePlotFigures(delay=10, threshold=0.05)		
		rp_figures.time_delay_phase_space_figure(signal, path)
		rp_figures.time_delay_phase_space_figure(signal, path, show_recurrences=True)
		rp_figures.projection_figure(signal, path)
		rp_figures.projection_figure(signal, path, thresholded=False)
