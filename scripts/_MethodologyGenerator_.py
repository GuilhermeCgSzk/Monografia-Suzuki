import matplotlib.pyplot as plt
import plotly.graph_objects as go
import random
from matplotlib.patches import Rectangle
import os
import sklearn
import pyts
import math
import pyts.image as proj
import numpy
import itertools
import numpy as np
import networkx as nx
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

class GramianAngularFieldFigures:
	def polar_coordinates(self, signal, path):
		signal = numpy_rescale(signal,dst_min_val=-1,dst_max_val=1, axis=0)
		
		fig, ax = plt.subplots(figsize=(10,7), subplot_kw={'projection':'polar'})
		fig.tight_layout()
		
		ax.plot([math.acos(x) for x in signal],list(range(len(signal))), color='red')
		
		ax.set_theta_zero_location("W")
		ax.set_theta_direction('clockwise') 
		ax.set_thetamin(0)
		ax.set_thetamax(180)
		fig.savefig(os.path.join(path,'polar'), dpi=300)
		plt.close()

	def projection_figure(self, signal, path, *, method, name=None):
		reshaped_signal = signal.reshape(tuple(reversed(signal.shape)))
		matrix = proj.GramianAngularField(method=method).transform([reshaped_signal])[0]
		
		
		method_name = {
			'summation' : 'Summation',
			'difference': 'Difference'
		}
			
		final_name = f'GramianAngular{method_name[method]}Field'
		
		if name is not None:
			final_name += f'({name})'
		
		save(matrix, final_name, 'Reds', path)

class MarkovTransitionFieldFigures:
	def __init__(self, n_bins):
		self.n_bins = n_bins
		
		reds = [x/(self.n_bins-1) for x in range(self.n_bins)]
		greens_reverse = [1-x/(self.n_bins-1) for x in range(self.n_bins)]
		self.colors = [(r,g,0,0.5) for r,g in zip(reds,greens_reverse)]
		
	def quantile_bins_figure(self, signal, path):
		sklearn_signal = sklearn.preprocessing.RobustScaler().fit_transform([[sample] for sample in signal])
		sklearn_signal = sklearn_signal - (sklearn_signal.max()+sklearn_signal.min())/2
		signal =[sample[0] for sample in sklearn_signal]	
		
		fig, ax = plt.subplots(figsize=(10, 5))
		
		
		[bins] = pyts.preprocessing.KBinsDiscretizer(n_bins=self.n_bins).transform([signal])
		
		#qbin = -1
		#for x in range(len(signal)):
		#	if bins[x] != qbin:
		#		qbin = bins[x]
		#		ax.plot(np.arange(x,len(signal)),signal[x:], color=colors[bins[x]], linewidth=3)
				
		ax.plot(list(range(len(signal))),signal,color='black',linewidth=2,zorder=3)
		ax.plot([0,len(signal)],[min(signal),min(signal)],color='grey',zorder=2)
		ax.plot([0,len(signal)],[max(signal),max(signal)],color='grey',zorder=2)
				
		for bin_i in range(0,self.n_bins):
			minimum,maximum = None,None
			for x,b in enumerate(bins):
				if b == bin_i:
					if minimum is None or signal[x]<minimum:
						minimum = signal[x]
					if maximum is None or signal[x]>maximum:
						maximum = signal[x]
			if bin_i>=1:
				ax.plot([0,len(signal)],[minimum,minimum],color='grey',zorder=2)
			ax.add_patch(Rectangle((0, minimum), len(signal), maximum-minimum, fill=True, facecolor=self.colors[bin_i],edgecolor='none'))					
				
				
		ax.set_xlim(0,len(signal))
		ax.set_xticks([])
		ax.set_yticks([])
			
		fig.savefig(os.path.join(path,'QuantileBins'), dpi=300)
		plt.close()
		
	def markov_chain(self, signal, path):
		def qbin_str(qbin_index):
			return '$Q_{'+str(qbin_index+1)+'}$'
	
		[bins] = pyts.preprocessing.KBinsDiscretizer(n_bins=self.n_bins).transform([signal])
		
		G = nx.MultiDiGraph()
		
		G.add_nodes_from([(qbin_str(qbin), {"color": self.colors[qbin]}) for qbin in range(self.n_bins)])
		
		for qbin in range(self.n_bins):
			transitions = [0 for _ in range(self.n_bins)]
			bin_size = 0
			for i in range(len(signal)):
				if bins[i]==qbin:
					bin_size += 1
					if i+1 < len(signal):
						transitions[bins[i+1]] += 1
		
			for qbin_2 in range(self.n_bins):
				if transitions[qbin_2] > 0:
					G.add_edge(qbin_str(qbin), qbin_str(qbin_2), weight=round(transitions[qbin_2]/bin_size,ndigits=2))
				
		pos = nx.circular_layout(G)
		
		
		color_map=[]
		# nodes
		for i,node in enumerate(G.nodes()):
			color = (self.colors[i][0], self.colors[i][1], self.colors[i][2], 1)
			color_map.append(color)		
		
		nx.draw_networkx_nodes(G, pos, node_color=color_map,node_size=700)
			
		# node labels
		nx.draw_networkx_labels(G, pos, font_size=14, font_family="sans-serif")

		connectionstyle = [f"arc3,rad={r}" for r in itertools.accumulate([0.15] * 4)]

		# edges
		nx.draw_networkx_edges(
		    G, pos, edgelist=[(i,j) for (i,j,w) in G.edges(data=True)], 
		    width=3, alpha=0.6, edge_color='blue', 
		    connectionstyle=connectionstyle,
		    arrows=True, arrowstyle="->", arrowsize=15
		)
		
		# edge weight labels
		#edge_labels = nx.get_edge_attributes(G, "weight")
		labels = {
			tuple(edge): f"{attrs['weight']}" for *edge, attrs in G.edges(keys=True, data=True)
		}
		
		nx.draw_networkx_edge_labels(
			G,
			pos,
			labels,
			connectionstyle=connectionstyle,
			label_pos=0.5,
			verticalalignment='bottom',
			font_color="blue",
			bbox={"alpha": 0},
		)
					
		plt.tight_layout()
		plt.savefig(os.path.join(path,'MarkovChain.pdf'), dpi=300)
		plt.close()
			
	def reconstruction(self, signal, path):
		[bins] = pyts.preprocessing.KBinsDiscretizer(n_bins=self.n_bins).transform([signal])
		
		transition_matrix = np.ndarray((self.n_bins,self.n_bins))
		
		bins_samples = []
		
		for qbin in range(self.n_bins):
			bins_samples.append([])
		
			transitions = [0 for _ in range(self.n_bins)]
			for i,x in enumerate(signal):
				if bins[i]==qbin:
					bins_samples[qbin].append(x)
					if i+1 < len(signal):
						transitions[bins[i+1]] += 1
		
			
		
			for qbin_2 in range(self.n_bins):
				transition_matrix[qbin][qbin_2] = transitions[qbin_2]/sum(transitions)
			
		reconstructed_signal = []
				
		qbin = random.randrange(self.n_bins)
		for _ in range(len(signal)):
			sample = np.random.choice(bins_samples[qbin])
			reconstructed_signal.append(sample)
			qbin = np.random.choice(list(range(self.n_bins)),p=transition_matrix[qbin])
			
		plt.figure(figsize=(6,3))
		plt.plot(range(len(reconstructed_signal)),reconstructed_signal,color='black')
		plt.savefig(os.path.join(path,'Reconstruction.pdf'), dpi=300)
		plt.close()

	def projection_figure(self, signal, path, *, name=None):
		reshaped_signal = signal.reshape(tuple(reversed(signal.shape)))
		matrix = proj.MarkovTransitionField(n_bins=self.n_bins).transform([reshaped_signal])[0]
		
		
		final_name = 'MarkovTransitionField'
		
		if name is not None:
			final_name += f'({name})'
		
		save(matrix, final_name, 'Greens', path)
	

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
					plt.plot([x1,x2],[y1,y2],color=recurrence_color,zorder=1, linewidth=0.5)
				
		plt.scatter(x,y,color=self.color,s=0.75,zorder=2)
		plt.xlabel('$x_{t}$', fontsize=16)
		plt.ylabel('$x_{t+'+str(self.delay)+'}$', fontsize=16)
		
		if show_recurrences:
			name = 'delay_phase_space_recurrences.pdf'
		else:
			name = 'delay_phase_space.pdf'
		
		plt.savefig(os.path.join(path,name), dpi=300)
		plt.close()
		
	def projection_figure(self, signal, path, *, thresholded=True, name=None):
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
			final_name = "RecurrencePlot"
		else:
			final_name = "RecurrencePlotUnthresholded"
		
		if name is not None:
			final_name += f'({name})'
		
		save(matrix, final_name, 'Blues', path)
			
	

class MethodologyGenerator(Generator):
	def generate(self, path):
		plt.tight_layout()
	
		N=500
		signal = np.array([math.sin(15*np.pi*i/N) + i/N for i in range(N)])
		signal = numpy_rescale(signal,dst_min_val=0,dst_max_val=1, axis=0)
		
		# Signal Cartesian Plot
		
		plt.figure(figsize=(6,3))
		plt.plot(range(len(signal)),signal,color='black')
		with open(os.path.join(path,'signal.png'), 'wb') as f:
			plt.savefig(f, dpi=300)
		plt.close()
			
		rp_figures = RecurrencePlotFigures(delay=10, threshold=0.05)		
		rp_figures.time_delay_phase_space_figure(signal, path)
		rp_figures.time_delay_phase_space_figure(signal, path, show_recurrences=True)
		rp_figures.projection_figure(signal, path)
		rp_figures.projection_figure(signal, path, thresholded=False)
		
		gaf_figures = GramianAngularFieldFigures()
		gaf_figures.polar_coordinates(signal, path)
		gaf_figures.projection_figure(signal, path, method='summation')
		gaf_figures.projection_figure(signal, path, method='difference')
		
		mtf_figures = MarkovTransitionFieldFigures(10)
		mtf_figures.reconstruction(signal, path)
		mtf_figures.quantile_bins_figure(signal, path)
		mtf_figures.markov_chain(signal, path)
		mtf_figures.projection_figure(signal, path)
		
		# simplest_signal -------------------------------------------
		
		examples_path = os.path.join(path,'examples')
		
		base_signal = np.array([math.sin((i/N)*10*np.pi) for i in range(N)])
		gaussian_noisy_signal = np.array([x1+x2 for x1,x2 in zip(base_signal, np.random.normal(0,0.5,N))])
		salt_and_pepper_signal = np.array([np.random.choice([min(base_signal),max(base_signal),x],p=[0.1,0.1,0.8]) for x in base_signal])
		low_frequency_noise_signal = np.array([x+0.4*math.sin((i/N)*100*np.pi) for i,x in enumerate(base_signal)])
		baseline_wander_signal = np.array([x1+x2 for x1,x2 in zip(base_signal,[2*math.sin((i/N)*2*np.pi) for i in range(N)])])	
		local_noise_signal = base_signal.copy()
		size = N//5
		pos = random.randrange(N-size)
		for k,i in enumerate(range(pos,min(pos+size,N))):
			local_noise_signal[i] += math.sin(np.pi*10*(k/size)) #np.random.normal(0,1)
		
		for signal,name in [
			(base_signal,'base'),
			(gaussian_noisy_signal,'gaussian'),
			(salt_and_pepper_signal,'salt_and_pepper'),
			(baseline_wander_signal,'baseline_wander'),
			(low_frequency_noise_signal,'low_frequency_noise'),
			(local_noise_signal,'local_noise')
		]:			
			plt.figure(figsize=(6,3))
			plt.plot(range(len(signal)),signal,color='black')
			with open(os.path.join(examples_path,f'signal({name}).png'), 'wb') as f:
				plt.savefig(f, dpi=300)
			plt.close()
			
			rp_figures.projection_figure(signal, examples_path, thresholded=False, name=name)
			gaf_figures.projection_figure(signal, examples_path, method='summation', name=name)
			gaf_figures.projection_figure(signal, examples_path, method='difference', name=name)
			mtf_figures.projection_figure(signal, examples_path, name=name)
		
		
		
