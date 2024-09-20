import sys
import simplevc
simplevc.register(sys.modules[__name__])

import itertools

import matplotlib.pyplot as plt
import numpy as np

@vc
def _plot_fast_bar_20240601(x, height, width=0.8, bottom=0, align='center', fill_kw={}, ax=None):
	'''
	Plot bar in a faster way using a single artist from fill_between (instead of many Rectangles) by sacrificing flexibility to customize individual bars. Good to use if you need to plot many (1000+) bars. 
	Note that unlike matplotlib.pyplot.bar, there are fewer available options.
	
	'''
	if ax is None:
		ax = plt.gca()
	
	x = np.array(x)
	height = np.array(height)
	if np.ndim(bottom) == 0:
		bottom = np.repeat(bottom, len(height))
	else:
		bottom = np.array(bottom)
	
	artists = []
	
	if isinstance(fill_kw, list):
		all_fill_kw_indice = set([i for indice, fkw in fill_kw for i in indice])
		no_kw_indice = [i for i in range(len(x)) if i not in all_fill_kw_indice]
		fill_kws = fill_kw + [[no_kw_indice, {}]]
	else:
		
		fill_kws = [[np.arange(len(x)), fill_kw]]
	
	if align == 'center':
			modifier = width / 2
	else:
		modifier = 0	
	for indice, fkw in fill_kws:
		if len(indice) == 0:
			continue
		sx = x[indice]
		sb = bottom[indice] 
		sy = height[indice]
		ny2 = np.repeat(sb, 4)
		ny = list(itertools.chain.from_iterable([(b, h+b, h+b, b) for h, b in zip(sy, sb)]))		
		nx = list(itertools.chain.from_iterable([(i - modifier, i - modifier, i + width - modifier, i + width - modifier) for i in sx]))
		combined_kwargs = {"linewidth":0, **fkw}	
		artist = ax.fill_between(nx, ny, ny2, **combined_kwargs)
		if len(bottom) > 0:
			artist.sticky_edges.y.append(min(bottom))
		artists.append(artist)
	return artists


@vc
def _plot_ranked_values_20240601(data, plot_kw={}, plot_kw_dict={}, ax=None):
	'''
	data could be a list of values, or a dict of list of values.
	'''
	if ax is None:
		ax = plt.gca()
	start_idx = 0
	if isinstance(data, dict):
		for g, y in data.items():
			group_plot_kw = plot_kw_dict[g] if g in plot_kw_dict else {}
			group_plot_kw = {**plot_kw, **group_plot_kw}
			artist = ax.scatter(np.arange(len(y)) + start_idx, sorted(y), **group_plot_kw, label=g)
			start_idx += len(y)
	else:
		y = data
		artist = ax.scatter(np.arange(len(y)) + start_idx, sorted(y), **plot_kw)
	return artist
