import sys
import simplevc
simplevc.register(sys.modules[__name__])

import itertools

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnchoredText
from commonhelper import safe_inverse_zip
from matplotlib.ticker import AutoLocator
import biodataplot.utils as bpu

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

@vc
def _add_scatter_odr_line_20240701(x, y, method="unilinear", plot_kw={}, ax=None):
	from scipy import odr
	if ax is None:
		ax = plt.gca()
	data = odr.RealData(x, y)
	if method == "unilinear":
		model = odr.unilinear
	else:
		raise Exception("Unsupported")
	m, b = np.polyfit(x, y, 1)
	ax.axline([0, b], slope=m, **plot_kw)
	myodr = odr.ODR(data, model, beta0=[m, b])
	myodr.set_job(fit_type=0)
	myoutput = myodr.run()
	m, b = myoutput.beta
	return ax.axline([0, b], slope=m, **plot_kw)
@vc
def _add_scatter_correlation_20240701(
		x, y, method="pearson",
		show_r=True, show_n=False, show_p=False, use_rsquare=False, 
		text_kw={}, ax=None
	):
	import scipy.stats
	import re
	def _format_pvalue(p):
		'''
		Formats the pvalue to $p=0.0333$ or $p=0.0333$
		
		:Example:
		
		.. code-block:: python
		
			p = 0.1234
			for _ in range(5):
				print(_format_pvalue(p))
				p /= 10
			
			# $p=0.123$
			# $p=0.0123$
			# $p=0.00123$
			# $p=1.23\\times 10^{-4}$
			# $p=1.23\\times 10^{-5}$
	
		'''
		if p == 0:
			pstr="0.000"
		elif 0 < p < 0.001:
			match = re.match("^(-?[0-9]\\.[0-9][0-9])e([+-]?[0-9]+)$", f"{p:.2e}")
			pstr = match.group(1) + "\\times 10^{" + str(int(match.group(2))) + "}"	
		elif 0.001 <= p < 0.01:
			pstr = f"{p:.5f}"
		elif 0.01 <= p < 0.1:
			pstr = f"{p:.4f}"
		elif 0.1 <= p < 0.9995:
			pstr = f"{p:.3f}"
		else:
			pstr = "1.00"
		return f"$p={pstr}$"

	if ax is None:
		ax = plt.gca()
	if method == "pearson":
		r, pvalue = scipy.stats.pearsonr(x, y)
	else:
		raise Exception("Unsupported correlation method")
	lines = []
	if show_r:
		if use_rsquare:
			lines.append(f"$r^2={r**2:.2f}$")
		else:
			lines.append(f"$r={r:.2f}$")
	if show_p:
		lines.append(f"{_format_pvalue(pvalue)}")
	if show_n:
		lines.append(f"$n={len(x)}$")
	atext = AnchoredText("\n".join(lines), **{"loc":"upper left","frameon":False, "prop":{"fontsize":14}, **text_kw})
	ax.add_artist(atext)
	return atext

@vc
def _plot_density_scatter_20240901(x, y, bins=20, ax=None, **kwargs)   :
	"""
	Scatter plot colored by 2d histogram
	"""
	from matplotlib import cm
	from matplotlib.colors import Normalize 
	from scipy.interpolate import interpn
	
	if ax is None :
		fig, ax = plt.subplots()
# 	else:
# 		fig = ax.figure
	x = np.array(x)
	y = np.array(y)
	data, x_e, y_e = np.histogram2d(x, y, bins = bins, density = True)
	z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)

	#To be sure to plot all data
	z[np.where(np.isnan(z))] = 0.0

	# Sort the points by density, so that the densest points are plotted last
	sort = True
	if sort :
		idx = z.argsort()
		x, y, z = x[idx], y[idx], z[idx]

	ax.scatter( x, y, c=z, **kwargs )

# 	norm = Normalize(vmin = np.min(z), vmax = np.max(z))
# 	cbar = fig.colorbar(cm.ScalarMappable(norm = norm), ax=ax)
# 	cbar.ax.set_ylabel('Density')

	return ax
@vc
def _plot_two_way_correlation_20240901(
	object_dict, xkeys=None, ykeys=None, labels=None,
	use_density_scatter=False, scatter_kw={}, 
	scatter_style_func={}, 
	value_func=None,
	filter_func=None,
	title=None, 
	identity_line_kw={},
	odr_line_kw=None, 
	correlation_kw={}, 
	skip_repeat=True, 
	skip_same_keys=False,
	axs=None):
		
	
	# Update default keys
	keys = list(object_dict.keys())
	if xkeys is None:
		xkeys = keys
	if ykeys is None:
		ykeys = keys
	if labels is None:
		labels = {k:k for k in itertools.chain(xkeys, ykeys)}
	
	if axs is None:
		fig, axs = plt.subplots(len(ykeys), len(xkeys), figsize=(len(xkeys)*3, len(ykeys)*3), squeeze=False)
	else:
		fig = axs[0][0].figure
		
	used_pairs = set() 
	scatter_kw = {"s":1, "alpha":0.5, **scatter_kw}
	for xidx, xkey in enumerate(xkeys):
		for yidx, ykey in enumerate(ykeys):
			ax = axs[yidx][xidx]
			t = tuple(sorted([xkey, ykey]))
			if (skip_repeat and t in used_pairs) or (skip_same_keys and xkey == ykey):
				for spine in ax.spines.values():
					spine.set_visible(False)
				ax.set_xticks([])
				ax.set_yticks([])
				continue
			used_pairs.add(t)
# 			if xkey == ykey:
# 				ax.add_artist(AnchoredText(xkey, loc="center", frameon=False, prop={"fontsize":20}))
# 				for spine in ax.spines.values():
# 					spine.set_visible(False)
# 				ax.set_xticks([])
# 				ax.set_yticks([])
# 				continue
			e1 = object_dict[xkey]
			e2 = object_dict[ykey]
			if isinstance(e1, dict) and isinstance(e2, dict):
				datapoints = [(k, e1[k], e2[k]) for k in sorted(set(e1.keys()).intersection(set(e2.keys())))]
				if filter_func is not None:
					datapoints = [(k, x, y) for k, x, y in datapoints if filter_func(k, x, y)]
				ks, xs, ys = safe_inverse_zip(datapoints, 3)
				custom_kw = {k:[func(*d) for d in datapoints] for k, func in scatter_style_func.items()}
			else:
				datapoints = [(x, y) for x, y in zip(e1, e2) if filter_func is None or filter_func(x, y)]
				xs, ys = safe_inverse_zip(datapoints, 2)
			if value_func is not None:
				xs = list(map(value_func, xs))
				ys = list(map(value_func, ys))
			custom_kw = {k:[func(x, y) for x, y in zip(xs, ys)] for k, func in scatter_style_func.items()}
			if use_density_scatter:
				_plot_density_scatter_20240901(xs, ys, ax=ax, **{**scatter_kw, **custom_kw})
			else:
				ax.scatter(xs, ys, **{**scatter_kw, **custom_kw})
			
			if identity_line_kw is not None:
				identity_line_kw = {"color":"grey", "ls":"--", **identity_line_kw}
				ax.axline([0, 0], slope=1, **identity_line_kw)
			if odr_line_kw is not None:
				_add_scatter_odr_line_20240701(xs, ys, **odr_line_kw, ax=ax)
			if correlation_kw is not None:
				_add_scatter_correlation_20240701(xs, ys, **correlation_kw, ax=ax)
			ax.set_aspect("equal")
			bpu._plt_equal_xylim_20240901(ax)
			lim = ax.get_xlim()
			ax.yaxis.set_major_locator(AutoLocator())
			ax.set_xticks(ax.get_yticks())
			ax.set_yticks(ax.get_yticks())
			ax.set_xlim(lim)
			ax.set_ylim(lim)
	for xidx, xkey in enumerate(xkeys):
		axs[-1][xidx].set_xlabel(labels[xkey])
	for yidx, ykey in enumerate(ykeys):
		axs[yidx][0].set_ylabel(labels[ykey])
	if title is not None:
		fig.suptitle(title)
	return fig