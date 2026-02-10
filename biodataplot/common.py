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
def _plot_ratio_bar_20250201(x=None, y=None, ytotal=None, target_key=None, text_height_ratio=0.5, maxy=None, text_kw={}, bar_kw={}, plot_pvalues=False, plot_pvalues_kw={}, ax=None):
	'''
	Plot ratio bar
	'''
	import pandas
	if ax is None:
		ax = plt.gca()
	if isinstance(y, pandas.core.frame.DataFrame):
		ykeys = list(y.columns)
		ytotal = list(np.sum(y, axis=0))
		y = list(y.loc[target_key])

	elif isinstance(y, dict):
		if ytotal is not None:
			if not isinstance(ytotal, dict):
				raise Exception("ytotal must be a dict if y is provided as dict")
			ykeys = list(ytotal.keys())
			ytotal = [ytotal[k] for k in ykeys]
		else:
			ykeys = list(y.keys())
		y = [y[k] for k in ykeys]
	else:
		ykeys = None
	if ytotal is not None:
		y = list(zip(y, ytotal))
	r = [e / t if t > 0 else 0 for e, t in y]
	if x is None:
		x = list(range(len(y)))
	_plot_fast_bar_20240601(x, r, ax=ax, **bar_kw)
	if ykeys is not None:
		ax.set_xticks(x)
		ax.set_xticklabels(ykeys)
	if maxy is None:
		mx = max(r)
	else:
		mx = maxy
	for i, (e, t) in zip(x, y):
		if t == 0:
			texty = mx/2
		elif e/t >= mx * text_height_ratio:
			texty = e/t/2
		else:
			texty = mx/2 + e/t/2
		text_kw = {"rotation":90, "ha":"center", "va":"center", **text_kw}
		ax.text(i, texty, f"{e} / {t}", **text_kw)
	if plot_pvalues:
		default_y_min = ax.get_ylim()[1]
		default_vspace = (ax.get_ylim()[1] - ax.get_ylim()[0]) / 10
		updated_plot_groups_pvalues_kwargs = {"y_min": default_y_min, 
											"vspace": default_vspace,
											"pvalue_func": "fisher_exact_total",
											**plot_pvalues_kw
											}
		_plot_pvalues_20250201(y, **updated_plot_groups_pvalues_kwargs, positions=x)
		
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
def _plot_ranked_values_20250201(data, plot_kw={}, plot_kw_dict={}, ax=None):
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
			y = [sy for sy in y if np.isfinite(sy)]
			artist = ax.scatter(np.arange(len(y)) + start_idx, sorted(y), **group_plot_kw, label=g)
			start_idx += len(y)
	else:
		y = data
		y = [sy for sy in y if np.isfinite(sy)]
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
# 	ax.axline([0, b], slope=m, **plot_kw)
	myodr = odr.ODR(data, model, beta0=[m, b])
	myodr.set_job(fit_type=0)
	myoutput = myodr.run()
	m, b = myoutput.beta
	return ax.axline([0, b], slope=m, **plot_kw)
@vc
def _add_scatter_odr_line_20250201(x, y, method="unilinear", plot_kw={}, ax=None):
	from scipy import odr
	if ax is None:
		ax = plt.gca()
	if method == "unilinear":
		model = odr.unilinear
	else:
		raise Exception("Unsupported method")
	if len(x) != len(y):
		raise Exception("Mismatched length of x and y")
	x, y = safe_inverse_zip([[sx, sy] for sx, sy in zip(x, y) if np.isfinite(sx) and np.isfinite(sy)], 2)
	if len(x) < 2:
		raise Exception("Insufficient data to generate odr line")
	data = odr.RealData(x, y)
	m, b = np.polyfit(x, y, 1)
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
def _add_scatter_correlation_20250201(
		x, y, method="pearson",
		show_r=True, show_n=False, show_p=False, use_rsquare=False, 
		text_kw={}, ax=None
	):
	import scipy.stats


	if ax is None:
		ax = plt.gca()
	if len(x) != len(y):
		raise Exception("Mismatched length of x and y")
	x, y = safe_inverse_zip([[sx, sy] for sx, sy in zip(x, y) if np.isfinite(sx) and np.isfinite(sy)], 2)
	if len(x) < 2:
		raise Exception("Insufficient data to generate correlation")
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
		lines.append(f"{bpu._format_pvalue_20250201(pvalue)}")
	if show_n:
		lines.append(f"$n={len(x)}$")
	atext = AnchoredText("\n".join(lines), **{"loc":"upper left","frameon":False, "prop":{"fontsize":14}, **text_kw})
	ax.add_artist(atext)
	return atext

@vc
def _plot_pvalue_20250201(xs, y, p, h=0, textspace=0, transform=None, vert=True, text_kw={}, pvalue_format_kw={}, ax=None):
	if ax is None:
		ax = plt.gca()
	if transform is None:
		transform = ax.transData
	
	if vert:
		ax.text(((xs[0] + xs[1]) / 2), y, bpu._format_pvalue_20250201(p, **pvalue_format_kw), ha="center", va="bottom", transform=transform, **text_kw)
		ax.plot(xs, (y, y), color="black", transform=transform)
		ax.plot((xs[0], xs[0]), (y - h, y), color="black", transform=transform)
		ax.plot((xs[1], xs[1]), (y - h, y), color="black", transform=transform)
		
	else:
		ax.text(y + textspace, ((xs[0] + xs[1]) / 2), bpu._format_pvalue_20250201(p, **pvalue_format_kw), ha="left", va="center", transform=transform, **text_kw)
		ax.plot((y, y), xs, color="black", transform=transform)
		ax.plot((y - h, y), (xs[0], xs[0]), color="black", transform=transform)
		ax.plot((y - h, y), (xs[1], xs[1]), color="black", transform=transform)

@vc
def _plot_pvalues_20250201(data, y_min, vspace, pvalue_func, h=0, textspace=0, positions=None, width_shrink=0, comparison_pairs="all", transform=None, vert=True, text_kw={}, pvalue_format_kw={}, ax=None):
	'''
	'''
	import scipy.stats
	
	if ax is None:
		ax = plt.gca()
		
	if isinstance(data, dict):
		data = list(data.values())
	
	if isinstance(comparison_pairs, str):
		if comparison_pairs == "all":
			comparison_pairs = itertools.combinations(range(len(data)), 2)
		elif comparison_pairs == "adjacent":
			comparison_pairs = [[i, i+1] for i in range(len(data) - 1)]
		else:
			raise Exception("Unidentified comparison pairs")
	if positions is None:
		positions = range(len(data))
	y = y_min
	if pvalue_func == "ttest_ind":
		pvalue_func = lambda d0, d1: scipy.stats.ttest_ind(d0, d1).pvalue
	elif pvalue_func == "ttest_rel":
		pvalue_func = lambda d0, d1: scipy.stats.ttest_rel(d0, d1).pvalue
	elif pvalue_func == "fisher_exact":
		pvalue_func = lambda d0, d1: scipy.stats.fisher_exact([[d0[0], d1[0]], [d0[1], d1[1]]]).pvalue
	elif pvalue_func == "fisher_exact_total":
		pvalue_func = lambda d0, d1: scipy.stats.fisher_exact([[d0[0], d1[0]], [d0[1] - d0[0], d1[1] - d1[0]]]).pvalue
	else:
		pass
	
	for pair in comparison_pairs:
		_plot_pvalue_20250201((positions[pair[0]] + width_shrink / 2, positions[pair[1]] - width_shrink / 2), y, pvalue_func(data[pair[0]], data[pair[1]]), h=h, textspace=textspace, transform=transform, vert=vert, text_kw=text_kw, pvalue_format_kw=pvalue_format_kw, ax=ax)
		y += vspace
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
def _plot_density_scatter_20250201(x, y, bins=20, ax=None, **kwargs):
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
	if len(x) != len(y):
		raise Exception("Mismatched length of x and y")
	x, y = safe_inverse_zip([[sx, sy] for sx, sy in zip(x, y) if np.isfinite(sx) and np.isfinite(sy)], 2)
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

@vc
def _plot_two_way_correlation_20250201(
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
				_plot_density_scatter_20250201(xs, ys, ax=ax, **{**scatter_kw, **custom_kw})
			else:
				ax.scatter(xs, ys, **{**scatter_kw, **custom_kw})
			
			if identity_line_kw is not None:
				identity_line_kw = {"color":"grey", "ls":"--", **identity_line_kw}
				ax.axline([0, 0], slope=1, **identity_line_kw)
			if odr_line_kw is not None:
				_add_scatter_odr_line_20250201(xs, ys, **odr_line_kw, ax=ax)
			if correlation_kw is not None:
				_add_scatter_correlation_20250201(xs, ys, **correlation_kw, ax=ax)
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


@vc
def _plot_histogram_20241201(data, nbins=10, bin_size=None, min_value=None, max_value=None, left_bound=True, right_bound=True, cumulative=False, plot_kw={}, bar_kw={}, ax=None):
	if ax is None:
		ax = plt.gca()
	if min_value is None:
		min_value = min(data)
	if max_value is None:
		max_value = max(data)
	
	if nbins is not None and bin_size is not None:
		nbins = None # If both supplied, use bin_size
	print(nbins, bin_size)
	if nbins is None and bin_size is None:
		raise Exception("You need to set either nbins or bin_size")
	elif nbins is not None and bin_size is None:
		bin_size = (max_value - min_value) / nbins
		bins = np.linspace(min_value, max_value, nbins+1)
	elif nbins is None and bin_size is not None:
		bins = np.arange(min_value, max_value + bin_size, step=bin_size)
		nbins = len(bins) - 1
	
	step_size=bin_size
	max_value = min_value + nbins * bin_size
	if not right_bound:
		bins = np.append(bins, np.inf)
	if not left_bound:
		bins = np.concatenate([[-np.inf], bins])
	h = np.histogram(data, bins=bins)
	if not left_bound: 
		x = np.concatenate([[h[1][1] - step_size / 2], h[1][1:-1] + step_size/2])
	else:
		x = h[1][:-1] + step_size/2
		
	_normalize = None # I don't wan to use this now
	if _normalize is None:
		y = h[0]/sum(h[0])
	else:
		y = h[0] / _normalize 
	if cumulative:
		y = np.cumsum(y)
	if plot_kw is not None:
		ax.plot(x, y, **plot_kw)
	if bar_kw is not None: 
		_plot_fast_bar_20240601(x, y, width=step_size, ax=ax, **{"fill_kw":{"alpha":0.2}, **bar_kw})
	ax.set_xlim(min(x) - step_size / 2, max(x) + step_size / 2)
	if nbins < 11:
		xt = np.linspace(min_value, max_value, nbins+1)
	else:
		xt = np.linspace(min_value, max_value, 5)
	ax.set_xticks(xt)
		
	return ax

@vc
def _plot_volcano_20250201(x, y=None, logFC_col=None, pvalue_col=None, logFC_cutoff=1, pvalue_cutoff=0.05, pos_color=None, neg_color=None, insig_color=None, scatter_plot_kw={}, ax=None):
	import pandas as pd
	if ax is None:
		ax = plt.gca()
	
	if isinstance(x, pd.DataFrame):
		df = x.dropna()
		x = list(df[logFC_col])
		y = list(df[pvalue_col])
	
	if pos_color is None: pos_color = '#e41a1c'
	if neg_color is None: neg_color = '#377eb8'
	if insig_color is None: insig_color = 'grey'
	min_value = min(i for i in y if i != 0)
	y = -np.log10([min_value if i == 0 else i for i in y])
	if pvalue_cutoff is not None and logFC_cutoff is not None:
		pvalue_cutoff = -np.log10(pvalue_cutoff)
		c = [(pos_color if ix >= abs(logFC_cutoff) else (neg_color if ix <= -abs(logFC_cutoff) else insig_color)) if iy >= pvalue_cutoff else insig_color for ix, iy in zip(x, y)]
	else:
		c = insig_color
	scatter_plot_kw = {"s":1, **scatter_plot_kw}
	ax.scatter(x, y, c=c, **scatter_plot_kw)
	ax.set_xlabel("log2 Fold Change")
	ax.set_ylabel("-log10 pvalue")
	
	
@vc
def _plot_violin_20250701(data, positions=None, width=0.8, *, 
						use_same_area=False, facecolor="#7fc97f", 
						edgecolor="black", half="both", showmedians=True, showextrema=True, 
						show_n_properties=None, plot_pvalues=False, plot_pvalues_kw={},
						ax=None):
	'''
	show_n_properties: if not None, [[heights], {ax text prop}]
	'''
	from commonhelper import isIterable, safe_inverse_zip
	import scipy.stats
	def _violin_adjacent_values(vals, q1, q3):
		upper_adjacent_value = q3 + (q3 - q1) * 1.5
		upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])
	
		lower_adjacent_value = q1 - (q3 - q1) * 1.5
		lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
		return lower_adjacent_value, upper_adjacent_value
	def _get_violin_widths(data, max_width, minx=None, maxx=None, est_num=10000, bw_method=None):
		
		widths = []
		for i in range(len(data)):
			density = scipy.stats.gaussian_kde(data[i], bw_method=bw_method)
			minx_arg = min(data[i]) if minx is None else minx
			maxx_arg = max(data[i]) if maxx is None else maxx
			xs = np.linspace(minx_arg, maxx_arg, est_num)
			ys = density(xs)
			widths.append(np.max(ys))
		factor = max_width / max(widths)
		return [width * factor for width in widths]
	if ax is None:
		ax = plt.gca()
	
	if positions is None:
		positions = list(range(len(data)))
	x = positions
	ax.set_xticks(x)
	
	if isinstance(data, dict):
		ax.set_xticklabels(list(data.keys()))
		y = list(data.values())
	else:
		y = data
	
	xplot, yplot = safe_inverse_zip([[tx, ty] for tx, ty in zip(x, y) if len(ty) > 1], 2)
# 	x, y = xplot, yplot
	if use_same_area:
		parts = ax.violinplot(
			yplot, showmeans=False, showmedians=False,
				showextrema=False, widths=_get_violin_widths(yplot, max_width=width, est_num=1000), positions=xplot)
	else:
		parts = ax.violinplot(
			yplot, showmeans=False, showmedians=False,
			showextrema=False, widths=width, positions=xplot)

	if isinstance(facecolor, str):
		for pc in parts['bodies']:
			pc.set_facecolor(facecolor)
			pc.set_edgecolor(edgecolor)
			pc.set_alpha(1)
			if half == "both":
				pass
			elif half == "left":
				m = np.mean(pc.get_paths()[0].vertices[:, 0])
				pc.get_paths()[0].vertices[:, 0] = np.clip(pc.get_paths()[0].vertices[:, 0], -np.inf, m)
			elif half == "right":
				m = np.mean(pc.get_paths()[0].vertices[:, 0])
				pc.get_paths()[0].vertices[:, 0] = np.clip(pc.get_paths()[0].vertices[:, 0], m, np.inf)
			else:
				raise Exception()
	else:
		for pc, fc in zip(parts['bodies'], facecolor):
			pc.set_facecolor(fc)
			pc.set_edgecolor(edgecolor)
			pc.set_alpha(1)
			
	quartile1, medians, quartile3 = list(zip(*[np.percentile(d, [25, 50, 75]) for d in yplot]))
	whiskers = np.array([_violin_adjacent_values(sorted_array, q1, q3) for sorted_array, q1, q3 in zip(yplot, quartile1, quartile3)])
	whiskersMin, whiskersMax = whiskers[:, 0], whiskers[:, 1]

	inds = xplot
		
	if showmedians:
		filled_marker_style = dict(marker='o', 
								   markerfacecolor="white",
								   markerfacecoloralt=(0,0,0,0),
								   markeredgecolor=(0,0,0,0))			
		if half == "both":
			ax.scatter(inds, medians, marker='o', color='white', s=30, zorder=3) # Draw middle white dot
		elif half == "left":
			ax.plot(inds, medians, fillstyle="left", lw=0, **filled_marker_style)
		elif half == "right":
			ax.plot(inds, medians, fillstyle="right", lw=0, **filled_marker_style)


	if quartile1 is not None:
		if half == "both":
			ax.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5) # Draw thick black line (quantile to quantile)
		elif half == "left":
			pass
	if showextrema:
		ax.vlines(inds, whiskersMin, whiskersMax, color='k', linestyle='-', lw=1) # Draw thin black line (adjacent to adjacent)
	if show_n_properties is not None:
		hs = show_n_properties[0] if isIterable(show_n_properties[0]) else ([show_n_properties[0]] * len(y)) 
		for i, h, ty in zip(x, hs, y):
			ax.text(i, h, f"n={len(ty)}", {"ha":"center", **show_n_properties[1]})
	if plot_pvalues:
		default_y_min = ax.get_ylim()[1]
		default_vspace = (ax.get_ylim()[1] - ax.get_ylim()[0]) / 10
		updated_plot_groups_pvalues_kwargs = {"y_min": default_y_min, 
											"vspace": default_vspace,
											"pvalue_func": "ttest_ind",
											**plot_pvalues_kw
											}
		_plot_pvalues_20250201(y, **updated_plot_groups_pvalues_kwargs, positions=x, ax=ax)

