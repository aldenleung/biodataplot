import sys
import simplevc
simplevc.register(sys.modules[__name__])

import io
import os
import tempfile
from collections import defaultdict
import re

import numpy as np
import matplotlib
from matplotlib import scale as mscale
from matplotlib import transforms as mtransforms
from matplotlib.ticker import FuncFormatter
from matplotlib.path import get_path_collection_extents


@vc
def _plt_share_x_axes_20221007(axs, display_xticklabel="bottom"):
	'''
	Create shared x axes for matplotlib subplots axs in the same column. 
	If one desires to share x axes for all axs, use `sharex=True` when creating the subplots

	:Example:
	
	.. code-block:: python
		
		import matplotlib.pyplot as plt
		
		fig, axs = plt.subplots(2, 2)
		axs[0][0].plot([0, 1], [0, 1]); axs[0][0].text(0.75, 0.5, "x:0-1\\ny:0-1", ha="left", va="center", transform=axs[0][0].transAxes)
		axs[0][1].plot([0, 2], [0, 2]); axs[0][1].text(0.75, 0.5, "x:0-2\\ny:0-2", ha="left", va="center", transform=axs[0][1].transAxes)
		axs[1][0].plot([1.5, 0], [0, 1.5]); axs[1][0].text(0.75, 0.5, "x:0-1.5\\ny:0-1.5", ha="left", va="center", transform=axs[1][0].transAxes)
		axs[1][1].plot([0, 1], [0, 1]); axs[1][1].text(0.75, 0.5, "x:0-1\\ny:0-1", ha="left", va="center", transform=axs[1][1].transAxes)
		fig.suptitle("No shared axes")
		
	:Example:
	
	.. code-block:: python
			
		import matplotlib.pyplot as plt

		fig, axs = plt.subplots(2, 2)
		axs[0][0].plot([0, 1], [0, 1]); axs[0][0].text(0.75, 0.5, "x:0-1\\ny:0-1", ha="left", va="center", transform=axs[0][0].transAxes)
		axs[0][1].plot([0, 2], [0, 2]); axs[0][1].text(0.75, 0.5, "x:0-2\\ny:0-2", ha="left", va="center", transform=axs[0][1].transAxes)
		axs[1][0].plot([1.5, 0], [0, 1.5]); axs[1][0].text(0.75, 0.5, "x:0-1.5\\ny:0-1.5", ha="left", va="center", transform=axs[1][0].transAxes)
		axs[1][1].plot([0, 1], [0, 1]); axs[1][1].text(0.75, 0.5, "x:0-1\\ny:0-1", ha="left", va="center", transform=axs[1][1].transAxes)
		plt_share_x_axes(axs)
		fig.suptitle("Shared x-axes")

	:Example:
	
	.. code-block:: python
			
		import matplotlib.pyplot as plt

		fig, axs = plt.subplots(2, 2)
		axs[0][0].plot([0, 1], [0, 1]); axs[0][0].text(0.75, 0.5, "x:0-1\\ny:0-1", ha="left", va="center", transform=axs[0][0].transAxes)
		axs[0][1].plot([0, 2], [0, 2]); axs[0][1].text(0.75, 0.5, "x:0-2\\ny:0-2", ha="left", va="center", transform=axs[0][1].transAxes)
		axs[1][0].plot([1.5, 0], [0, 1.5]); axs[1][0].text(0.75, 0.5, "x:0-1.5\\ny:0-1.5", ha="left", va="center", transform=axs[1][0].transAxes)
		axs[1][1].plot([0, 1], [0, 1]); axs[1][1].text(0.75, 0.5, "x:0-1\\ny:0-1", ha="left", va="center", transform=axs[1][1].transAxes)
		plt_share_y_axes(axs)
		fig.suptitle("Shared y-axes")
		
	
	:Example:
	
	.. code-block:: python
			
		import matplotlib.pyplot as plt

		fig, axs = plt.subplots(2, 2)
		axs[0][0].plot([0, 1], [0, 1]); axs[0][0].text(0.75, 0.5, "x:0-1\\ny:0-1", ha="left", va="center", transform=axs[0][0].transAxes)
		axs[0][1].plot([0, 2], [0, 2]); axs[0][1].text(0.75, 0.5, "x:0-2\\ny:0-2", ha="left", va="center", transform=axs[0][1].transAxes)
		axs[1][0].plot([1.5, 0], [0, 1.5]); axs[1][0].text(0.75, 0.5, "x:0-1.5\\ny:0-1.5", ha="left", va="center", transform=axs[1][0].transAxes)
		axs[1][1].plot([0, 1], [0, 1]); axs[1][1].text(0.75, 0.5, "x:0-1\\ny:0-1", ha="left", va="center", transform=axs[1][1].transAxes)
		plt_share_x_axes(axs)
		plt_share_y_axes(axs)
		fig.suptitle("Shared x-axes and y-axes")

	'''
	rows = len(axs)
	columns = len(axs[0])
	for j in range(columns):
# 		axs[0][j].get_shared_x_axes().join(*[axs[i][j] for i in range(rows)])
		axs[0][j].get_shared_x_axes()._grouper.join(*[axs[i][j] for i in range(rows)])
		if display_xticklabel == "bottom":
			r = range(rows - 1)
		elif display_xticklabel == "top":
			r = range(1, rows)
		elif display_xticklabel == "all":
			r = range(0)
		else:
			raise Exception()
		for i in r:
			axs[i][j].xaxis.set_tick_params(which='both', labelbottom=False, labeltop=False)
@vc
def _plt_share_y_axes_20221007(axs, display_yticklabel='left'):
	'''
	Create shared y axes for matplotlib subplots axs in the same row.
	
	See also :func:`~plt_share_x_axes`
	'''
	rows = len(axs)
	columns = len(axs[0])
	for i in range(rows):
		axs[i][0].get_shared_y_axes().join(*[axs[i][j] for j in range(columns)])
		if display_yticklabel == "right":
			r = range(columns - 1)
		elif display_yticklabel == "left":
			r = range(1, columns)
		elif display_yticklabel == "all":
			r = range(0)
		else:
			raise Exception()
		for j in r:
			axs[i][j].yaxis.set_tick_params(which='both', labelleft=False, labelright=False)

@vc
def _plt_register_asym_scale_20230101():
	'''
	Register a new scale, "asym" to matplotlib.scale. This scale is two distinct part divided at zero, where one could supply a parameter a to determine the ratio of p
	
	:Example:
	
	.. code-block:: python
	
		import matplotlib.pyplot as plt
		
		plt_register_asym_scale()
		
		# Make 0 at center of axis
		fig, axs = plt.subplots(1, 2)
		axs[0].margins(0)
		axs[0].plot([-10, 0, 5], [0,1,0])
		axs[0].set_title("Linear scale")
		
		axs[1].margins(0)
		axs[1].plot([-10, 0, 5], [0,1,0])
		xmin, xmax = axs[1].get_xlim()
		assert xmin < 0 < xmax
		axs[1].set_xscale("asym", a=xmax / abs(xmin))
		axs[1].set_title("Asym scale, zero at center of plot")
		fig.tight_layout()
	
	:Example:
	
	.. code-block:: python
		
		import matplotlib.pyplot as plt
		
		plt_register_asym_scale()
		
		# Note that plot must be separated at 0
		ratios = [0.5, 1, 2]
		n = len(ratios)
		fig, axs = plt.subplots(2, n)
		for i in range(n):
			axs[0][i].plot([-1, 0, 1],[0, 0.5, 1], "o-") # Correct usage
			axs[0][i].set_xscale("asym", a=ratios[i])
			axs[0][i].set_title(f"a={ratios[i]}\\nplot sep at 0")
		for i in range(n):
			axs[1][i].plot([-1, 1],[0, 1], "o-") # Incorrect usage
			axs[1][i].set_xscale("asym", a=ratios[i])
			axs[1][i].set_title(f"a={ratios[i]}\\nplot NOT sep at 0")
		fig.tight_layout()
		 
	'''
	# Adapted from stackoverflow with modification

	class AsymScale(mscale.ScaleBase):
		name = 'asym'

		def __init__(self, axis, **kwargs):
			mscale.ScaleBase.__init__(self, axis)
			self.a = kwargs.get("a", 1)

		def get_transform(self):
			return self.AsymTrans(self.a)

		def set_default_locators_and_formatters(self, axis):
			# possibly, set a different locator and formatter here.
			fmt = lambda x,pos: "{}".format(np.abs(x))
			axis.set_major_formatter(FuncFormatter(fmt))

		class AsymTrans(mtransforms.Transform):
			input_dims = 1
			output_dims = 1
			is_separable = True

			def __init__(self, a):
				mtransforms.Transform.__init__(self)
				self.a = a

			def transform_non_affine(self, x):
				return (x >= 0)*x + (x < 0)*x*self.a

			def inverted(self):
				return AsymScale.InvertedAsymTrans(self.a) 

		class InvertedAsymTrans(AsymTrans):

			def transform_non_affine(self, x):
				return (x >= 0)*x + (x < 0)*x/self.a
			def inverted(self):
				return AsymScale.AsymTrans(self.a)
			
	mscale.register_scale(AsymScale)

@vc
def _plt_adjust_text_positions_20240501(
		ax, pathcollections=None, texts=None,
		min_dist_for_anno_line = .1, step_size = 0.1, expand_size = 0.025,
		anno_plot_kw = {},
		max_trial = 10000):
	'''
	An automatic method to adjust text not to overlap with other texts and path collections.
	
	While there is a default value, you almost always need to customize min_dist_for_anno_line, step_size and expand_size to get the best results
	min_dist_for_anno_line: How long does the text move from the original point to generate a new anno line
	step_size: How long the text is moved each time. If the value is too small you will find solutions very slowly.
	expand_size: Margin size reserved around the text 
	'''

	def _getbb(sc, ax):
		# Learned from stackoverflow
		ax.figure.canvas.draw() # need to draw before the transforms are set.
		transform = sc.get_transform()
		transOffset = sc.get_offset_transform()
		offsets = sc._offsets
		paths = sc.get_paths()
		transforms = sc.get_transforms()
	
		if not transform.is_affine:
			paths = [transform.transform_path_non_affine(p) for p in paths]
			transform = transform.get_affine()
		if not transOffset.is_affine:
			offsets = transOffset.transform_non_affine(offsets)
			transOffset = transOffset.get_affine()
	
		if isinstance(offsets, np.ma.MaskedArray):
			offsets = offsets.filled(np.nan)
	
		bboxes = []
	
		if len(paths) and len(offsets):
			if len(paths) < len(offsets):
				# for usual scatters you have one path, but several offsets
				paths = [paths[0]]*len(offsets)
			if len(transforms) < len(offsets):
				# often you may have a single scatter size, but several offsets
				transforms = [transforms[0]]*len(offsets)
	
			for p, o, t in zip(paths, offsets, transforms):
				result = get_path_collection_extents(
					transform.frozen(), [p], [t],
					[o], transOffset.frozen())
				bboxes.append(result.transformed(ax.transData.inverted()))
	
		return bboxes	
	def _get_anc_pts(t):
		x0 = t.x0
		x1 = t.x1
		y0 = t.y0
		y1 = t.y1
		return np.array([
		[x0, (y0+y1)/2],
		[(x0+x1)/2, y0],
		[(x0+x1)/2, y1],
		[x1, (y0+y1)/2],])
	def _bbox_contains_bbox(query_bbox, ref_bbox):
		return (
			ref_bbox.x0 <= query_bbox.x0 
			and ref_bbox.x1 >= query_bbox.x1
			and ref_bbox.y0 <= query_bbox.y0 
			and ref_bbox.y1 >= query_bbox.y1 
		)
	def _expand_bbox(bbox, s):
		return matplotlib.transforms.Bbox([[bbox.x0 - s, bbox.y0 - s], [bbox.x1 + s, bbox.y1 + s]])
	
	
	vectors = [
	np.array([0, 1]),
	np.array([.707, .707]),
	np.array([1, 0]),
	np.array([.707, -.707]),
	np.array([0, -1]),
	np.array([-.707, -.707]),
	np.array([-1, 0]),
	np.array([-.707, .707])
	]

	ax_bbox = ax.transData.inverted().transform_bbox(ax.get_tightbbox())
	if pathcollections is None:
		pathcollections = [c for c in ax.get_children() if isinstance(c, matplotlib.collections.PathCollection)]
	if texts is None:
		texts = [c for c in ax.texts]
	bboxes = []
	for c in pathcollections:
		bboxes.extend(_getbb(c, ax))
	for t in texts:
		trials = 0
		ori_position = t.get_position()
		cur_bbox = _expand_bbox(ax.transData.inverted().transform_bbox(t.get_tightbbox()), expand_size)
		while any(bb.overlaps(cur_bbox) for bb in bboxes) or not _bbox_contains_bbox(cur_bbox, ax_bbox):
			t.set_position(ori_position + vectors[trials % len(vectors)] * step_size * (trials // len(vectors) + 1))
			cur_bbox = _expand_bbox(ax.transData.inverted().transform_bbox(t.get_tightbbox()),  expand_size)
			trials += 1
			if trials >= max_trial:
				break
		if (np.linalg.norm(ori_position - np.array(t.get_position()))) >= min_dist_for_anno_line:
			anc_pts = _get_anc_pts(ax.transData.inverted().transform_bbox(t.get_tightbbox()))
			dist, anc_pt = min((np.linalg.norm(ori_position - anc_pt), anc_pt) for anc_pt in anc_pts)
			ax.plot([ori_position[0], anc_pt[0]], [ori_position[1], anc_pt[1]], **anno_plot_kw)

		bboxes.append(ax.transData.inverted().transform_bbox(t.get_tightbbox()))
		
		

@vc
def _plt_change_ax_properties_20240501(ax, *, 
		ax_prop={}, 
		xticklabels_prop={}, yticklabels_prop={}, 
		xlabel_prop={}, ylabel_prop={}, 
		title_prop={}, 
		legend_prop={}, legend_text_prop={}, 
		text_prop={}, locator_prop={}, 
		additional_axhlines=[], additional_axvlines=[], additional_axlines=[], 
		additional_texts=[], additional_texts_transAxes=False, 
		adjust_text_positions_prop=None, 
		spines_prop_dict={}):
	'''
	Provide a way to wrap change of different ax properties into a single function
	
	'''
	ax.set(**ax_prop)
	for t in ax.get_xticklabels():
		t.set(**xticklabels_prop)
	for t in ax.get_yticklabels():
		t.set(**yticklabels_prop)
	ax.get_xaxis().get_label().set(**xlabel_prop)
	ax.get_yaxis().get_label().set(**ylabel_prop)
	ax.title.set(**title_prop)
	legend = ax.get_legend()
	if legend is not None:
		legend.set(**legend_prop)
		for text in legend.get_texts():
			text.set(**legend_text_prop)
	for child in ax.get_children():
		if isinstance(child, matplotlib.text.Text):
			if child is ax.title:
				continue
			child.set(**text_prop)
	ax.locator_params(**locator_prop)
	for p, prop in additional_axhlines:
		ax.axhline(p, **prop)
	for p, prop in additional_axvlines:
		ax.axvline(p, **prop)
	for args, kwargs in additional_axlines:
		ax.axline(*args, **kwargs)
	for args, kwargs in additional_texts:
		if additional_texts_transAxes:
			ax.text(*args, **kwargs, transform=ax.transAxes)
		else:
			ax.text(*args, **kwargs)		
	if adjust_text_positions_prop is not None:
		_plt_adjust_text_positions_20240501(ax, **adjust_text_positions_prop)
	for s, d in spines_prop_dict.items():
		ax.spines[s].set(**d)		
		
@vc
def _plt_change_figure_properties_20240501(fig, *, fig_prop={}, fig_suptitle=None, fig_suptitle_kwargs={}, fig_supxlabel=None, fig_supxlabel_kwargs={}, fig_supylabel=None, fig_supylabel_kwargs={}, ax_targets=None, ax_prop={}, xticklabels_prop={}, yticklabels_prop={}, xlabel_prop={}, ylabel_prop={}, title_prop={}, legend_prop={}, legend_text_prop={}, text_prop={}, locator_prop={}, additional_axhlines=[], additional_axvlines=[], additional_axlines=[], additional_texts=[], adjust_text_positions_prop=None, spines_prop_dict={}, additional_texts_transAxes=False):
	'''
	Provide a way to apply different properties across to the matplotlib figure including changes to the selected axs
	'''
	
	if ax_targets is None:
		axes = fig.axes
	else:
		axes = [fig.axes[k] for k in ax_targets]
	for ax in axes:
		_plt_change_ax_properties_20240501(ax, ax_prop=ax_prop, xticklabels_prop=xticklabels_prop, yticklabels_prop=yticklabels_prop, xlabel_prop=xlabel_prop, ylabel_prop=ylabel_prop, title_prop=title_prop, legend_prop=legend_prop, legend_text_prop=legend_text_prop, text_prop=text_prop, locator_prop=locator_prop, additional_axhlines=additional_axhlines, additional_axvlines=additional_axvlines, additional_axlines=additional_axlines, additional_texts=additional_texts, adjust_text_positions_prop=adjust_text_positions_prop, spines_prop_dict=spines_prop_dict, additional_texts_transAxes=additional_texts_transAxes)
	fig.set(**fig_prop)
	
	
	if fig_suptitle is not None:
		fig.suptitle(fig_suptitle, **fig_suptitle_kwargs)
	elif fig._suptitle is not None:
		fig._suptitle.set(**fig_suptitle_kwargs)
	if fig_supxlabel is not None:
		fig.supxlabel(fig_supxlabel, **fig_supxlabel_kwargs)
	elif fig._supxlabel is not None:
		fig._supxlabel.set(**fig_supxlabel_kwargs)
	if fig_supylabel is not None:
		fig.supylabel(fig_supylabel, **fig_supylabel_kwargs)
	elif fig._supylabel is not None:
		fig._supylabel.set(**fig_supylabel_kwargs)
		
@vc		
def _compose_SVG_panel_20221231(data, w=None, h=None, cols=None, rows=None, output=None, save_fig_kw={}, text_kw={}, text_height=12):
	'''
	Make use of svgutils, this method allows merging of multiple matplotlib.figure.Figure or SVG file into a single SVG file
	
	'''
	try:
		from svgutils.compose import Figure, Panel, Text, SVG 
		from lxml import etree 
	except:
		raise Exception("compose_SVG_panel requires svgutils and lxml packages")
	
	if w is None and h is None:
		raise Exception("You must provide either w or h")
	if len(data) == 0:
		raise Exception("Empty data")
	
	def convert(s):
		pt_pattern = re.compile("^([0-9]+(?:[.][0-9]+)?)pt$")
		return float(pt_pattern.match(s).group(1))
	svgfiles = []
	tmp_svgfiles = [] # This is temporary storage from fig
	if isinstance(data, dict):
		datakeys = list(data.keys())
		data = list(data.values())
	else:
		datakeys = None
	for d in data:
		if isinstance(d, str) or isinstance(d, io.IOBase):
			svgfiles.append(d)
		elif isinstance(d, matplotlib.figure.Figure):
			filename = tempfile.NamedTemporaryFile(mode='w+', suffix=".svg", delete=False).name
			d.savefig(filename, **save_fig_kw)
			svgfiles.append(filename)
			tmp_svgfiles.append(filename) 
		else:
			raise Exception("Unknown data type")
		
	if rows is None:
		rows = (len(svgfiles) + cols - 1) // cols
	
	text_kw = {**text_kw}
	row_ys = {0:0}
	row_max_heights = defaultdict(int)
	panels = []
	for idx, svgfile in enumerate(svgfiles):
		if isinstance(svgfile, str):
			f = open(svgfile, "rb")
		else:
			f = svgfile
		s = f.read()
		if isinstance(svgfile, str):
			f.close()
		e = etree.fromstring(s)
		attrib = e.attrib
		width, height = convert(attrib["width"]), convert(attrib["height"])
		row_idx = idx // cols
		col_idx = idx % cols
		
		# Get y
		if h is None:
			if row_idx not in row_ys:
				row_ys[row_idx] = row_ys[row_idx - 1] + row_max_heights[row_idx - 1] + (text_height if datakeys is not None else 0)
			y = row_ys[row_idx]
		else:
			y = h / rows * row_idx
		
		# Get scales
		scales = []
		if w is not None: 
			scales.append(w / cols / width)
		if h is not None: 
			if datakeys is not None:
				scales.append((h / rows - text_height) / height)
			else:
				scales.append(h / rows / height)
		scale = min(scales)
		row_max_heights[row_idx] = max(row_max_heights[row_idx], height * scale)
# 		element = Element(e).scale(scale)
# 		panels.append(Panel(element.scale(scale)).move(w / cols * col_idx, y))
# There is no way I can use Element instead of SVG for scaling. svgutils behavior. 
		if isinstance(svgfile, str):
			svginstance = SVG(svgfile)
		else:
			tmp = tempfile.NamedTemporaryFile(mode='w+', suffix=".svg", delete=False).name
			with open(tmp, 'wb') as tmpw:
				tmpw.write(s)
			svginstance = SVG(tmp)
			os.unlink(tmp)
		if datakeys is not None:
			textinstance = Text(datakeys[idx], 0, text_height, **text_kw)
			panelinstance = Panel(svginstance.scale(scale).move(0, text_height), textinstance).move(w / cols * col_idx, y)
		else:
			panelinstance =  Panel(svginstance.scale(scale)).move(w / cols * col_idx, y)
		panels.append(
			panelinstance
		)
	
	if h is None:
		if row_idx not in row_ys:
			row_ys[row_idx] = row_ys[row_idx - 1] + row_max_heights[row_idx - 1]
		h = row_ys[row_idx] + row_max_heights[row_idx] + ((row_idx + 1) * text_height if datakeys is not None else 0)
	figure = Figure(w, h, *panels)
	
	if output is not None:
		
		if output.lower().endswith(".svg"):
			figure.save(output)
		else:
			try:
				import cairosvg
			except:
				raise Exception("Saving to PDF, PNG or PS requires cairosvg")
			tmpfile = tempfile.NamedTemporaryFile(mode='w+', suffix=".svg", delete=False).name
			figure.save(tmpfile)
			
			if output.lower().endswith(".pdf"):
				cairosvg.svg2pdf(url=tmpfile, write_to=output)
			elif output.lower().endswith(".png"):
				cairosvg.svg2png(url=tmpfile, write_to=output)
			elif output.lower().endswith(".ps"):
				cairosvg.svg2ps(url=tmpfile, write_to=output)
			os.unlink(tmpfile)
	for filename in tmp_svgfiles:
		os.unlink(filename)
	return figure		