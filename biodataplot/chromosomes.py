import sys
import simplevc
simplevc.register(sys.modules[__name__])

@vc
def _plot_chromosomes_regions_density_20250910(
	data, ref_size_dict, centromeres,
	bin_size=100000,
	included_chrs=None, excluded_chrs=None,
	height=0.8, 
	corner_width = 1000000, colormap=None, 
	ncols=2,
	xspace = 50000000, textspace = 40000000,
	scalebar_size=100000000, scalebar_text="100 Mbp", scalebar_kw={},
	cbx = 1.1, cby=.8, cbw= .02, cbh = .2, cborientation="horizontal",
	ax=None
):
	from collections import defaultdict
	import matplotlib
	import matplotlib.pyplot as plt
	from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
	import numpy as np
	
	def create_chromosome_patch(chrom_start, chrom_end, left_acen, right_acen, bottom, top, corner_width, corner_height):
		verts = [
			(chrom_start+corner_width, bottom),# Left bottom start of chromosome
			(left_acen, bottom), # Left bottom before acen
			(right_acen, top),# Right top after acen
			(chrom_end-corner_width, top), # Right top end of chromosome
			(chrom_end, top), # Right top CORNER PT-1
			(chrom_end, top-corner_height), # Right top CORNER PT-2
			(chrom_end, bottom+corner_height),# Right bottom CORNER PT-2
			(chrom_end, bottom),# Right bottom CORNER PT-1
			(chrom_end-corner_width, bottom), # Right bottom end of chromosome
			(right_acen, bottom), # Right bottom after accen
			(left_acen, top), # Left top before accen
			(chrom_start+corner_width, top),# Left top start of chromosome
			(chrom_start, top),# CORNER PT-1
			(chrom_start, top-corner_height),# CORNER PT-2
			(chrom_start, bottom+corner_height), # CORNER PT-2
			(chrom_start, bottom), # CORNER PT1
			(chrom_start+corner_width, bottom)# Left bottom start of chromosome
		]
		codes = [
			matplotlib.path.Path.MOVETO,
			matplotlib.path.Path.LINETO,
			matplotlib.path.Path.LINETO,
			matplotlib.path.Path.LINETO,
			matplotlib.path.Path.CURVE3,
			matplotlib.path.Path.CURVE3,
			matplotlib.path.Path.LINETO,
			matplotlib.path.Path.CURVE3,
			matplotlib.path.Path.CURVE3,
			matplotlib.path.Path.LINETO,
			matplotlib.path.Path.LINETO,
			matplotlib.path.Path.LINETO,
			matplotlib.path.Path.CURVE3,
			matplotlib.path.Path.CURVE3,
			matplotlib.path.Path.LINETO,
			matplotlib.path.Path.CURVE3,
			matplotlib.path.Path.CURVE3,
		]
		rounded_verts = matplotlib.path.Path(verts, codes)
		rounded_verts = matplotlib.patches.PathPatch(rounded_verts, facecolor='orange', lw=2)
		return rounded_verts
	
	if included_chrs is not None:
		ref_size_dict = {k : v for k, v in ref_size_dict.items() if k in included_chrs}
	if excluded_chrs is not None:
		ref_size_dict = {k : v for k, v in ref_size_dict.items() if k not in excluded_chrs}
	if colormap is None:
		colormap = {k:"blue" for k in data.keys()}	
	if ax is None:
		fig, ax = plt.subplots(figsize=(3*ncols, 6))
	
	# Determine the number of chromosomes and maximum size of each column
	fncols = int(np.ceil(len(ref_size_dict) / ncols))
	col_sizes = defaultdict(list)
	for idx, (ref, size) in enumerate(ref_size_dict.items()):
		col_sizes[idx // fncols].append(size)
	col_max_size = {k:max(sizes) for k, sizes in col_sizes.items()}
	
	# Plot chromosome names
	col_start_pos = {k:sum([col_max_size[i] + xspace for i in range(k)]) + xspace for k in col_max_size}
	for idx, (ref, size) in enumerate(ref_size_dict.items()):
		xmod = col_start_pos[idx // fncols]
		ax.text(xmod - textspace, idx % fncols,ref, va="center")
	
	# Calculate density and create the artists. They are not added to axes yet.
	regions_chr_dict = defaultdict(list)
	color_list = []
	boxes_list = []
	recs_list = []
	alphas_list = []
	ymod = 0 - height / 2
	for k, regions in data.items():
		for r in regions:
			regions_chr_dict[r.genomic_pos.name].append(r.genomic_pos) 
		boxes=[]
		recs = []
		alphas = []
		for idx, (ref, size) in enumerate(ref_size_dict.items()):
			xmod = col_start_pos[idx // fncols]
			h1 = idx % fncols + ymod
			h2 = h1 + height / len(data)
			# Signals
			binned = defaultdict(int)
			for r in regions_chr_dict[ref]:
				for i in range((r.genomic_pos.start - 1) // bin_size, (r.genomic_pos.stop - 1) // bin_size + 1):
					binned[i] += 1
			for i, v in binned.items():
				recs.append(matplotlib.patches.Rectangle((i * bin_size + xmod, h1), bin_size, h2 - h1))
				alphas.append(v)
			
			# Chroms
			acens = [g for g in centromeres if g.name == ref]
			s1, s2 = min([r.start for r in acens]), max([r.stop for r in acens])
			p = create_chromosome_patch(0 + xmod, size + xmod, s1 + xmod, s2 + xmod, h1, h2, corner_width, height/len(data)/4)
			boxes.append(p)
			

		color = colormap[k]
		color_list.append(color)
		boxes_list.append(boxes)
		recs_list.append(recs)
		alphas_list.append(alphas)
		ymod += height / len(data)

	max_alphas = []
	for color, boxes, recs, alphas in zip(color_list, boxes_list, recs_list, alphas_list):
		max_alpha = max(alphas)
		max_alphas.append(max_alpha)
		alphas = np.array(alphas) / max_alpha
		pc = matplotlib.collections.PatchCollection(recs, alpha=alphas, facecolor=color)#matplotlib.colors.to_hex(color))
		ax.add_collection(pc)
		pc = matplotlib.collections.PatchCollection(boxes, edgecolor="k", lw=1, facecolor="None") 
		ax.add_collection(pc)

	ax.set_xlim(0, xspace * (len(col_max_size) + 1) + sum(col_max_size.values()))
	ax.set_ylim(-height, fncols -1 + height)
	ax.invert_yaxis()
	
	# Scalebar
	if scalebar_kw is not None:
		scalebar_kw = {
			'loc': 'center left', 
			'color': 'black',
			'frameon': False,
			'size_vertical': 0,
			'bbox_to_anchor': [1, 0.5],
			'bbox_transform': ax.transAxes,
			**scalebar_kw
		}
		scalebar = AnchoredSizeBar(
			ax.transData,
			scalebar_size, 
			scalebar_text, 
			**scalebar_kw,
		)
		ax.add_artist(scalebar)
	ax.axis("off")
	
	# Color Bar
	max_alpha = dict(list(zip(data.keys(), max_alphas)))
	orientation = cborientation
	if orientation != "vertical":
		tmp = cbw
		cbw = cbh
		cbh = tmp
	for cidx, (category, color) in enumerate(reversed(colormap.items())):
		iax= ax.inset_axes([cbx + (cidx * cbw*3 if orientation == "vertical" else 0), cby + (cidx * cbh*3 if orientation != "vertical" else 0), cbw, cbh])
		norm = matplotlib.colors.Normalize(vmin=0, vmax=max_alpha[category])
		cmap = matplotlib.colors.ListedColormap(np.linspace([*matplotlib.colors.to_rgb(color),0], ([*matplotlib.colors.to_rgb(color),1]), 100))
		ax.figure.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
					 cax=iax, orientation=orientation)
		# Add label
		if (orientation == "vertical" and cidx == len(colormap) - 1) or (orientation == "horizontal" and cidx == 0):
			func = iax.set_ylabel if orientation == "vertical" else iax.set_xlabel
			func("Density (log10 regions\nin 100kb window)")
		if orientation == "vertical":
			iax.set_xlabel(category, rotation=90)
			iax.tick_params("y", rotation=90)
		else:
			iax.set_ylabel(category, rotation=0, ha="right", va="center")
