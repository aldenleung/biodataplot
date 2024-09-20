import sys
import simplevc
simplevc.register(sys.modules[__name__])

from collections import deque, Counter, defaultdict

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from genomictools import GenomicPos, GenomicCollection, GenomicAnnotation

from biodata.baseio import BaseIReader, get_text_file_extension
from biodata.bigwig import BigWigIReader
from biodata.bed import BED, BEDReader, BEDIReader, BEDGraph, BEDGraphReader, BEDGraphIReader
from biodata.gff import GFF3, GFF3Reader,GFF3IReader, GTF, GTFReader, GTFIReader
from biodata.fasta import FASTAReader, FASTAIReader
from commonhelper import safe_inverse_zip

# Only have plot_genome_view as visible now 

def _initiate_data_20240601(arg, formatter={}):
	'''
	Pre-process the track and returns the actual data to plot.
	
	1. Load data (if a file is provided)
	2. Guess the data type (if data type is not provided)
	3. Guess the track type  (if track type is not provided)
	
	4. Returns the track type, data type, initiated data and opened readers
	'''
	def initiate_reader(arg, iReader, bReader, bReader_func, kwargs):
		# Try to use iReader. If unsuccessful, use bReader
		try:
			data = iReader(arg, **kwargs)
		except:
			data = bReader.read_all(bReader_func, arg, **kwargs)
		return data
	def parse_file(arg, tracktype=None, datatype=None, file_parse_kwargs={}):
		# Alternative path to parse file with known data types
		if tracktype is not None:
			if tracktype == "arc":
				data = initiate_reader(arg, BEDGraphIReader, BEDGraphReader, GenomicCollection, {"dataValueType":lambda s: Counter(map(int, s.split(","))), **file_parse_kwargs})
				return data
		# Parse file based on file extension
		ext = get_text_file_extension(arg).lower()
		if ext == "bw" or ext == "bigwig":
			data = BigWigIReader(arg)
		elif ext == "bg" or ext == "bedgraph":
			data = initiate_reader(arg, BEDGraphIReader, BEDGraphReader, GenomicCollection, file_parse_kwargs)
		elif ext == "bed":
			data = initiate_reader(arg, BEDIReader, BEDReader, GenomicCollection, file_parse_kwargs)
		elif ext == "gff3":
			data = initiate_reader(arg, GFF3IReader, GFF3Reader, GenomicCollection, file_parse_kwargs)
		elif ext == "gtf":
			data = initiate_reader(arg, GTFIReader, GTFReader, GenomicCollection, file_parse_kwargs)
		elif ext == "fa" or ext == "fasta":
			data = initiate_reader(arg, FASTAIReader, FASTAReader, lambda fs: {f.name:f.seq for f in fs}, file_parse_kwargs)
		else:
			raise Exception("Unknown file type")
		return data
	
	def guess_datatype(arg):
		if isinstance(arg, BigWigIReader):
			datatype = "bigwig"
		elif isinstance(arg, FASTAIReader):
			datatype = "nucleotides"
		elif isinstance(arg, BEDGraphIReader):
			datatype = "bedgraph"
		elif isinstance(arg, BEDIReader):
			datatype = "bed"
		elif isinstance(arg, GFF3IReader):
			datatype = "gff3"
		elif isinstance(arg, GTFIReader):
			datatype = "gtf"
		elif isinstance(arg, dict):
			 datatype = "dict-nucleotides"
		elif isinstance(arg, GenomicCollection):
			if len(arg) == 0:
				datatype = "GenomicCollection"
			else:
				ar = next(iter(arg))
				if isinstance(ar, BED):
					datatype = "GenomicCollection-bed"
				elif isinstance(ar, BEDGraph):
					datatype = "GenomicCollection-bedgraph"	
				elif isinstance(ar, GFF3):
					datatype = "GenomicCollection-gff3"	
				elif isinstance(ar, GTF):
					datatype = "GenomicCollection-gtf"	
				else:
					datatype = "GenomicCollection" # Unknown data type
		else:
			raise Exception("Fail to guess data type for " + str(type(arg)))
		return datatype
	
	# Read file if needed
	if "file_parse_kwargs" in formatter:
		file_parse_kwargs = formatter["file_parse_kwargs"]
	else:
		file_parse_kwargs = {}
	if "tracktype" in formatter:
		file_parse_tracktype = formatter["tracktype"]
	else:
		file_parse_tracktype= None
	if "datatype" in formatter:
		file_parse_datatype = formatter["datatype"]
	else:
		file_parse_datatype = None
		
	readers = [] # For closing	
	if isinstance(arg, str, ):
		arg = parse_file(arg, file_parse_tracktype, file_parse_datatype, file_parse_kwargs) # Parse file into notable formats
		if isinstance(arg, BaseIReader):
			readers.append(arg)
	if isinstance(arg, tuple) or isinstance(arg, list):
		tmp = arg
		args = []
		for entry in tmp:
			if isinstance(entry, str):
				arg = parse_file(entry, file_parse_tracktype, file_parse_datatype, file_parse_kwargs)
				if isinstance(arg, BaseIReader):
					readers.append(arg)
			else:
				arg = entry
			args.append(arg)
	else: 
		args = [arg]
		
	# Guess datatype
	if "datatype" in formatter:
		datatype = formatter["datatype"]
	else:
		datatype = guess_datatype(args[0])
	
	# Guess tracktype
	if "tracktype" in formatter:
		tracktype = formatter["tracktype"]
	else:
		# Auto-interpret track type
		if datatype == "bigwig":
			tracktype = "signal"
		elif datatype == "nucleotides":
			tracktype = "nucleotide_density"
		else:
			tracktype = "annotation"
	
	
	return {"tracktype":tracktype, "datatype":datatype, "args":args, "readers": readers}
	
	
def _preprocess_track_data_20240601(idata, r, formatter):
	'''
	Process data on the region according to the initiated data
	Returns parsed data (dict)
	'''
	tracktype, datatype, args = [idata[k] for k in ["tracktype", "datatype", "args"]]
	# Process data
	#======================== Signal ========================#
	if tracktype == "signal":		
		x = []
		y = []
		for arg in args:
			if datatype == "bigwig" or datatype == "bedgraph":
				tmpx, tmpy = safe_inverse_zip(arg.values_dict(r).items(), 2)
			elif datatype.startswith("GenomicCollection"):
				tmpx, tmpy = safe_inverse_zip([(p, hit.dataValue) for hit in arg.find_overlaps(r) for p in range(hit.genomic_pos.ostart, hit.genomic_pos.ostop+1)], 2)
			else:
				raise Exception(f"Cannot plot signal track on data type - {datatype}")
			x.extend(tmpx)
			y.extend(tmpy)
		if "vmod" in formatter:
			y = list(formatter["vmod"](py) for py in y)			
		
		if "density" in formatter:
			density_x = []
			density_y = []
			density_kw = formatter["density"]
			win_size = density_kw["winsize"]
			
			expanded_r = GenomicPos(r.name, r.start - win_size // 2, r.stop - 1 + win_size // 2 + win_size % 2)
			for arg in args: 
				if datatype == "bigwig" or datatype == "bedgraph":
					count_dict = arg.values_dict(expanded_r)
				elif datatype == "GenomicCollection":
					count_dict = {p:hit.dataValue for hit in arg.find_overlaps(r) for p in range(hit.genomic_pos.ostart, hit.genomic_pos.ostop+1)}
				v = 0
				to_add = deque(sorted(count_dict.keys()))
				cur_pool = deque()
				sliding_window_values = []
				for i in range(r.start, r.stop + 1):
					start = i - win_size // 2
					stop = i - 1 + win_size // 2 + win_size % 2
					while len(to_add) > 0 and to_add[0] <= stop:
						idx = to_add.popleft()
						cur_pool.append(idx)
						v += count_dict[idx]
					while len(cur_pool) > 0 and cur_pool[0] < start:
						idx = cur_pool.popleft()
						v -= count_dict[idx]
					sliding_window_values.append(v)
				density_x.extend(list(range(r.start, r.stop + 1)))
				density_y.extend(sliding_window_values)		
			if "vmod" in formatter:
				density_y = list(formatter["vmod"](py) for py in density_y)			
			density_y = [i / win_size for i in density_y]	
		else:
			density_x = None
			density_y = None
		return {"x":x, "y":y, "density_x":density_x, "density_y":density_y}
	#======================== Arc ========================#
	elif tracktype == "arc":
		min_signal = formatter["min_signal"] if "min_signal" in formatter else 0
		hits = []
		for arg in args:
			if datatype == "bedgraph":
				iterator = arg.entries_iterator(r)
			elif datatype.startswith("GenomicCollection"):
				iterator = arg.find_overlaps(r)
			else:
				raise Exception()
			for hit in iterator:
				if min_signal > 0:
					cter = {d: cnt for d, cnt in hit.dataValue.items() if cnt >= min_signal}
					if len(cter) > 0:
						hits.append(BEDGraph(hit.chrom, hit.chromStart, hit.chromEnd, cter))
				else:
					hits.append(hit)
		return {"hits":hits}

	#======================== Annotation ========================#
	elif tracktype == "annotation":
		hits = []
		for arg in args:
			if datatype.startswith("GenomicCollection"):
				hits.extend(list(arg.find_overlaps(r)))
			elif datatype in ["bed", "bedgraph", "gff3", "gtf"]:
				hits.extend(arg.entries(r))
			else:
				raise Exception()
		if "filter_func" in formatter:
			filter_func = formatter["filter_func"]
			if isinstance(filter_func, str):
				filter_func = eval(filter_func, {})
			hits = [hit for hit in hits if filter_func(hit)]
				
		disjoint_regions_list = []
		for ar in hits:
			i = 0
			while i < len(disjoint_regions_list):
				if not ar.genomic_pos.overlaps(disjoint_regions_list[i][-1].genomic_pos):
					disjoint_regions_list[i].append(ar)
					break
				i += 1
			else:
				disjoint_regions_list.append(list())
				disjoint_regions_list[i].append(ar)
		return {"regions_list":disjoint_regions_list}
	#======================== Nucleotide density ========================#
	elif tracktype == "nucleotide_density":
		if len(args) != 1:
			raise Exception("You should only provide one seq dict for nucleotide density")
		win_size = formatter["winsize"]
		ds = defaultdict(list)
		
		for i in range(r.zstart, r.stop):
			start = i - win_size // 2
			stop = i + win_size // 2 + win_size % 2
			if start < 0:
				start = 0
			if datatype == "nucleotides":
				fir = [fir for fir in args if r.name in fir.faidx_dict][0]
				if stop > fir.faidx_dict[r.name].length:
					stop = fir.faidx_dict[r.name].length
				seq = fir[GenomicPos(r.name, start, stop)].seq.upper()
			elif datatype == "dict-nucleotides":
				seq_dict = [seq_dict for seq_dict in args if r.name in seq_dict][0]
				if stop > len(seq_dict[r.name]):
					stop = len(seq_dict[r.name])
				seq = seq_dict[r.name][start:stop].upper()
			else:
				raise Exception()
			if len(seq) == 0:
				for c in "ACGT":
					ds[c].append(0)
			else:
				cter = Counter(seq)
				for c in "ACGT":
					ds[c].append(cter[c] / len(seq))
		
		x = list(range(r.start, r.stop + 1))
		return {"x": x, "densities": ds}
	else:
		raise Exception("Unknown tracktype " + tracktype)

	#return {"tracktype":tracktype, "datatype":datatype, "properties":properties}


	#======================== Unknown ========================#
	


def _plot_track_20240601(r, idata, parsed_data, properties, formatter, ax):
	'''
	Plot the track at the target region according to the initaited data, parsed data and the track properties
	'''
	from biodataplot.utils import _plt_register_asym_scale_20230101 
	from biodataplot.common import _plot_fast_bar_20240601
	
	tracktype, datatype = [idata[k] for k in ["tracktype", "datatype"]]
	
	if tracktype == "signal":
		x, y, density_x, density_y = [parsed_data[k] for k in ["x", "y", "density_x", "density_y"]]
		if "trackstyle" in formatter:
			trackstyle = formatter["trackstyle"]
		else:
			trackstyle = "bar"
			
		if trackstyle == "bar":
			custom_kw = {"width":1}
			pcolor = formatter["pcolor"] if "pcolor" in formatter else (formatter["color"] if "color" in "formatter" else "#e31a1c")
			ncolor = formatter["ncolor"] if "ncolor" in formatter else (formatter["color"] if "color" in "formatter" else "#4A4AFF")
			
			custom_kw_pos = {**custom_kw}
			custom_kw_neg = {**custom_kw}
			custom_density_kw_pos = {**custom_kw}
			custom_density_kw_neg = {**custom_kw}
			
			custom_kw_pos["fill_kw"] = {"color": pcolor}
			custom_kw_neg["fill_kw"] = {"color": ncolor}
			custom_density_kw_pos["fill_kw"] = {"alpha":0.1,"color": pcolor}
			custom_density_kw_neg["fill_kw"] = {"alpha":0.1,"color": ncolor}
			if "plot_kw" in formatter:
				custom_kw_pos.update(formatter["plot_kw"])
				custom_kw_neg.update(formatter["plot_kw"])
				custom_density_kw_pos.update(formatter["plot_kw"])
				custom_density_kw_neg.update(formatter["plot_kw"])
				
			xpos, ypos = safe_inverse_zip([(px, py) for px, py in zip(x, y) if py >= 0], 2)
			xneg, yneg = safe_inverse_zip([(px, py) for px, py in zip(x, y) if py < 0], 2)
			_plot_fast_bar_20240601(xpos, ypos, ax=ax, **custom_kw_pos)
			_plot_fast_bar_20240601(xneg, yneg, ax=ax, **custom_kw_neg)
			
			
			if "density" in formatter:
				ymin, ymax, density_ymin, density_ymax = [properties[k] for k in ["ymin", "ymax", "density_ymin", "density_ymax"]]
				xpos, ypos = safe_inverse_zip([(px, py) for px, py in zip(density_x, density_y) if py > 0], 2)
				xneg, yneg = safe_inverse_zip([(px, py) for px, py in zip(density_x, density_y) if py < 0], 2)
				if density_ymax != 0 and density_ymin != 0:
					factor = max([abs(ymin), abs(ymax)]) / max([abs(density_ymin), abs(density_ymax)])
				ypos = [factor * i for i in ypos]
				yneg = [factor * i for i in yneg]
				_plot_fast_bar_20240601(xpos, ypos, ax=ax, **custom_density_kw_pos)
				_plot_fast_bar_20240601(xneg, yneg, ax=ax, **custom_density_kw_neg)
		elif trackstyle == "heatmap":
			d = dict(zip(x, y))
			imshowx = np.arange(r.ostart, r.ostop+1)
			imshowy = np.array([d[i] if i in d else 0 for i in imshowx])
			custom_plot_kw = dict(cmap="Reds", aspect="auto",extent=[imshowx[0]-.5, imshowx[-1]+.5,0,1])
			if "plot_kw" in formatter:
				custom_plot_kw = {**custom_plot_kw, **formatter["plot_kw"]}
			ax.imshow(imshowy[np.newaxis,:], **custom_plot_kw)
		else:
			raise Exception(f"trackstyle '{trackstyle}' is not available in tracktype '{tracktype}'")
	elif tracktype == "arc":
		hits, = [parsed_data[k] for k in ["hits"]]
		rmax, = [properties[k] for k in ["rmax"]]	
		for hit in hits:
			cter = hit.dataValue
			for d, cnt in cter.items():
				x = hit.genomic_pos.start + d / 2 
				plot_kw = {}
				arc = matplotlib.patches.Arc((x, 0), d, d*2, theta2=180, linewidth=cnt / rmax, **plot_kw)
				ax.add_patch(arc)
	elif tracktype == "annotation":			
		regions_list, = [parsed_data[k] for k in ["regions_list"]]
		if "anno_height" in formatter:
			annotation_height = formatter["anno_height"]
		else:
			annotation_height = 1	
		if "anno_vspace" in formatter:
			annotation_vspace = formatter["anno_vspace"]
		else:
			annotation_vspace = 1	
		custom_plot_kw = {"facecolor":"#e5f5e0"}
		if "plot_kw" in formatter:
			custom_plot_kw = {**formatter["plot_kw"]}
		custom_strand_plot_kw = {"facecolor":"#a1d99b"}
		if "strand_plot_kw" in formatter:
			custom_strand_plot_kw = {**formatter["strand_plot_kw"]}
		custom_text_kw = {"ha":"center", "va":"top",}
		if "anno_text_kw" in formatter:
			custom_text_kw = {**formatter["anno_text_kw"]}
		for y in range(len(regions_list)):
			# Annotate
			for ar in regions_list[y]:
				# Determine the annotation position
				start = max(ar.genomic_pos.start, r.start)
				stop = min(ar.genomic_pos.stop, r.stop)
				arlen = stop - start + 1
				p = (start + stop) / 2
				# Determine the annotation text
				if "anno_name" in formatter:
					anno_str = formatter["anno_name"](ar)
				elif datatype in ["bed", "GenomicCollection-bed"]:
					if ar.name is not None and ar.name != "":
						anno_str = ar.name
					else:
						anno_str = ""
					pass
				elif datatype in ["gff3", "GenomicCollection-gff3", "gtf", "GenomicCollection-gff3"]:
					if "gene_name" in ar.attribute:
						anno_str = ar.attribute["gene_name"]# + ", " + ar.attribute["gene_type"]						start = 
					else:
						anno_str = ""
				else:
					anno_str = ""
				# Draw Text	
				if anno_str != "":
					ax.text(p, y * (annotation_height + annotation_vspace) + annotation_vspace, anno_str, **custom_text_kw)
				
				# Draw Anno
				ax.add_patch(matplotlib.patches.Rectangle((start, y * (annotation_height + annotation_vspace) + annotation_vspace), arlen, annotation_height,**custom_plot_kw))
				if hasattr(ar, "strand"):
					w = len(r.genomic_pos) / 50
					n = int(arlen / w)
					for i in range(n):
						if ar.strand == "+":
							ax.add_patch(matplotlib.patches.Polygon(
								[
									[start + i * w, y * (annotation_height + annotation_vspace) + annotation_vspace],
									[start + i * w, y * (annotation_height + annotation_vspace) + annotation_height + annotation_vspace],
									[start + (i+1) * w, y * (annotation_height + annotation_vspace) + annotation_height/2 + annotation_vspace]
								], 
							**custom_strand_plot_kw))
						elif ar.strand == "-":
							ax.add_patch(matplotlib.patches.Polygon(
								[
									[start + (i+1) * w, y * (annotation_height + annotation_vspace) + annotation_vspace],
									[start + (i+1) * w, y * (annotation_height + annotation_vspace) + annotation_height + annotation_vspace],
									[start + i * w, y * (annotation_height + annotation_vspace) + annotation_height/2 + annotation_vspace]
								], 
	
							**custom_strand_plot_kw))
	elif tracktype == "nucleotide_density":			
		x, densities = [parsed_data[k] for k in ["x", "densities"]]
		plot_kw_dict = formatter["plot_kw_dict"] if "plot_kw_dict" in formatter else {}		
		for k, density in densities.items():
			ax.plot(x, density, **(plot_kw_dict[k] if k in plot_kw_dict else {}), label=k)
		ax.set_xticks([])
		ax.set_yticks([])
	else:
		raise Exception()
		
	# ax common	 settings
	ax.margins(x=0, y=0)
	ax.spines['top'].set_color('none')
	ax.spines['right'].set_color("none")
	ax.spines['left'].set_position(("outward", 10))
	ax.tick_params(axis='x', which='both',length=0)
	# ax specific settings
	if tracktype == "signal" or tracktype == "arc":
		ymin, ymax = [properties[k] for k in ["ymin", "ymax"]]
		if "fixed_ymin_ymax" in properties:	
			ymin, ymax = properties["fixed_ymin_ymax"]		
		
		if "yscale" in properties:
			yscale = properties["yscale"]
			if yscale == "asym_pos_neg":
				if ymin >= 0:
					if "fixed_ymin_ymax" in properties:	
						raise Exception("Cannot use asym_pos_neg with ymin fixed to non negative")
					ymin = -1
				if ymax <= 0:
					if "fixed_ymin_ymax" in properties:	
						raise Exception("Cannot use asym_pos_neg with ymax fixed to non positive")
					ymax = 1
				_plt_register_asym_scale_20230101()
				ax.set_yscale("asym", a = ymax / abs(ymin))
			elif yscale == "same_ymin_ymax":
				if abs(ymin) == 0 and abs(ymax) == 0:
					if "fixed_ymin_ymax" in properties:
						raise Exception("Cannot use same_ymin_ymax with ymin / ymax fixed to zero")
					ymin = -1
					ymax = 1
				unsigned_ymax = max([abs(ymin), ymax])
				if "fixed_ymin_ymax" in properties and ymin != -unsigned_ymax and ymax != unsigned_ymax:
					raise Exception("Cannot use same_ymin_ymax with fixed_ymin_ymax at different values")
				ymin = -unsigned_ymax
				ymax = unsigned_ymax
			else:
				raise Exception("Unknown yscale")
		ax.set_ylim(ymin, ymax)
		
		yticks = [ymin, ymax]
		if 0 not in yticks:
			yticks = sorted([*yticks, 0])
		ax.set_yticks(yticks)
		ax.spines['bottom'].set_position('zero')		
	elif tracktype == "annotation":
		rmax, = [properties[k] for k in ["rmax"]]
		ax.set_ylim(0, rmax * (annotation_height + annotation_vspace))
		ax.set_yticks([])
		ax.spines['bottom'].set_color("none")
	elif tracktype == "nucleotide_density":
		ymin, ymax = [properties[k] for k in ["ymin", "ymax"]]
		if ymin == 0 and ymax == 0:
			ymin = 0
			ymax = 1
		ax.set_ylim(ymin, ymax)
		ax.set_yticks([ymin, ymax])
		ax.spines['bottom'].set_color("none")
	else:
		raise Exception()
		

def _plot_grouped_tracks_20240601(regions, data, formatters, group_autoscales, axs):
	# Auto fill up group auto scales
	def _fill_grouped_keys(grouped_keys, all_keys):
		used_keys = set([k for keys in grouped_keys for k in keys])
		to_add_grouped_keys = [[k] for k in all_keys if k not in used_keys]
		return grouped_keys + to_add_grouped_keys
	group_autoscales = _fill_grouped_keys(group_autoscales, list(data.keys()))
	
	# Auto fil formatters
	formatters = {k:formatters[k] if k in formatters else {} for k in data.keys()}
	
	# Load data
	initiated_data_dict = {k:_initiate_data_20240601(arg, formatters[k]) for k, arg in data.items()}
	# Process data according to the regions
	preprocessed_data = {k:[_preprocess_track_data_20240601(idata, r, formatters[k]) for r in regions] for k, idata in initiated_data_dict.items()}
	# Determine common properties across groups
	properties_dict = {}
	for group in group_autoscales:
		tracktypes = [initiated_data_dict[key]["tracktype"] for key in group]
		if len(set(tracktypes)) != 1:
			raise Exception("You can only group autoscale the same tracktype!")
		tracktype = tracktypes[0]
		if tracktype == "signal":
			y = []
			for key in group: 
				for parsed_data in preprocessed_data[key]:
					y.extend(parsed_data["y"])
			ymin = min(y + [0])
			ymax = max(y + [0])
			density_y = []
			for key in group: 
				for parsed_data in preprocessed_data[key]:
					if parsed_data["density_y"] is not None:
						density_y.extend(parsed_data["density_y"])
			density_ymin = min(density_y + [0])
			density_ymax = max(density_y + [0])
			properties = {"ymin":ymin, "ymax":ymax, "density_ymin":density_ymin, "density_ymax":density_ymax}
		elif tracktype == "arc":
			hits = []
			for key in group: 
				for parsed_data in preprocessed_data[key]:
					hits.extend(parsed_data["hits"])
		
			# determine ymin, ymax (distances)
			distances = [k for hit in hits for k in hit.dataValue.keys()]
			if len(distances) > 0:
				ymax = max(distances)
				if ymax < 0:
					ymax = 0
				ymin = min(distances)
				if ymin > 0:
					ymin = 0
			else:
				ymax = 0
				ymin = 0
				
			# determine line width (largest counts)
			if len(hits) > 0:
				rmax = max(v for hit in hits for v in hit.dataValue.values())
			else:
				rmax = 0
			properties = {"ymin":ymin, "ymax":ymax, "rmax":rmax}
		elif tracktype == "annotation":
			rmax = 0 #max_overlapped_regions
			for key in group: 
				for parsed_data in preprocessed_data[key]:
					if len(parsed_data["regions_list"]) > rmax:
						rmax = len(parsed_data["regions_list"])
			properties = {"rmax":rmax}
		elif tracktype == "nucleotide_density":
			flatten_density = []
			for key in group: 
				for parsed_data in preprocessed_data[key]:
					for density in parsed_data["densities"].values():
						flatten_density.extend(density)
			ymax = max(flatten_density)
			ymin = min(flatten_density)
			properties = {"ymin":ymin, "ymax":ymax}
		else:
			raise Exception()
		
		fixed_ymin_ymaxs = [formatters[key]["fixed_ymin_ymax"] for key in group if "fixed_ymin_ymax" in formatters[key]]
		if len(fixed_ymin_ymaxs) > 0:
			if len(set([tuple(i) for i in fixed_ymin_ymaxs])) > 1:
				raise Exception("You cannot provide different fixed ymin ymax in group autoscales")
			fixed_ymin_ymax = fixed_ymin_ymaxs[0]
			properties["fixed_ymin_ymax"] = fixed_ymin_ymax
		yscales = [formatters[key]["yscale"] for key in group if "yscale" in formatters[key]]
		if len(yscales) > 0:
			if len(set(yscales)) > 1:
				raise Exception("You cannot provide different yscales in group autoscales")
			yscale = yscales[0]
			properties["yscale"] = yscale
		for key in group:
			properties_dict[key] = properties
			
	# Plot tracks		
	for didx, key in enumerate(data.keys()):
		for ridx in range(len(regions)):
			ax = axs[didx, ridx]
			_plot_track_20240601(regions[ridx], initiated_data_dict[key], preprocessed_data[key][ridx], properties_dict[key], formatters[key], ax)
			if ridx == len(regions) - 1 and initiated_data_dict[key]["tracktype"] == "nucleotide_density":
				ax.legend(loc="center left", bbox_to_anchor=[1, 0.5])
			if ridx == 0:
				ax.set_ylabel(key, rotation=0, ha="right", va="center")	
			
	# Close any readers initiated when reading the data
	for idata in initiated_data_dict.values():
		for reader in idata["readers"]:
			reader.close()

	
def _plot_coordinatebar_20240601(r, major_coordinate_unit=None, minor_coordinate_unit=None, label_kw={}, ax=None):
	if ax is None:
		ax = plt.gca()	
	ax.set_xlabel(f"Genomic Position ({r.genomic_pos.name})", **label_kw)
	if major_coordinate_unit is not None:
		coordinate_unit = major_coordinate_unit
		ax.set_xticks([i for i in range((r.genomic_pos.start + coordinate_unit - 1) // coordinate_unit * coordinate_unit, r.genomic_pos.stop // coordinate_unit * coordinate_unit + 1, coordinate_unit)])
	if minor_coordinate_unit is not None:
		coordinate_unit = minor_coordinate_unit
		ax.set_xticks([i for i in range((r.genomic_pos.start + coordinate_unit - 1) // coordinate_unit * coordinate_unit, r.genomic_pos.stop // coordinate_unit * coordinate_unit + 1, coordinate_unit)], minor=True)
	ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: f"{x:.0f}"))
	ax.spines['top'].set_color('none')
	ax.spines['bottom'].set_position("zero")
	ax.spines['right'].set_color("none")
	ax.spines['left'].set_color("none")
	ax.set_ylim(-1, 1)
	ax.set_yticks([])

def _plot_scalebar_20240601(r, unit_length, loc="left", text_kw={}, plot_kw={}, ax=None):
	plot_kw={"color":"black", **plot_kw}
	text_kw={"ha":"center", "va":"top", **text_kw}
	if loc == "left":
		x1 = r.genomic_pos.start
		x2 = r.genomic_pos.start + unit_length
	elif loc == "right":
		x1 = r.genomic_pos.stop - unit_length
		x2 = r.genomic_pos.stop
	ax.plot([x1, x1, x2, x2], 
			[0.2, 0, 0, 0.2], 
			**plot_kw)
	if unit_length < 1000:
		s = f"{unit_length} bp"
	elif unit_length < 1000000:
		x = unit_length // 1000 if unit_length % 1000 == 0 else unit_length / 1000
		s = f"{x} kb"
	else:
		x = unit_length // 1000000 if unit_length % 1000000 == 0 else unit_length / 1000000
		s = f"{x} Mb"
	ax.text((x1 + x2) / 2, -0.2, s, **text_kw)
	ax.set_ylim(-1, 1)  
	ax.spines['top'].set_color('none')
	ax.spines['bottom'].set_color("none")
	ax.spines['right'].set_color("none")
	ax.spines['left'].set_color("none")
	ax.set_xticks([])
	ax.set_yticks([])

@vc
def _plot_genome_view_20240601(
		rs, 
		data={},
		formatters={}, 
		height_ratios={}, 
		group_autoscales=[],
		scalebar_kw=None,
		coordinate_kw={},
		subplots_kw={},	axs=None):
	'''
	Plot genome view for data tracks and other default tracks
	
	Supported data tracks format: bigwig, GenomicCollection
	Supported default tracks: neculeotide density, coordinate
	
	rs: a single region (GenomicAnnotation or string), or a list of regions
	data: a dictionary of data
	formatters: a dictionary or list of formats to use to plot the data track
	height_ratios: a dictionary of track heights
	group_autoscales: a list of groups. y axes of data track within the same group are shared
	subplots_kw
	
	'''
	from biodataplot.utils import _plt_share_x_axes_20221007
	# group_autoscales is a list of list
	if isinstance(rs, GenomicAnnotation) or isinstance(rs, str):
		rs = [rs]
	rs = [GenomicPos(r) for r in rs]
	if len(rs) > 32:
		raise Exception("Are you sure you want to have so many subplots along the x-axis?")
	extra_height_ratios = []
	if scalebar_kw is not None:
		extra_height_ratios.append(1)
	if coordinate_kw is not None:
		extra_height_ratios.append(1)		
	if axs is None:
		hrs = [height_ratios[key] if key in height_ratios else 3 for key in data]
		fig, full_axs = plt.subplots(len(data)+len(extra_height_ratios), len(rs), gridspec_kw=dict(height_ratios=hrs + extra_height_ratios), squeeze=False, **subplots_kw)
	else:
		full_axs = axs
		fig = full_axs[0][0].figure
		
	# All axes should share the same x-axis
	_plt_share_x_axes_20221007(full_axs)

	# Plot tracks		
	_plot_grouped_tracks_20240601(rs, data, formatters, group_autoscales, full_axs)
	
	# Plot other tracks (scale bar or coordinate bar)
	for ridx, r in enumerate(rs):
		full_axs[0, ridx].set_xlim(r.start, r.stop)
		ax_idx = 1
		if coordinate_kw is not None:
			_plot_coordinatebar_20240601(r,**coordinate_kw, ax=full_axs[-ax_idx, ridx])
			ax_idx += 1
		if scalebar_kw is not None:
			_plot_scalebar_20240601(r, **scalebar_kw, ax=full_axs[-ax_idx, ridx])
			ax_idx += 1
	
	# Add title
	for ridx, r in enumerate(rs):
		full_axs[0, ridx].set_title(str(r))
	# Final style changes on yticklabels
	for column in range(len(rs)):
		for ax in full_axs[:-1, column]:
			if len(ax.get_yticklabels()) >= 2:
				ax.get_yticklabels()[-1].set_va("top")
				ax.get_yticklabels()[0].set_va("bottom")
				ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: f"{round(x)}"  if round(x) == x else f"{abs(x):.2f}" ))
	fig.align_ylabels()

	return fig

