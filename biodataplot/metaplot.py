import sys
import simplevc
simplevc.register(sys.modules[__name__])

import numpy as np

from genomictools import GenomicPos, StrandedGenomicPos

def _expand_genomic_pos(r, size):
	s = size - len(r.genomic_pos)
	return GenomicPos(r.genomic_pos.name, r.genomic_pos.start - (s // 2), r.genomic_pos.stop + (s // 2 + s % 2))

@vc
def _generate_signal_profile_20240801(regions, signals, fixed_size=None, nbins=None, use_strand=False):
	'''
	By default, input regions are assumed to have a fixed length. If not, two workarounds are provided in the method.
	- Use `fixed_size` parameter to forcibly resize all regions into the target size. This is useful when you center around certain feature to look at the signal profile (e.g., promoter TSS +- n bp). 
	- Use `nbins` parameter to divide all region into n bins. This is useful when you summarize signal profile across different-size regions (e.g. gene body). For example, if `nbins` is 10, a 50-bp region will be divided into 10 bins (5-bp long), while a 80-bp region will be divided into 10 bins (8-bp long).
	
	If `use_strand` is supplied, signals are reversed for minus-strand region.
	
	'''
	
	def _rebin(data, n):
		'''
		data: a list of data
		n: number of bins
		'''
		
		bin_size = len(data) / n
		bins = [0] * n
		step = len(data) // np.gcd(len(data), n)
		for i, v in enumerate(data):
			mid = (i / bin_size) # Get the point of the original bin on the new bin (start pos)
			next_mid = ((i + 1)/ bin_size) # Get the point of the original bin on the new bin (stop pos)
			floor = int(mid)
			next_floor = int(next_mid)
			v_per_bin = v * bin_size
			if floor == next_floor: # They refer to the same new bin
				bins[floor] += v
				continue

			if i % step == 0: 
				for bin_idx in range(floor, next_floor):
					bins[bin_idx] += v_per_bin
			else:
				bins[floor] += v_per_bin * (1 - (mid - floor))
				for bin_idx in range(floor + 1, next_floor):
					bins[bin_idx] += v_per_bin
			if (i + 1) % step != 0:
				bins[next_floor] += v_per_bin * (next_mid - next_floor)
		return bins
	
	if use_strand:
		regions = [StrandedGenomicPos(r) for r in regions]
		pl = _generate_signal_profile_20240801([r for r in regions if r.strand == "+"], signals, fixed_size, nbins, use_strand=False)
		mn = _generate_signal_profile_20240801([r for r in regions if r.strand == "-"], signals, fixed_size, nbins, use_strand=False)
		return pl + [i[::-1] for i in mn]
	else:
		if len(regions) == 0:
			return []
		regions = [GenomicPos(r) for r in regions]
		if fixed_size is not None:
			regions = [_expand_genomic_pos(r, fixed_size) for r in regions]
			
		all_bins = []
		if nbins is None:
			if not len(set([len(r) for r in regions])) == 1:
				raise Exception("Inconsistent region size. Either use fixed_size or nbins to fix the problem.")
			for r in regions:
				all_bins.append(signals.values(r))
		else:
			for r in regions:
				original_bins = signals.values(r)
				if len(original_bins) == nbins:
					all_bins.append(original_bins)
				else:
					all_bins.append(_rebin(original_bins, nbins))
		return all_bins
	
@vc
def _generate_signal_profile_20241015(regions, signals, fixed_size=None, nbins=None, use_strand=False):
	'''
	By default, input regions are assumed to have a fixed length. If not, two workarounds are provided in the method.
	- Use `fixed_size` parameter to forcibly resize all regions into the target size. This is useful when you center around certain feature to look at the signal profile (e.g., promoter TSS +- n bp). 
	- Use `nbins` parameter to divide all region into n bins. This is useful when you summarize signal profile across different-size regions (e.g. gene body). For example, if `nbins` is 10, a 50-bp region will be divided into 10 bins (5-bp long), while a 80-bp region will be divided into 10 bins (8-bp long).
	
	If `use_strand` is supplied, signals are reversed for minus-strand region.
	
	'''
	
	def _rebin(data, n):
		'''
		data: a list of data
		n: number of bins
		'''
		
		bin_size = len(data) / n
		bins = [0] * n
		step = len(data) // np.gcd(len(data), n)
		for i, v in enumerate(data):
			mid = (i / bin_size) # Get the point of the original bin on the new bin (start pos)
			next_mid = ((i + 1)/ bin_size) # Get the point of the original bin on the new bin (stop pos)
			floor = int(mid)
			next_floor = int(next_mid)
			v_per_bin = v * bin_size
			if floor == next_floor: # They refer to the same new bin
				bins[floor] += v
				continue

			if i % step == 0: 
				for bin_idx in range(floor, next_floor):
					bins[bin_idx] += v_per_bin
			else:
				bins[floor] += v_per_bin * (1 - (mid - floor))
				for bin_idx in range(floor + 1, next_floor):
					bins[bin_idx] += v_per_bin
			if (i + 1) % step != 0:
				bins[next_floor] += v_per_bin * (next_mid - next_floor)
		return bins
	
	if use_strand:
		
		regions = [StrandedGenomicPos(r) for r in regions]
		plus_indice = [r.strand != "-" for r in regions] 
		pl = _generate_signal_profile_20241015([r for r in regions if r.strand != "-"], signals, fixed_size, nbins, use_strand=False)
		mn = _generate_signal_profile_20241015([r for r in regions if r.strand == "-"], signals, fixed_size, nbins, use_strand=False)
		mn = [i[::-1] for i in mn]
		plidx = 0
		mnidx = 0
		arr = []
		for is_plus in plus_indice:
			if is_plus:
				arr.append(pl[plidx])
				plidx += 1
			else:
				arr.append(mn[mnidx])
				mnidx += 1
		return arr
	else:
		if len(regions) == 0:
			return []
		regions = [GenomicPos(r) for r in regions]
		if fixed_size is not None:
			regions = [_expand_genomic_pos(r, fixed_size) for r in regions]
			
		all_bins = []
		if nbins is None:
			if not len(set([len(r) for r in regions])) == 1:
				raise Exception("Inconsistent region size. Either use fixed_size or nbins to fix the problem.")
			for r in regions:
				all_bins.append(signals.values(r))
		else:
			for r in regions:
				original_bins = signals.values(r)
				if len(original_bins) == nbins:
					all_bins.append(original_bins)
				else:
					all_bins.append(_rebin(original_bins, nbins))
		return all_bins	