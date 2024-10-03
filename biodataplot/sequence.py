import sys
import simplevc
simplevc.register(sys.modules[__name__])

import importlib.util

@vc
def _plot_fastq_layout_20230105(i, n=10000, title=None):
	if any(importlib.util.find_spec(p) is None for p in ["pandas", "logomaker"]):
		raise Exception("pandas and logomaker are required.")
		
	import itertools
	import os
	from collections import Counter
	import numpy as np
	import pandas as pd
	import logomaker
	from biodata.fasta import FASTQReader
	
	if title is None:
		title = os.path.basename(i)
	with FASTQReader(i) as fr:
		if n != -1:
			seqs = [fa.seq for fa in itertools.islice(fr, n)]
		else:
			seqs = [fa.seq for fa in fr]
	n = len(seqs)
	seqlen = max(len(seq) for seq in seqs)
	heights = []
	e = 1/np.log(2) * (4 - 1) / 2 / n
	for p in range(seqlen):
		cter = Counter(seq[p] for seq in seqs if len(seq) > p)
		total = sum(cter.values())
		h = sum([-1 * cter[c] / total * np.log2(cter[c] / total) if cter[c] > 0 else 0 for c in "ACGT"])
		r = np.log2(4) - (h + e)
		heights.append([cter[c] / total * r for c in "ACGT"])
	df = pd.DataFrame(heights, columns=list("ACGT"))
	logo = logomaker.Logo(df)
	logo.fig.set_size_inches(seqlen / 10, 1.75)
	logo.ax.set_ylabel("bits")
	logo.ax.set_yticks([0, 1, 2])
	logo.ax.set_xlabel("position")
	logo.ax.set_title(title, fontsize=16)
	logo.fig.tight_layout()
	return logo.fig