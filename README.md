# biodataplot - A standard biological data plot package

`biodataplot` is a package that provides various useful biological plotting functions using matplotlib. 

## Installation 

```
pip install biodataplot
```

## biodataplot

`biodataplot` contains different modules that serve different purposes. The `common` module includes general plotting functions (not biology-related). The `utils` module includes some plot utility functions. Other modules are related to certain biological area. 

In biodataplot, the plot functions could be separated into two types. The first type works on a matplotlib axes Object and return the artists. The second type works on a figure and return the matplotlib figure itself. Some functions may require additional packages. 

### common

A module with general plotting functions.  

##### plot_fast_bar

An alternative bar plotting using `fill_between`. This allows quick plot on many bars (10000+) but with less customization available. 

#### plot_ranked_values



### genomeview

The genomeview contains the `plot_genome_view` function, which draws browser snapshot on certain regions. 

```python
import biodataplot.genomeview as gv
gv.plot_genome_view(r, data)
gv.plot_genome_view(r, data, formatters=formatters, height_ratios=height_ratios)

```


- `rs`: Region(s) to be plotted. It could be a single string `chr1:1-1000`, a `GenomicAnnotation` instance, or a list with string or `GenomicAnnotation` as elements. 
- `data`: A dictionary of data to be plotted
- `formatters`: A dictionary of formatter. Other than datatype and tracktype, formatters are unique for different track types. On could find more details in the track section.
- `height_ratios`: A dictionary of relative track heights. The default height of track is 3. 
- `group_autoscales`: A list of list that indicates the tracks to be put in groups. 
- `scalebar_kw`: If None, no scalebar is drawn. 
- `coordinate_kw`: If None, no coordinate ruler is drawn



#### Data 

Users could input the data as a dictionary of file(s)

```
data={"Signal":"signal.bw", "Anno1":"anno1.bed"}
```

All supported file / data types:

- bigwig (`signal`)
- bedgraph (`signal, arc, annotation`, default to `annotation` if not specified)
- bed (`annotation`)
- gff3 / gtf (`annotation`)
- fasta (`nucleotide_density`)

`biodataplot` will automatically look for any index file whenever applicable. Without an index file, it may take a long time for `biodataplot` to load big data files. For certain data types, multiple track types could be used. In such case, one may want to specify the tracktype in the formatters:

```
data={"Signal":"signal.bedgraph"}
formatters={"Signal":{"tracktype":"signal"}}
```

#### Tracks

There are 4 types of tracks supported. Each track has its specific options used in formatters.

##### Signal

In signal plotting, either bigwig or bedgraph files are supported. 

- `vmod`: Modify the values in the signal before plotting. 
- `density`: If specified, plot a smoothed signals on the same graph. `{"winsize":100}`
- `trackstyle`: Two track styles are support: either `bar` or `heatmap`
- `trackstyle: bar`
  - `pcolor`: Color for positive signals 
  - `ncolor`: Color for negative signals 
  - `plot_kw`: Customized parameters used in `plot_fast_bar`
- `trackstyle: heatmap`
  - `plot_kw`: Customized parameters used in `imshow`
- `fixed_ymin_ymax`: Fix `ymin`, `ymax` as the input
- `yscale`:  Either `asym_pos_neg` or `same_ymin_ymax`. These alternative yscales are useful when dealing with both positive and negative signals. 

##### Arc

- `min_signal`: Minimum signal cutoff to draw the arc

##### Annotation

- `filter_func`: A filtering function to retain annotations that pass this filter
- `anno_height`: Annotation height
- `anno_vspace`: Vertical space between two annotations
- `plot_kw`: Customized parameters used in `matplotlib.patches.Rectangle`
- `anno_name`: A function `f(anno)` that returns a string to display. Otherwise, the `anno_name` is automatically determined. 

##### Nucleotide_density

- `winsize` The window size for nucleotide density. Only `ACGT` are supported in the density plot. All other nucleotides are ignored. 
- `plot_kw_dict`: A dictionary of customized parameters used in `plot`. The dictionary keys should be in `[A, C, G, T]`. 



### metaplot

This module is under testing. 

### sequence

#### plot_sequence_layout



### utils

A module with useful plot utilities.





