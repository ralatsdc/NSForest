#!/usr/bin/env python
# coding: utf-8

# Setting up environment
import sys
import os

sys.path.insert(0, os.path.abspath("../"))
sys.path.insert(0, os.path.abspath("../nsforest/nsforesting"))
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import nsforest as ns
from nsforest import utils
from nsforest import preprocessing as pp
from nsforest import nsforesting
from nsforest import evaluating as ev
from nsforest import plotting as pl


# === Data Exploration

# Loading h5ad AnnData file
data_folder = "../demo_data/"
file = data_folder + "adata_layer1.h5ad"
adata = sc.read_h5ad(file)

# Defining `cluster_header` as cell type annotation.
#
# Note: Some datasets have multiple annotations per sample
# (ex. "broad_cell_type" and "granular_cell_type"). NS-Forest can be
# run on multiple `cluster_header`'s. Combining the parent and child
# markers may improve classification results.
cluster_header = "cluster"

# Defining `output_folder` for saving results
output_folder = "../outputs_layer1/"

# Looking at sample labels
adata.obs_names

# Looking at genes
#
# Note: `adata.var_names` must be unique. If there is a problem,
# usually it can be solved by assigning `adata.var.index =
# adata.var["ensembl_id"]`.
adata.var_names

# Checking cell annotation sizes
#
# Note: Some datasets are too large and need to be downsampled to be
# run through the pipeline. When downsampling, be sure to have all the
# granular cluster annotations represented.
adata.obs[cluster_header].value_counts()


# === Preprocessing

# Generating scanpy dendrogram
#
# Note: Only run if there is no pre-defined dendrogram order. This
# step can still be run with no effects, but the runtime may
# increase. Dendrogram order is stored in
# `adata.uns["dendrogram_cluster"]["categories_ordered"]`.
ns.pp.dendrogram(
    adata,
    cluster_header,
    save=True,
    output_folder=output_folder,
    outputfilename_suffix=cluster_header,
)

# Calculating cluster medians per gene
#
# Note: Run `ns.pp.prep_medians` before running NS-Forest.
adata = ns.pp.prep_medians(adata, cluster_header)
adata

# Calculating binary scores per gene per cluster
#
# Note: Run `ns.pp.prep_binary_scores` before running NS-Forest.
adata = ns.pp.prep_binary_scores(adata, cluster_header)
adata

# Plotting median and binary score distributions
plt.clf()
filename = output_folder + cluster_header + "_medians.png"
print(f"Saving median distributions as...\n{filename}")
a = plt.figure(figsize=(6, 4))
a = plt.hist(adata.varm["medians_" + cluster_header].unstack(), bins=100)
a = plt.title(
    f'{file.split("/")[-1].replace(".h5ad", "")}: {"medians_" + cluster_header} histogram'
)
a = plt.xlabel("medians_" + cluster_header)
a = plt.yscale("log")
a = plt.savefig(filename, bbox_inches="tight")
plt.show()
plt.clf()
filename = output_folder + cluster_header + "_binary_scores.png"
print(f"Saving binary_score distributions as...\n{filename}")
a = plt.figure(figsize=(6, 4))
a = plt.hist(adata.varm["binary_scores_" + cluster_header].unstack(), bins=100)
a = plt.title(
    f'{file.split("/")[-1].replace(".h5ad", "")}: {"binary_scores_" + cluster_header} histogram'
)
a = plt.xlabel("binary_scores_" + cluster_header)
a = plt.yscale("log")
a = plt.savefig(filename, bbox_inches="tight")
plt.show()

# Saving preprocessed AnnData as new h5ad
filename = file.replace(".h5ad", "_preprocessed.h5ad")
print(f"Saving new anndata object as...\n{filename}")
adata.write_h5ad(filename)


# === Running NS-Forest and plotting classification metrics

# Running NS-Forest
outputfilename_prefix = cluster_header
results = nsforesting.NSForest(
    adata,
    cluster_header,
    output_folder=output_folder,
    outputfilename_prefix=outputfilename_prefix,
)
results


# Plotting classification metrics from NS-Forest results
ns.pl.boxplot(results, "f_score")
ns.pl.boxplot(results, "PPV")
ns.pl.boxplot(results, "recall")
ns.pl.boxplot(results, "onTarget")
ns.pl.scatter_w_clusterSize(results, "f_score")
ns.pl.scatter_w_clusterSize(results, "PPV")
ns.pl.scatter_w_clusterSize(results, "recall")
ns.pl.scatter_w_clusterSize(results, "onTarget")

# Plotting scanpy dot plot, violin plot, matrix plot for NS-Forest markers
#
# Note: Assign pre-defined dendrogram order here **or** use
# `adata.uns["dendrogram_" + cluster_header]["categories_ordered"]`.
to_plot = results.copy()
dendrogram = []  # custom dendrogram order
dendrogram = list(adata.uns["dendrogram_" + cluster_header]["categories_ordered"])
to_plot["clusterName"] = to_plot["clusterName"].astype("category")
to_plot["clusterName"] = to_plot["clusterName"].cat.set_categories(dendrogram)
to_plot = to_plot.sort_values("clusterName")
to_plot = to_plot.rename(columns={"NSForest_markers": "markers"})
to_plot.head()
markers_dict = dict(zip(to_plot["clusterName"], to_plot["markers"]))
markers_dict
ns.pl.dotplot(
    adata,
    markers_dict,
    cluster_header,
    dendrogram=dendrogram,
    save=True,
    output_folder=output_folder,
    outputfilename_suffix=outputfilename_prefix,
)
ns.pl.stackedviolin(
    adata,
    markers_dict,
    cluster_header,
    dendrogram=dendrogram,
    save=True,
    output_folder=output_folder,
    outputfilename_suffix=outputfilename_prefix,
)
ns.pl.matrixplot(
    adata,
    markers_dict,
    cluster_header,
    dendrogram=dendrogram,
    save=True,
    output_folder=output_folder,
    outputfilename_suffix=outputfilename_prefix,
)
