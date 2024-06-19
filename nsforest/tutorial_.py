#!/usr/bin/env python
# coding: utf-8

# # Tutorial

# ## Tutorial

# ### Setting up environment

# In[1]:


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


# ### Data Exploration

# #### Loading h5ad AnnData file

# In[2]:


data_folder = "../demo_data/"
file = data_folder + "adata_layer1.h5ad"
adata = sc.read_h5ad(file)
adata


# #### Defining `cluster_header` as cell type annotation. 
# 
# **Note:** Some datasets have multiple annotations per sample (ex. "broad_cell_type" and "granular_cell_type"). NS-Forest can be run on multiple `cluster_header`'s. Combining the parent and child markers may improve classification results. 

# In[3]:


cluster_header = "cluster"


# #### Defining `output_folder` for saving results

# In[4]:


output_folder = "../outputs_layer1/"


# #### Looking at sample labels

# In[5]:


adata.obs_names


# #### Looking at genes
# 
# **Note:** `adata.var_names` must be unique. If there is a problem, usually it can be solved by assigning `adata.var.index = adata.var["ensembl_id"]`. 

# In[6]:


adata.var_names


# #### Checking cell annotation sizes 
# 
# **Note:** Some datasets are too large and need to be downsampled to be run through the pipeline. When downsampling, be sure to have all the granular cluster annotations represented. 

# In[7]:


adata.obs[cluster_header].value_counts()


# ### Preprocessing

# #### Generating scanpy dendrogram
# 
# **Note:** Only run if there is no pre-defined dendrogram order. This step can still be run with no effects, but the runtime may increase. 
# 
# Dendrogram order is stored in `adata.uns["dendrogram_cluster"]["categories_ordered"]`. 
# 
# This dataset has a pre-defined dendrogram order, so running this step is not necessary. 

# In[8]:


ns.pp.dendrogram(adata, cluster_header, save = True, output_folder = output_folder, outputfilename_suffix = cluster_header)


# #### Calculating cluster medians per gene
# 
# Run `ns.pp.prep_medians` before running NS-Forest.
# 
# **Note:** Do **not** run if evaluating marker lists. Do **not** run when generating scanpy plots (e.g. dot plot, violin plot, matrix plot). 

# In[9]:


adata = ns.pp.prep_medians(adata, cluster_header)
adata


# #### Calculating binary scores per gene per cluster
# 
# Run `ns.pp.prep_binary_scores` before running NS-Forest. Do not need to run if evaluating marker lists. Do not need to run when generating scanpy plots. 

# In[10]:


adata = ns.pp.prep_binary_scores(adata, cluster_header)
adata


# #### Plotting median and binary score distributions

# In[11]:


plt.clf()
filename = output_folder + cluster_header + '_medians.png'
print(f"Saving median distributions as...\n{filename}")
a = plt.figure(figsize = (6, 4))
a = plt.hist(adata.varm["medians_" + cluster_header].unstack(), bins = 100)
a = plt.title(f'{file.split("/")[-1].replace(".h5ad", "")}: {"medians_" + cluster_header} histogram')
a = plt.xlabel("medians_" + cluster_header)
a = plt.yscale("log")
a = plt.savefig(filename, bbox_inches='tight')
plt.show()


# In[12]:


plt.clf()
filename = output_folder + cluster_header + '_binary_scores.png'
print(f"Saving binary_score distributions as...\n{filename}")
a = plt.figure(figsize = (6, 4))
a = plt.hist(adata.varm["binary_scores_" + cluster_header].unstack(), bins = 100)
a = plt.title(f'{file.split("/")[-1].replace(".h5ad", "")}: {"binary_scores_" + cluster_header} histogram')
a = plt.xlabel("binary_scores_" + cluster_header)
a = plt.yscale("log")
a = plt.savefig(filename, bbox_inches='tight')
plt.show()


# #### Saving preprocessed AnnData as new h5ad

# In[13]:


filename = file.replace(".h5ad", "_preprocessed.h5ad")
print(f"Saving new anndata object as...\n{filename}")
adata.write_h5ad(filename)


# ### Running NS-Forest
# 
# **Note:** Do not run NS-Forest if only evaluating input marker lists. 

# In[14]:


outputfilename_prefix = cluster_header
results = nsforesting.NSForest(adata, cluster_header, output_folder = output_folder, outputfilename_prefix = outputfilename_prefix)


# In[15]:


results


# #### Plotting classification metrics from NS-Forest results

# In[16]:


ns.pl.boxplot(results, "f_score")


# In[17]:


ns.pl.boxplot(results, "PPV")


# In[18]:


ns.pl.boxplot(results, "recall")


# In[19]:


ns.pl.boxplot(results, "onTarget")


# In[20]:


ns.pl.scatter_w_clusterSize(results, "f_score")


# In[21]:


ns.pl.scatter_w_clusterSize(results, "PPV")


# In[22]:


ns.pl.scatter_w_clusterSize(results, "recall")


# In[23]:


ns.pl.scatter_w_clusterSize(results, "onTarget")


# ### Plotting scanpy dot plot, violin plot, matrix plot for NS-Forest markers
# 
# **Note:** Assign pre-defined dendrogram order here **or** use `adata.uns["dendrogram_" + cluster_header]["categories_ordered"]`. 

# In[24]:


to_plot = results.copy()


# In[25]:


dendrogram = [] # custom dendrogram order
dendrogram = list(adata.uns["dendrogram_" + cluster_header]["categories_ordered"])
to_plot["clusterName"] = to_plot["clusterName"].astype("category")
to_plot["clusterName"] = to_plot["clusterName"].cat.set_categories(dendrogram)
to_plot = to_plot.sort_values("clusterName")
to_plot = to_plot.rename(columns = {"NSForest_markers": "markers"})
to_plot.head()


# In[26]:


markers_dict = dict(zip(to_plot["clusterName"], to_plot["markers"]))
markers_dict


# In[27]:


ns.pl.dotplot(adata, markers_dict, cluster_header, dendrogram = dendrogram, save = True, output_folder = output_folder, outputfilename_suffix = outputfilename_prefix)


# In[28]:


ns.pl.stackedviolin(adata, markers_dict, cluster_header, dendrogram = dendrogram, save = True, output_folder = output_folder, outputfilename_suffix = outputfilename_prefix)


# In[29]:


ns.pl.matrixplot(adata, markers_dict, cluster_header, dendrogram = dendrogram, save = True, output_folder = output_folder, outputfilename_suffix = outputfilename_prefix)


# ### Evaluating input marker list

# #### Getting marker list in dictionary format: {cluster: marker_list}

# In[30]:


markers = pd.read_csv("../demo_data/marker_list.csv")
markers_dict = utils.prepare_markers(markers, "clusterName", "markers")
markers_dict


# In[31]:


outputfilename_prefix = "marker_eval"
evaluation_results = ev.DecisionTree(adata, cluster_header, markers_dict, combinations = False, use_mean = False, 
                                             output_folder = output_folder, outputfilename_prefix = outputfilename_prefix)


# In[32]:


evaluation_results


# #### Plotting classification metrics from marker evaluation

# In[33]:


ns.pl.boxplot(evaluation_results, "f_score")


# In[34]:


ns.pl.boxplot(evaluation_results, "PPV")


# In[35]:


ns.pl.boxplot(evaluation_results, "recall")


# In[36]:


ns.pl.boxplot(evaluation_results, "onTarget")


# In[37]:


ns.pl.scatter_w_clusterSize(evaluation_results, "f_score")


# In[38]:


ns.pl.scatter_w_clusterSize(evaluation_results, "PPV")


# In[39]:


ns.pl.scatter_w_clusterSize(evaluation_results, "recall")


# In[40]:


ns.pl.scatter_w_clusterSize(evaluation_results, "onTarget")


# ### Plotting scanpy dot plot, violin plot, matrix plot for input marker list
# 
# **Note:** Assign pre-defined dendrogram order here **or** use `adata.uns["dendrogram_" + cluster_header]["categories_ordered"]`. 

# In[41]:


to_plot = evaluation_results.copy()


# In[42]:


dendrogram = [] # custom dendrogram order
dendrogram = list(adata.uns["dendrogram_" + cluster_header]["categories_ordered"])
to_plot["clusterName"] = to_plot["clusterName"].astype("category")
to_plot["clusterName"] = to_plot["clusterName"].cat.set_categories(dendrogram)
to_plot = to_plot.sort_values("clusterName")
to_plot = to_plot.rename(columns = {"NSForest_markers": "markers"})
to_plot.head()


# In[43]:


markers_dict = dict(zip(to_plot["clusterName"], to_plot["markers"]))
markers_dict


# In[44]:


ns.pl.dotplot(adata, markers_dict, cluster_header, dendrogram = dendrogram, save = True, output_folder = output_folder, outputfilename_suffix = outputfilename_prefix)


# In[46]:


ns.pl.stackedviolin(adata, markers_dict, cluster_header, dendrogram = dendrogram, save = True, output_folder = output_folder, outputfilename_suffix = outputfilename_prefix)


# In[47]:


ns.pl.matrixplot(adata, markers_dict, cluster_header, dendrogram = dendrogram, save = True, output_folder = output_folder, outputfilename_suffix = outputfilename_prefix)

