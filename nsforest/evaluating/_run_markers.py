
import time
import pandas as pd
from nsforest.nsforesting import mydecisiontreeevaluation
from nsforest.nsforesting import calculate_fraction

def DecisionTree(adata, cluster_header, medians_header = "medians_", 
                 markers_dict = {}, beta = 0.5, combinations = False, use_mean = False,
                 output_folder = "", outputfilename_prefix = ""): 
    """\
    Calculating sklearn.metrics's fbeta_score, sklearn.metrics's prevision_score, sklearn.metrics's confusion_matrix for each `genes_eval` combination. 
    Returning set of genes and scores with highest score sum. 

    Parameters
    ----------
    adata: AnnData
        Annotated data matrix.
    cluster_header
        Column in `adata`'s `.obs` representing cell annotation.
    markers_dict
        Dictionary containing marker genes for cell annotations (clusterName: list of markers)
    beta
        Beta value in sklearn.metrics's fbeta_score. 
    combinations
        Whether to use myDecisionTreeEvaluation on various combinations of `genes_eval`. 
    use_mean
        Whether to use the mean or median for minimum gene expression threshold. 
    output_folder
        Output folder. 
    outputfilename_prefix
        Prefix for all output files. 
    
    Returns
    -------
    df_results: pd.DataFrame of the NS-Forest results. Contains classification metrics (f_score, PPV, recall, onTarget). 
    """

    # default medians_header
    if medians_header == "medians_": medians_header = "medians_" + cluster_header

    ##-----
    ## prepare adata
    ##-----
    print("Preparing data...")
    start_time = time.time()
    ## densify X from sparse matrix format
    adata.X = adata.to_df()
    ## categorial cluster labels
    adata.obs[cluster_header] = adata.obs[cluster_header].astype('category')
    ## dummy/indicator for one vs. all Random Forest model
    df_dummies = pd.get_dummies(adata.obs[cluster_header]) #cell-by-cluster
    print("--- %s seconds ---" % (time.time() - start_time))
    
    ############################## START iterations ######################################
    cluster_list = list(markers_dict.keys())
    n_clusters = len(cluster_list)
    
    print("Number of clusters to evaluate: " + str(n_clusters))
    df_results = pd.DataFrame()
    start_time = time.time()
    
    for cl in cluster_list[:]:
        ct = list(cluster_list).index(cl) + 1
        print(f"{ct} out of {n_clusters}:")
        print(f"\t{cl}")
        print(f"\tmarker genes to be evaluated: {markers_dict[cl]}")
        
        ##=== reset parameters for this iteration!!! (for taking care of special cases) ===##
        markers = []
        for marker in markers_dict[cl]: 
            if marker in list(adata.var_names): 
                markers.append(marker)
            else: 
                print(f"cannot find {marker} in adata.var_names, excluding from DecisionTree.")

        if len(markers) == 0: continue
        
        ## Evaluation step: calculate F-beta score for gene combinations
        markers, scores, score_max = mydecisiontreeevaluation.myDecisionTreeEvaluation(adata, df_dummies, cl, markers, beta, combinations = combinations)
        print("\t" + str(markers))
        print("\tf-beta:" + str(scores[0]))
        print("\tPPV:" + str(scores[1]))

        ## return final results as dataframe
        dict_results_cl = {'clusterName': cl,
                           'clusterSize': int(scores[4]+scores[5]),
                           'f_score': scores[0],
                           'recall': int(scores[5]) / (int(scores[5]) + int(scores[4])),
                           'PPV': scores[1],
                           'TN': int(scores[2]),
                           'FP': int(scores[3]),
                           'FN': int(scores[4]),
                           'TP': int(scores[5]),
                           'marker_count': len(markers),
                           'markers': [markers] 
                           }
        df_results_cl = pd.DataFrame(dict_results_cl)
        df_results = pd.concat([df_results,df_results_cl]).reset_index(drop=True)
        df_results.to_csv(output_folder + outputfilename_prefix + "_results.csv", index=False)

    markers_dict = dict(zip(df_results["clusterName"], df_results["markers"]))
    on_target_ratio = calculate_fraction.markers_onTarget(adata, markers_dict, cluster_header, medians_header, use_mean = use_mean, output_folder = output_folder, outputfilename_prefix = outputfilename_prefix)
    df_results = df_results.merge(on_target_ratio, on = "clusterName", how = "left")
    df_results.to_csv(f"{output_folder}{outputfilename_prefix}_results.csv", index=False)
    print(f"Saving final results table as...\n{output_folder}{outputfilename_prefix}_results.csv")
    print("--- %s seconds ---" % (time.time() - start_time))
    ### END iterations ###
    
    return df_results

def add_fraction(adata, df_results, cluster_header, medians_header = "medians_", use_mean = False, output_folder = "", outputfilename_prefix = ""): 
    """\
    Calculating sklearn.metrics's fbeta_score, sklearn.metrics's prevision_score, sklearn.metrics's confusion_matrix for each `genes_eval` combination. 
    Returning set of genes and scores with highest score sum. 

    Parameters
    ----------
    adata: AnnData
        Annotated data matrix.
    df_results: pd.DataFrame
        NS-Forest results. Contains classification metrics (f_score, PPV, recall, onTarget). 
    cluster_header
        Column in `adata`'s `.obs` representing cell annotation.
    medians_header: str
        Key in `adata`'s `.varm` storing median expression matrix. 
    use_mean
        Whether to use the mean or median for minimum gene expression threshold.
    output_folder
        Output folder. 
    outputfilename_prefix
        Prefix for all output files. 
    
    Returns
    -------
    df_results: pd.DataFrame of the NS-Forest results. Contains classification metrics (f_score, PPV, recall, onTarget). 
    """

    # default medians_header
    if medians_header == "medians_": medians_header = "medians_" + cluster_header

    markers_dict = dict(zip(df_results["clusterName"], df_results["markers"]))
    on_target_ratio = calculate_fraction.markers_onTarget(adata, markers_dict, cluster_header, medians_header, use_mean, output_folder, outputfilename_prefix)
    if "fraction" in list(df_results.columns): del df_results["fraction"]
    if "onTarget" in list(df_results.columns): del df_results["onTarget"]
    df_results = df_results.merge(on_target_ratio, on = "clusterName", how = "left")
    df_results.to_csv(f"{output_folder}{outputfilename_prefix}_results.csv", index=False)
    print(f"Saving final results table as...\n{output_folder}{outputfilename_prefix}_results.csv")
    return df_results
