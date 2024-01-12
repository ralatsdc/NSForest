
_epilog = """
Examples

python3 nsforest -f demo_data/adata_layer1.h5ad -c cluster
python3 nsforest -a arguments.csv

"""

import sys
# import os
import pandas as pd
import scanpy as sc
import utils
import nsforest

def main():

    # Reading in arguments
    args = utils._parse_args(sys.argv[1:], epilog=_epilog, description=__doc__)

    if args.a: 
        df = pd.read_csv(args.a).astype(str)
        df = df[df["val"] != "nan"]
        kwargs = dict(zip(df["argument"], df["val"]))
        file = kwargs["filename"]
        cluster_header = kwargs["cluster_header"]
        del kwargs["filename"], kwargs["cluster_header"]
        # https://www.geeksforgeeks.org/python-remove-empty-value-types-in-dictionaries-list/
        kwargs = [ele for ele in ({key: val for key, val in sub.items() if val} for sub in [kwargs]) if ele][0]
    else: 
        kwargs = {}
        if args.f: file = args.f
        if args.c: cluster_header = args.c
            
    print(file, cluster_header, kwargs)

    adata = sc.read_h5ad(file)

    nsforest.NSForest(adata, cluster_header, **kwargs)

main()