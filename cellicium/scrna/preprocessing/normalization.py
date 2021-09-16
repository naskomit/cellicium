import numpy as np
import scanpy as sc

def log_normalize(adata : sc.AnnData, target_sum : float = 1e4, exclude_highly_expressed : bool = False,
                max_fraction :float = 0.05):
    key_added = 'norm_factor'
    mods = sc.pp.normalize_total(adata, target_sum = target_sum, exclude_highly_expressed = exclude_highly_expressed,
                            max_fraction = max_fraction, key_added = key_added, inplace = False)
    #key_added = 'X_log'
    adata.layers['X_norm'] = mods['X']
    sc.logging.info('Added layer X_norm')
    adata.obs[key_added] = mods['norm_factor']
    sc.logging.info(f'Added column obs.{key_added}')

    log_data = sc.pp.log1p(mods['X'], base = 10)
    adata.layers['X_log_norm'] = sc.pp.log1p(mods['X'], base = 10)
    sc.logging.info('Added layer X_log_norm')
