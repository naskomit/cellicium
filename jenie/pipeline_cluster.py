import numpy as np
import pandas as pd
import scipy.io as sio
import os

class Pipeline():
    def __init__(self):
        self.data_path = "/data/Workspace/SysMo/IRENE/MammaryEC"
        self.dataset = "MammaryEC"

    def read_expression_data(self):
        # Read expression file
        expr_file = os.path.join(self.data_path, f"{self.dataset}_rawExp.tsv")
        expr_data = pd.read_csv(expr_file, sep='\t')
        print(f"Loaded raw expression data: {expr_data.shape}")
        expr_data = expr_data.iloc[:, [0, 5]]
        expr_data.iloc[:, 0] = expr_data.iloc[:, 0].replace(r'\.[0-9]*', '', regex = True, inplace = False)
        expr_data.set_index('gene_id', inplace = True)

        ensmbl2tf_file = os.path.join(self.data_path, "EnsemblToTF.txt")
        ensmbl2tf_map = pd.read_csv(ensmbl2tf_file, sep = '\t')
        ensmbl2tf_map.set_index('Gene stable ID', inplace = True)

        expr_data = ensmbl2tf_map.join(expr_data)
        expr_data.reset_index(inplace = True)
        expr_data.columns = ["Ensembl", "Gene Symbol", "TPM"]
#        print(np.sum(expr_data['TPM'] == None))
        #print(expr_data)
        print(expr_data.describe())
        self.expr_data = expr_data

        background_gene_file = os.path.join(self.data_path, "background_genes.txt")
        background_genes = pd.read_csv(background_gene_file, sep = "\t")
        print(f"Loaded background genes: {background_genes.shape}")

        ref_expr_file = os.path.join(self.data_path, "RefBool_ReferenceDistributions/Thresholds_CoTFcRF.mat")
        res = sio.loadmat(ref_expr_file)
        self.ref_expr = res["dists"]


p = Pipeline()
p.read_expression_data()

def plot_dist(dist, i = None):
    import random
    import pylab as plt
    N = dist.shape[0]
    if i is None:
        i = random.randrange(N)
    x1 = dist[i][0][0,0].reshape((1000,))
    x2 = dist[i][0][0,1].reshape((1000,))
    plt.plot(x1)
    plt.plot(x2)
    plt.title(i)
    plt.show()