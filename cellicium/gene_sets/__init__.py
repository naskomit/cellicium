import pandas as pd
import os.path as path
#from typing import String

class GeneSetManager():
    def __init__(self):
        self.cache = {}

    def cell_cycle_macosko(self, group: str):
        """
        Paper: Highly Parallel Genome-wide Expression Profiling of Individual Cells Using Nanoliter Droplets
        URL: https://doi.org/10.1016/j.cell.2015.05.002
        """
        data_file_name = 'Macosko_cell_cycle_genes.txt'
        data_file_loc = path.join(path.dirname(__file__), data_file_name)
        data = pd.read_csv(data_file_loc, sep = '\t')
        s_genes = data['S'].dropna().values
        g2m_genes = data['G2.M'].dropna().values
        m_genes = data['M'].dropna().values
        if group == 'S':
            return s_genes
        elif group == 'G2M':
            return g2m_genes
        elif group == 'M':
            return m_genes
        else:
            raise ValueError(f'Unknown cell cycle gene group ${group}')

    def tf_interactions_dorothea(self):
        """
        Paper: Benchmark and integration of resources for the estimation of human transcription factor activities
        URL: https://doi.org/10.1101/gr.240663.118
        """
        # TF activities From Saez-Rodriguez' Lab 'Dorothea'
        data_file_name = 'Human_Regulons_Normal.csv.gz'
        data_file_loc = path.join(path.dirname(__file__), 'dorothea', data_file_name)
        data = pd.read_csv(data_file_loc)
        data.rename({'TF': 'tf'}, inplace = True, axis = 1)
        return data

    def tf_interactions_trrust_v2(self, species : str):
        """
        Paper: TRRUST v2: an expanded reference database of human and mouse transcriptional regulatory interactions
        URL: https://doi.org/10.1093/nar/gkx1013
        """
        if species == 'human':
            data_file_name = 'trrust_rawdata.human.tsv'
        elif species == 'mouse':
            data_file_name = 'trrust_rawdata.mouse.tsv'
        else:
            raise ValueError(f'Species must be human or mouse, given ${species}')
        data_file_loc = path.join(path.dirname(__file__), 'trrustv2', data_file_name)
        data = pd.read_csv(data_file_loc, sep = '\t', header = None, names = ["tf", "gene", "effect", "reference"])
        return data
