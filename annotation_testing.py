import pickle 
import json
import os
from langsmith import utils
from langchain_openai import ChatOpenAI
import time
import re
import openai
from Tools_reasoning import function_descriptions
import json
import pandas as pd
import scanpy as sc
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import scvi
from scvi.model import SCVI
import plotly.express as px
import plotly.graph_objects as go
import ast
import matplotlib
import warnings
import numpy as np
import shutil
import gseapy as gp
import requests

resolution = 0.5

def extract_cells(data):
    # If the current element is a dictionary, iterate through its items.
    if isinstance(data, dict):
        new_dict = {}
        for key, value in data.items():
            # Skip keys that we want to remove.
            if key in ["explanation", "source"]:
                continue
            # Recursively process the value.
            new_dict[key] = extract_cells(value)
        return new_dict
    # If it's a list, process each element in the list.
    elif isinstance(data, list):
        return [extract_cells(item) for item in data]
    # If it's neither, return the element as is.
    else:
        return data

def get_rag_and_markers(tag):
    specification = None
    file_path = "media/specification.json"
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            specification = json.load(file)
    else:
        print("specification not found")
        return "-"
    base_file_path = os.path.join("schatbot/scChat_RAG", specification['marker'].lower())
    file_paths = []
    for tissue in specification['tissue']:
        tissue_path = os.path.join(base_file_path, tissue.lower(), specification['condition'] + '.json')
        file_paths.append(tissue_path)
    print("Constructed file paths:", file_paths)
    combined_data = {}
    for file_path in file_paths:
        if os.path.exists(file_path):
            print(f"File found: {file_path}")
            with open(file_path, 'r') as file:
                data = json.load(file)
        else:
            print(f"File not found: {file_path}")
            continue
        
        for cell_type, cell_data in data.items():
            if tag:
                if cell_type not in combined_data:
                    combined_data[cell_type] = cell_data
                else:
                    combined_data[cell_type]['markers'].extend(cell_data['markers'])
            else:
                combined_data = extract_cells(data)
    with open("testop.txt", "w") as fptr:
        fptr.write(json.dumps(combined_data, indent=4))
    return combined_data

def find_file_with_extension(directory_path, extension):
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(extension):
                return os.path.join(root, file)
    return None

def find_and_load_sample_mapping(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == 'sample_mapping.json':
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    sample_mapping = json.load(f)

    return sample_mapping

def filter_existing_genes(adata, gene_list):
    existing_genes = [gene for gene in gene_list if gene in adata.raw.var_names]
    return existing_genes

def calculate_cluster_statistics(adata, category, n_genes=25):
    markers = get_rag_and_markers(False)
    markers = extract_genes(markers)
    markers = filter_existing_genes(adata, markers)
    markers = list(set(markers))
    sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon', n_genes=n_genes)
    top_genes_df = extract_top_genes_stats(adata, groupby='leiden', n_genes=25)
    if sample_mapping:
        sc.tl.dendrogram(adata, groupby='leiden', use_rep='X_scVI')
    else:
        sc.tl.dendrogram(adata, groupby='leiden')
    marker_expression = sc.get.obs_df(adata, keys=['leiden'] + markers, use_raw=True)
    marker_expression.set_index('leiden', inplace=True)
    mean_expression = marker_expression.groupby('leiden').mean()
    expression_proportion = marker_expression.gt(0).groupby('leiden').mean()
    global_top_genes_df = top_genes_df
    global_mean_expression = mean_expression
    global_expression_proportion = expression_proportion
    return global_top_genes_df, global_mean_expression, global_expression_proportion

def extract_top_genes_stats(adata, groupby='leiden', n_genes=25):
    result = adata.uns['rank_genes_groups']
    gene_names = result['names']
    pvals = result['pvals']
    pvals_adj = result['pvals_adj']
    logfoldchanges = result['logfoldchanges']
    top_genes_stats = {group: {} for group in gene_names.dtype.names}
    for group in gene_names.dtype.names:
        top_genes_stats[group]['gene'] = gene_names[group][:n_genes]
        top_genes_stats[group]['pval'] = pvals[group][:n_genes]
        top_genes_stats[group]['pval_adj'] = pvals_adj[group][:n_genes]
        top_genes_stats[group]['logfoldchange'] = logfoldchanges[group][:n_genes]
    top_genes_stats_df = pd.concat({group: pd.DataFrame(top_genes_stats[group])
                                    for group in top_genes_stats}, axis=0)
    top_genes_stats_df = top_genes_stats_df.reset_index()
    top_genes_stats_df = top_genes_stats_df.rename(columns={'level_0': 'cluster', 'level_1': 'index'})
    return top_genes_stats_df

def extract_genes(data):
    genes = []
    if isinstance(data, dict):
        # If a "markers" key exists, extract gene names from its list
        if 'markers' in data:
            for marker in data['markers']:
                if isinstance(marker, dict) and 'gene' in marker:
                    genes.append(marker['gene'])
        # Recurse into every value of the dictionary
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                genes.extend(extract_genes(value))
    elif isinstance(data, list):
        for item in data:
            genes.extend(extract_genes(item))
    return genes


def generate_umap():
    global sample_mapping
    matplotlib.use('Agg')
    path = find_file_with_extension("media", ".h5ad")
    if not path:
        return ".h5ad file isn't given, unable to generate UMAP."
    adata = sc.read_h5ad(path)
    # Data preprocessing
    sc.pp.filter_cells(adata, min_genes=100)
    sc.pp.filter_genes(adata, min_cells=3)
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    adata = adata[adata.obs.pct_counts_mt < 20]
    adata.layers['counts'] = adata.X.copy()  # used by scVI-tools
    # Normalization
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata
    sample_mapping = find_and_load_sample_mapping("media")
    if sample_mapping:
        sc.pp.highly_variable_genes(adata, n_top_genes=3000, subset=True, layer='counts', flavor="seurat_v3", batch_key="Sample")
        SCVI.setup_anndata(adata, layer="counts", categorical_covariate_keys=["Sample"],
                            continuous_covariate_keys=['pct_counts_mt', 'total_counts'])
        model = SCVI.load(dir_path="schatbot/glioma_scvi_model", adata=adata)
        latent = model.get_latent_representation()
        adata.obsm['X_scVI'] = latent
        adata.layers['scvi_normalized'] = model.get_normalized_expression(library_size=1e4)
        sc.pp.neighbors(adata, use_rep='X_scVI')
    else:
        sc.pp.highly_variable_genes(adata, n_top_genes=3000, subset=True, layer='counts', flavor="seurat_v3")
        sc.pp.neighbors(adata)
    sc.tl.umap(adata, random_state=42)
    sc.tl.leiden(adata, resolution=resolution, random_state=42)
    if sample_mapping:
        adata.obs['patient_name'] = adata.obs['Sample'].map(sample_mapping)
    umap_df = adata.obsm['X_umap']
    adata.obs['UMAP_1'] = umap_df[:, 0]
    adata.obs['UMAP_2'] = umap_df[:, 1]
    markers = get_rag_and_markers(False)
    marker_tree = markers.copy()
    markers = extract_genes(markers)
    markers = filter_existing_genes(adata, markers)
    markers = list(set(markers))       
    statistic_data = sc.get.obs_df(adata, keys=['leiden'] + markers, use_raw=True)
    statistic_data.set_index('leiden', inplace=True)
    pd_mean_expression = pd.DataFrame(statistic_data.groupby('leiden').mean())
    pd_mean_expression.to_csv("basic_data/mean_expression.csv")
    pd_mean_expression.to_json("basic_data/mean_expression.json")
    expression_proportion = statistic_data.gt(0).groupby('leiden').mean()
    pd_expression_proportion = pd.DataFrame(expression_proportion)
    pd_expression_proportion.to_csv("basic_data/expression_proportion.csv")
    pd_expression_proportion.to_json("basic_data/expression_proportion.json")
    if sample_mapping is None:
        sc.tl.dendrogram(adata, groupby='leiden')
        with plt.rc_context({'figure.figsize': (10, 10)}):
            sc.pl.dotplot(adata, markers, groupby='leiden', swap_axes=True, use_raw=True,
                            standard_scale='var', dendrogram=True, color_map="Blues", save="dotplot.png")
            plt.close()
    else:
        sc.tl.dendrogram(adata, groupby='leiden', use_rep='X_scVI')
        with plt.rc_context({'figure.figsize': (10, 10)}):
            sc.pl.dotplot(adata, markers, groupby='leiden', swap_axes=True, use_raw=True,
                            standard_scale='var', dendrogram=True, color_map="Blues", save="dotplot.png")
            plt.close()
    adata.obs['cell_type'] = 'Unknown'
    adata.obs[['UMAP_1', 'UMAP_2', 'leiden', 'patient_name']].to_csv("basic_data/Overall cells_umap_data.csv", index=False)
    adata.obs[['UMAP_1', 'UMAP_2', 'leiden', 'patient_name']].to_csv("process_cell_data/Overall cells_umap_data.csv", index=False)
    dot_plot_data = statistic_data.reset_index().melt(id_vars='leiden', var_name='gene', value_name='expression')
    dot_plot_data.to_csv("basic_data/dot_plot_data.csv", index=False)
    dendrogram_data = adata.uns['dendrogram_leiden']
    pd_dendrogram_linkage = pd.DataFrame(dendrogram_data['linkage'], columns=['source', 'target', 'distance', 'count'])
    pd_dendrogram_linkage.to_csv("basic_data/dendrogram_data.csv", index=False)
    rag_data = get_rag_and_markers(True)
    summary = f"UMAP analysis completed. Data summary: {adata}, " \
                f"RAG Data : {str(rag_data)}. " \
                f"Cell counts details are provided. " \
                f"Additional data file generated: preface.txt."
    
    agent_content = """
        You are a bioinformatics researcher that recognizes the relationship between a cell and its subsets basing on the provide information. 
        As you receive input messages, you will record the cells and their markers' genes, as well as their subsets, and turn them into a readable format. 

        Purpose and Goals:
        * Accurately identify and record cells, their marker genes, and their subsets from input messages.
        * Organize the recorded information into a clear and readable format should have the the subsets labeled as subsets.
        * A clear relationship between root cell (the ancestor) and their subsets (the child) should be provided.
        * Do not include any additional words or explanations.
        """

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": agent_content},
            {"role": "user", "content": str(marker_tree)},
        ],
        temperature=0
    )
    marker_tree =  response.choices[0].message.content
    global_top_genes_df, global_mean_expression, global_expression_proportion = calculate_cluster_statistics(adata, "overall")
    
    gene_dict = {}
    for cluster, group in global_top_genes_df.groupby("cluster"):
        gene_dict[cluster] = list(group["gene"])
        
    summary2 = (f"Top genes details: {gene_dict}. "
                f"markers: {marker_tree}. ")
    final_summary = summary2
    return final_summary

if __name__ == "__main__":
    # final_summary = generate_umap() 
    # pd.to_pickle(final_summary, "annotation_testing.pkl")
    try:
        with open("annotation_testing.pkl", "rb") as file:
            final_summary = pd.read_pickle(file)
    except EOFError:
        print("The pickle file is empty or corrupted.")
    print(final_summary)
        
    agent_content = """
    You are a bioinformatics researcher that can do the cell annotation.
    Identify the cell type for each cluster using the following markers in the marker tree, arranged from highest to lowest expression levels. 
    This means you should consider the first several markers in the list to be more important. 
    Provide your result as the most specific cell type that is possible to be determined.
    
    Markers: <marker_tree>
    Provide your output in the following example format:
    Analysis: group_to_cell_type = {'0': 'Cell type','1': 'Cell type','2': 'Cell type','3': 'Cell type','4': 'Cell type', ...}.
    
    Working pipeline:
    * 1. Do the annotation basing on the instruction.
    * 2. Record all the cell type in the annotation result as their root cell.
         We call the highest level of the marker tree as root cell.
         Specifically, root cells are the cells that belong to none of the subsets.
    
    Strictly adhere to follwing rules:
    * 1. Adhere to the format and do not include any additional words or explanations.
    * 2. Results should contain root cells only.
    * 3. In the response, let the clusters in the answer stay in the format of raising power.
    """
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": agent_content},
            {"role": "user", "content": final_summary},
        ],
        temperature=0
    )
    results =  response.choices[0].message.content
    print(results)