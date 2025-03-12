import pickle 
import json
import os
import time
import re
import openai
from Tools_reasoning import function_descriptions
import json
import pandas as pd
import scanpy as sc
from dotenv import load_dotenv
import matplotlib.pyplot as plt
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
from langsmith import utils
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
from langgraph.func import entrypoint, task
from langgraph.graph import add_messages
from typing import TypedDict, Dict, List, Any
import ast
from collections import deque
import anndata

resolution = 0.5

def leaf_checker(gene_dict, marker_tree, annotation_results):
    def extract_end_leaf_cells(data):
        leaf_cells = []
        if isinstance(data, dict):
            for cell, details in data.items():
                if "subsets" not in details or not details["subsets"]:
                    leaf_cells.append(cell)
                else:
                    leaf_cells.extend(extract_end_leaf_cells(details["subsets"]))
        elif isinstance(data, list):
            for item in data:
                leaf_cells.extend(extract_end_leaf_cells(item))
        return leaf_cells

    leaf_cells = extract_end_leaf_cells(marker_tree)

    annotation_results = str(annotation_results)
    dict_str = annotation_results.split("=", 1)[1].strip().rstrip(".")
    group_to_cell_type = ast.literal_eval(dict_str)

    non_leaf_clusters = {}
    leaf_clusters = {}
    for cluster, cell_type in group_to_cell_type.items():
        if cell_type in leaf_cells:
            leaf_clusters[cluster] = cell_type
        else:
            non_leaf_clusters[cluster] = cell_type
            
    non_leaf_gene = {
        cluster: gene_dict[str(cluster)]
        for cluster in non_leaf_clusters
        if str(cluster) in gene_dict
    }
    return non_leaf_clusters, leaf_clusters

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

def extract_cells(data):
    if isinstance(data, dict):
        new_dict = {}
        for key, value in data.items():
            if key in ["explanation", "source"]:
                continue
            new_dict[key] = extract_cells(value)
        return new_dict
    elif isinstance(data, list):
        return [extract_cells(item) for item in data]
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

def display_processed_umap(cell_type):
    cell_type2 = cell_type.split()[0].capitalize() + " cell"
    cell_type = cell_type.split()[0].capitalize() + " cells"
    if os.path.exists(f'umaps/{cell_type}_annotated_umap_data.csv'):
        umap_data = pd.read_csv(f'umaps/{cell_type}_annotated_umap_data.csv')
    else:
        umap_data = pd.read_csv(f'umaps/{cell_type2}_annotated_umap_data.csv')
    fig = px.scatter(
        umap_data,
        x="UMAP_1",
        y="UMAP_2",
        color="cell_type",
        symbol="patient_name",
        title=f'{cell_type} UMAP Plot',
        labels={"UMAP_1": "UMAP 1", "UMAP_2": "UMAP 2"}
    )
    fig.update_traces(marker=dict(size=5, opacity=0.8))
    fig.update_layout(
        width=1200,
        height=800,
        autosize=True,
        showlegend=True
    )
    fig.show()
    fig_json = fig.to_json()
    return fig_json

def display_umap(cell_type):
    display_flag = True
    cell_type = cell_type.split()[0].capitalize() + " cells"
    umap_data = pd.read_csv(f'process_cell_data/{cell_type}_umap_data.csv')
    if cell_type != "Overall cells":
        umap_data['original_cell_type'] = umap_data['cell_type']
        umap_data['cell_type'] = 'Unknown'
    fig = px.scatter(
        umap_data,
        x="UMAP_1",
        y="UMAP_2",
        color="leiden",
        symbol="patient_name",
        title=f"{cell_type} UMAP Plot",
        labels={"UMAP_1": "UMAP 1", "UMAP_2": "UMAP 2"}
    )
    fig.update_traces(marker=dict(size=5, opacity=0.8))
    fig.update_layout(
        width=1200,
        height=800,
        autosize=True,
        showlegend=False
    )
    custom_legend = go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=10, color='rgba(0,0,0,0)'),
        legendgroup="Unknown",
        showlegend=True,
        name="Unknown"
    )
    fig.add_trace(custom_legend)
    fig.show()
    fig_json = fig.to_json()
    return fig_json

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
    
    marker_idx = adata.var_names.get_indexer(markers)
    adata_markers = anndata.AnnData(
        X=adata.raw[:, markers].X.copy(),
        obs=adata.obs.copy(),   
        var=adata.raw[:, markers].var.copy(),
        uns=adata.uns.copy(),
        obsm=adata.obsm.copy(),
        layers={layer: adata.layers[layer][:, marker_idx].copy() for layer in adata.layers},
        obsp=adata.obsp.copy()
    )
    adata = adata_markers
    adata.raw = adata
        
    statistic_data = sc.get.obs_df(adata, keys=['leiden'] + markers, use_raw=True)
    statistic_data.set_index('leiden', inplace=True)
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
    global_top_genes_df, global_mean_expression, global_expression_proportion = calculate_cluster_statistics(adata, "overall")
    
    gene_dict = {}
    for cluster, group in global_top_genes_df.groupby("cluster"):
        gene_dict[cluster] = list(group["gene"])
    
    return gene_dict, marker_tree, global_mean_expression, global_expression_proportion, adata

def label_clusters(annotation_result, cell_type, adata):
    standardized_cell_type3 = cell_type.split()[0].capitalize()
    standardized_cell_type2 = cell_type.split()[0].capitalize() + " cell"
    standardized_cell_type = cell_type.split()[0].capitalize() + " cells"
    try:
        adata = adata.copy()
        start_idx = annotation_result.find("{")
        end_idx = annotation_result.rfind("}") + 1
        str_map = annotation_result[start_idx:end_idx]
        map2 = ast.literal_eval(str_map)
        map2 = {str(key): value for key, value in map2.items()}
        if standardized_cell_type == "Overall cells" or standardized_cell_type2 == "Overall cell":
            adata.obs['cell_type'] = 'Unknown'
            for group, cell_type in map2.items():
                adata.obs.loc[adata.obs['leiden'] == group, 'cell_type'] = cell_type
            adata.obs[['UMAP_1', 'UMAP_2', 'leiden', 'patient_name', 'cell_type']].to_csv(
                f'umaps/{standardized_cell_type}_annotated_umap_data.csv', index=False)
            annotated_adata = adata.copy()
            fname = f'annotated_adata/Overall cells_annotated_adata.pkl'
            with open(fname, "wb") as file:
                pickle.dump(annotated_adata, file)
        else:
            specific_cells = adata[adata.obs['cell_type'].isin([standardized_cell_type])].copy()
            if specific_cells.shape[0] == 0:
                specific_cells = adata[adata.obs['cell_type'].isin([standardized_cell_type2])].copy()
            if specific_cells.shape[0] == 0:
                specific_cells = adata[adata.obs['cell_type'].isin([standardized_cell_type3])].copy()
            sc.tl.pca(specific_cells, svd_solver='arpack')
            sc.pp.neighbors(specific_cells, random_state=42)
            sc.tl.umap(specific_cells, random_state=42)
            sc.tl.leiden(specific_cells, resolution=resolution, random_state=42)
            specific_cells.obs['cell_type'] = 'Unknown'
            for group, cell_type in map2.items():
                specific_cells.obs.loc[specific_cells.obs['leiden'] == group, 'cell_type'] = cell_type
            specific_cells.obs[['UMAP_1', 'UMAP_2', 'leiden', 'patient_name', 'cell_type']].to_csv(
                f'umaps/{standardized_cell_type}_annotated_umap_data.csv', index=False)
            annotated_adata = specific_cells.copy()
            fname = f'annotated_adata/{standardized_cell_type}_annotated_adata.pkl'
            with open(fname, "wb") as file:
                pickle.dump(annotated_adata, file)
    except (SyntaxError, ValueError) as e:
        print(f"Error in parsing the map: {e}")
    return adata

def process_cells(adata, cell_type):
    global sample_mapping
    sample_mapping = find_and_load_sample_mapping("media")
    processed_cell_name = cell_type
    adata = adata.copy()
    cell_type_split = cell_type.split()[0]
    cell_type_cap = cell_type_split.capitalize()
    cell_type_full = cell_type_cap + " cells"
    cell_type_alt = cell_type_cap + " cell"
    filtered_cells = adata[adata.obs['cell_type'].isin([cell_type_full])].copy()
    if filtered_cells.shape[0] == 0:
        filtered_cells = adata[adata.obs['cell_type'].isin([cell_type_alt])].copy()
    if filtered_cells.shape[0] == 0:
        filtered_cells = adata[adata.obs['cell_type'].isin([cell_type_cap])].copy()
    sc.tl.pca(filtered_cells, svd_solver='arpack')
    sc.pp.neighbors(filtered_cells)
    sc.tl.umap(filtered_cells)
    sc.tl.leiden(filtered_cells, resolution=resolution)
    umap_data = filtered_cells.obsm['X_umap']
    umap_df = pd.DataFrame(umap_data, columns=['UMAP_1', 'UMAP_2'])
    umap_df['cell_type'] = filtered_cells.obs['cell_type'].values
    umap_df['patient_name'] = filtered_cells.obs['patient_name'].values
    umap_df['leiden'] = filtered_cells.obs['leiden'].values
    umap_df.to_csv(f'process_cell_data/{cell_type_full}_umap_data.csv', index=False)
    global_top_genes_df, global_mean_expression, global_expression_proportion = calculate_cluster_statistics(filtered_cells, cell_type_full)
    gene_dict = {}
    for cluster, group in global_top_genes_df.groupby("cluster"):
        gene_dict[cluster] = list(group["gene"])
        
    fname = f'{cell_type_full}_adata_processed.pkl'
    with open(fname, 'wb') as file:
        pickle.dump(umap_df, file)
    return gene_dict

if __name__ == "__main__":
    gene_dict, marker_tree, mean_expression, expression_proportion, adata = generate_umap()  
    # pd.to_pickle(marker_tree, "marker_tree.pkl")
    # pd.to_pickle(gene_dict, "gene_dict.pkl")
    # pd.to_pickle(adata, "adata.pkl")
    # with open("gene_dict.pkl", "rb") as file:
    #     gene_dict = pd.read_pickle(file)
    # with open("marker_tree.pkl", "rb") as file:
    #     marker_tree = pd.read_pickle(file)    
    # with open("adata.pkl", "rb") as file:
    #     adata = pd.read_pickle(file)

    def flatten_tree(tree, mapping=None):
        if mapping is None:
            mapping = {}
        if isinstance(tree, dict):
            for key, value in tree.items():
                mapping[key] = value
                if isinstance(value, dict) and "subsets" in value:
                    flatten_tree(value["subsets"], mapping)
        return mapping

    def get_markers_by_effective_level_for_cell(root_cell, target_level, cell_lookup):
        def collect_all_marker_genes(cell_node):
            markers = []
            if "markers" in cell_node:
                markers.extend(marker["gene"] for marker in cell_node["markers"])
            if "subsets" in cell_node:
                for subcell in cell_node["subsets"].values():
                    markers.extend(collect_all_marker_genes(subcell))
            return markers

        if root_cell:
            if root_cell not in cell_lookup:
                print(f"Cell type '{root_cell}' not found.")
                return {}
            starting_nodes = [(root_cell, 1)]
        else:
            starting_nodes = [(cell, 1) for cell in cell_lookup.keys()]

        descendant_levels = {}
        queue = deque(starting_nodes)

        while queue:
            cell, rel_level = queue.popleft()
            descendant_levels[cell] = rel_level
            node = cell_lookup[cell]
            if "subsets" in node and node["subsets"]:
                for subcell in node["subsets"].keys():
                    queue.append((subcell, rel_level + 1))

        result = {}
        for cell, rel_level in descendant_levels.items():
            node = cell_lookup[cell]
            is_leaf = not ("subsets" in node and node["subsets"])
            if rel_level == target_level or (is_leaf and rel_level < target_level):
                result[cell] = collect_all_marker_genes(node)

        return result

    # print("===============================================================================")
    # print("First round")
    # print("===============================================================================")
    marker_tree = get_rag_and_markers(False)
    cell_lookup = flatten_tree(marker_tree)
    spec_marker_tree = get_markers_by_effective_level_for_cell("", 4, cell_lookup)
    input_msg =(f"Top genes details: {gene_dict}. "
                f"Markers: {spec_marker_tree}. ")
    
    messages = [
        SystemMessage(content="""
            You are a bioinformatics researcher that can do the cell annotation.
            The following are the data and its decription that you will receive.
            * 1. Gene list: top 25 cells arragned from the highest to lowest expression levels in each cluster.
            * 2. Marker tree: marker genes that can help annotating cell type.           
            
            Identify the cell type for each cluster using the following markers in the marker tree.
            This means you will have to use the markers to annotate the cell type in the gene list. 
            Provide your result as the most specific cell type that is possible to be determined.
            
            Provide your output in the following example format:
            Analysis: group_to_cell_type = {'0': 'Cell type','1': 'Cell type','2': 'Cell type','3': 'Cell type','4': 'Cell type', ...}.
            
            Strictly adhere to follwing rules:
            * 1. Adhere to the dictionary format and do not include any additional words or explanations.
            * 2. The cluster number in the result dictionary must be arranged with raising power.
            """),
        HumanMessage(content=input_msg)
    ]
    model = ChatOpenAI(
        model="gpt-4o",
        temperature=0
    )
    results = model.invoke(messages)
    annotation_result = results.content
    print(annotation_result)

    cell_type = "Overall cells"
    adata = label_clusters(annotation_result, cell_type, adata)
    # display_umap(cell_type)
    # display_processed_umap(cell_type)
    
    # print("===============================================================================")
    # print("Second round")
    # print("===============================================================================")
    
    
    spec_cell_type = "T cell"
    gene_dict = process_cells(adata, spec_cell_type)
    spec_marker_tree = get_markers_by_effective_level_for_cell(spec_cell_type, 2, cell_lookup)
    model = ChatOpenAI(
        model="gpt-4o",
        temperature=0
    )

    input_msg =(f"Top genes details: {gene_dict}. "
                f"Markers: {spec_marker_tree}. ")
    messages = [
        SystemMessage(content="""
            You are a bioinformatics researcher that can do the cell annotation.
            The following are the data and its decription that you will receive.
            * 1. Gene list: top 25 cells arragned from the highest to lowest expression levels in each cluster.
            * 2. Marker tree: marker genes that can help annotating cell type.           
            
            Identify the cell type for each cluster using the following markers in the marker tree.
            This means you will have to use the markers to annotate the cell type in the gene list. 
            Provide your result as the most specific cell type that is possible to be determined.
            
            Provide your output in the following example format:
            Analysis: group_to_cell_type = {'0': 'Cell type','1': 'Cell type','2': 'Cell type','3': 'Cell type','4': 'Cell type', ...}.
            
            Strictly adhere to follwing rules:
            * 1. Adhere to the dictionary format and do not include any additional words or explanations.
            * 2. The cluster number in the result dictionary must be arranged with raising power.
            """),
        HumanMessage(content=input_msg)
    ]

    results = model.invoke(messages)
    spec_annotation_result = results.content
    adata = label_clusters(spec_annotation_result, spec_cell_type, adata)
    # display_umap(spec_cell_type)
    # display_processed_umap(spec_cell_type)
    print(spec_annotation_result)