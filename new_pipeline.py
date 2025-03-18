import pickle 
import json
import os
import time
import re
import openai
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

def clear_directory(directory_path):
    if not os.path.exists(directory_path):
        return
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Remove the file or symbolic link
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Remove the directory and all its contents
        except Exception as e:
            print("Not found")

def display_processed_umap(cell_type):
    cell_type2 = cell_type.split()[0].capitalize() + " cell"
    cell_type = cell_type.split()[0].capitalize() + " cells"
    if os.path.exists(f'umaps/{cell_type}_umap_data.csv'):
        umap_data = pd.read_csv(f'umaps/{cell_type}_umap_data.csv')
    else:
        umap_data = pd.read_csv(f'umaps/{cell_type2}_umap_data.csv')
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


def get_mapping(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == 'sample_mapping.json':
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    sample_mapping = json.load(f)
            # else:
            #     sample_mapping = None
    return sample_mapping

def get_h5ad(directory_path, extension):
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(extension):
                return os.path.join(root, file)
    return None

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

def get_rag(tag):
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

def filter_existing_genes(adata, gene_list):
    existing_genes = [gene for gene in gene_list if gene in adata.raw.var_names]
    return existing_genes

def preprocess_data(adata, sample_mapping=None):
    """Preprocess the AnnData object with consistent steps."""
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
    
    # Variable genes and neighbors
    if sample_mapping:
        sc.pp.highly_variable_genes(adata, n_top_genes=3000, subset=True, layer='counts', 
                                   flavor="seurat_v3", batch_key="Sample")
        SCVI.setup_anndata(adata, layer="counts", categorical_covariate_keys=["Sample"],
                          continuous_covariate_keys=['pct_counts_mt', 'total_counts'])
        model = SCVI.load(dir_path="schatbot/glioma_scvi_model", adata=adata)
        latent = model.get_latent_representation()
        adata.obsm['X_scVI'] = latent
        adata.layers['scvi_normalized'] = model.get_normalized_expression(library_size=1e4)
        sc.pp.neighbors(adata, use_rep='X_scVI')
    else:
        sc.pp.highly_variable_genes(adata, n_top_genes=3000, subset=True, layer='counts', 
                                   flavor="seurat_v3")
        sc.pp.neighbors(adata)
    
    return adata

def perform_clustering(adata, resolution=0.5, random_state=42):
    """Perform UMAP and Leiden clustering with consistent parameters."""
    sc.tl.umap(adata, random_state=random_state)
    sc.tl.leiden(adata, resolution=resolution, random_state=random_state)
    
    # Add UMAP coordinates to obs for easier access
    umap_df = adata.obsm['X_umap']
    adata.obs['UMAP_1'] = umap_df[:, 0]
    adata.obs['UMAP_2'] = umap_df[:, 1]
    
    return adata

def rank_genes(adata, groupby='leiden', method='wilcoxon', n_genes=25, key_added=None):
    """Rank genes by group with customizable key name."""
    if key_added is None:
        key_added = f'rank_genes_{groupby}'
    
    sc.tl.rank_genes_groups(adata, groupby, method=method, n_genes=n_genes, key_added=key_added)
    
    return adata

def create_marker_anndata(adata, markers, copy_uns=True, copy_obsm=True, copy_layers=True):
    """Create a copy of AnnData with only marker genes.
    
    Parameters:
    -----------
    adata : AnnData
        Original AnnData object
    markers : list
        List of marker genes
    copy_uns : bool
        Whether to copy uns dictionary
    copy_obsm : bool
        Whether to copy obsm dictionary
    copy_layers : bool
        Whether to copy layers dictionary
        
    Returns:
    --------
    adata_markers : AnnData
        AnnData object containing only marker genes
    """
    # Filter markers to those present in the dataset
    markers = filter_existing_genes(adata, markers)
    markers = list(set(markers))
    
    # Create a copy with only marker genes
    marker_idx = adata.var_names.get_indexer(markers)
    
    # Create data structures to copy
    uns_dict = adata.uns.copy() if copy_uns else {}
    obsm_dict = adata.obsm.copy() if copy_obsm else {}
    
    # Create layers dictionary if needed
    layers_dict = {}
    if copy_layers and marker_idx.size > 0:
        for layer in adata.layers:
            try:
                layers_dict[layer] = adata.layers[layer][:, marker_idx].copy()
            except IndexError:
                # Skip layers that can't be indexed with marker_idx
                pass
    
    # Create the marker-specific AnnData
    adata_markers = anndata.AnnData(
        X=adata.raw[:, markers].X.copy(),
        obs=adata.obs.copy(),   
        var=adata.raw[:, markers].var.copy(),
        uns=uns_dict,
        obsm=obsm_dict,
        layers=layers_dict,
        obsp=adata.obsp.copy() if hasattr(adata, 'obsp') else None
    )
    
    return adata_markers, markers

def rank_ordering(adata_or_result, key=None, n_genes=25):
    """Extract top genes statistics from ranking results.
    
    Works with either AnnData object and key or direct result dictionary.
    """
    # Handle either AnnData with key or direct result dictionary
    if isinstance(adata_or_result, anndata.AnnData):
        if key is None:
            # Try to find a suitable key
            rank_keys = [k for k in adata_or_result.uns.keys() if k.startswith('rank_genes_')]
            if not rank_keys:
                raise ValueError("No rank_genes results found in AnnData object")
            key = rank_keys[0]
        gene_names = adata_or_result.uns[key]['names']
    else:
        # Assume it's a direct result dictionary
        gene_names = adata_or_result['names']
    
    # Extract gene names for each group
    top_genes_stats = {group: {} for group in gene_names.dtype.names}
    for group in gene_names.dtype.names:
        top_genes_stats[group]['gene'] = gene_names[group][:n_genes]
    
    # Convert to DataFrame
    top_genes_stats_df = pd.concat({group: pd.DataFrame(top_genes_stats[group])
                                  for group in top_genes_stats}, axis=0)
    top_genes_stats_df = top_genes_stats_df.reset_index()
    top_genes_stats_df = top_genes_stats_df.rename(columns={'level_0': 'cluster', 'level_1': 'index'})
    
    return top_genes_stats_df

def save_analysis_results(adata, prefix, leiden_key='leiden', save_umap=True, 
                         save_dendrogram=True, save_dotplot=False, markers=None):
    """Save analysis results to files with consistent naming."""
    # Save UMAP data
    if save_umap:
        umap_cols = ['UMAP_1', 'UMAP_2', leiden_key]
        if 'patient_name' in adata.obs.columns:
            umap_cols.append('patient_name')
        if 'cell_type' in adata.obs.columns:
            umap_cols.append('cell_type')
            
        adata.obs[umap_cols].to_csv(f"{prefix}_umap_data.csv", index=False)
    
    # Save dendrogram data
    if save_dendrogram and f'dendrogram_{leiden_key}' in adata.uns:
        dendrogram_data = adata.uns[f'dendrogram_{leiden_key}']
        pd_dendrogram_linkage = pd.DataFrame(
            dendrogram_data['linkage'],
            columns=['source', 'target', 'distance', 'count']
        )
        pd_dendrogram_linkage.to_csv(f"{prefix}_dendrogram_data.csv", index=False)
    
    # Save dot plot data
    if save_dotplot and markers:
        statistic_data = sc.get.obs_df(adata, keys=[leiden_key] + markers, use_raw=True)
        statistic_data.set_index(leiden_key, inplace=True)
        dot_plot_data = statistic_data.reset_index().melt(
            id_vars=leiden_key, var_name='gene', value_name='expression'
        )
        dot_plot_data.to_csv(f"{prefix}_dot_plot_data.csv", index=False)

# Main pipeline functions
def generate_umap(resolution=0.5):
    """Generate initial UMAP clustering on the full dataset."""
    global sample_mapping
    
    # Setup
    matplotlib.use('Agg')
    path = get_h5ad("media", ".h5ad")
    if not path:
        return ".h5ad file isn't given, unable to generate UMAP."
    
    # Load data
    adata = sc.read_h5ad(path)
    
    # Get sample mapping
    sample_mapping = get_mapping("media")
    
    # Apply sample mapping if available
    if sample_mapping:
        adata.obs['patient_name'] = adata.obs['Sample'].map(sample_mapping)
    
    # Preprocess data
    adata = preprocess_data(adata, sample_mapping)
    
    # Perform clustering
    adata = perform_clustering(adata, resolution=resolution)
    
    # Rank all genes
    adata = rank_genes(adata, groupby='leiden', n_genes=25, key_added='rank_genes_all')
    
    # Get markers and create marker-specific AnnData
    markers = get_rag(False)
    marker_tree = markers.copy()
    markers = extract_genes(markers)
    
    adata_markers, filtered_markers = create_marker_anndata(adata, markers)
    
    # Rank genes in marker dataset
    adata_markers = rank_genes(adata_markers, n_genes=25, key_added='rank_genes_markers')
    
    # Copy marker ranking to original dataset
    adata.uns['rank_genes_markers'] = adata_markers.uns['rank_genes_markers']
            
    # Create dendrogram
    use_rep = 'X_scVI' if sample_mapping else None
    if use_rep:
        sc.tl.dendrogram(adata, groupby='leiden', use_rep=use_rep)
    else:
        sc.tl.dendrogram(adata, groupby='leiden')
    
    # Create dot plot
    with plt.rc_context({'figure.figsize': (10, 10)}):
        sc.pl.dotplot(adata, filtered_markers, groupby='leiden', swap_axes=True, use_raw=True,
                    standard_scale='var', dendrogram=True, color_map="Blues", save="dotplot.png")
        plt.close()
    
    # Initialize cell type as unknown
    adata.obs['cell_type'] = 'Unknown'
    
    # Save data
    save_analysis_results(
        adata, 
        prefix="basic_data/Overall cells", 
        save_dotplot=True, 
        markers=filtered_markers
    )
    save_analysis_results(
        adata, 
        prefix="process_cell_data/Overall cells", 
        save_dotplot=False
    )
    
    # Extract top genes from marker-specific ranking
    top_genes_df = rank_ordering(adata, key='rank_genes_markers', n_genes=25)
    
    # Create gene dictionary
    gene_dict = {}
    for cluster, group in top_genes_df.groupby("cluster"):
        gene_dict[cluster] = list(group["gene"])
    
    return gene_dict, marker_tree, adata

def process_cells(adata, cell_type, resolution=None):
    """Process specific cell type with consistent workflow."""
    if resolution is None:
        resolution = 0.8  # Default higher resolution for subtype clustering
    
    # Standardize cell type names for flexible matching
    cell_type_base = cell_type.split()[0]
    cell_type_cap = cell_type_base.capitalize()
    possible_types = [
        f"{cell_type_cap} cells",
        f"{cell_type_cap} cell",
        cell_type_cap
    ]
    
    # Filter cells based on cell type with flexible matching
    mask = adata.obs['cell_type'].isin(possible_types)
    if mask.sum() == 0:
        print(f"WARNING: No cells found with cell type matching '{cell_type_cap}'")
        return None, None
    
    filtered_cells = adata[mask].copy()
    
    # Perform new clustering on filtered cells
    sc.tl.pca(filtered_cells, svd_solver='arpack')
    sc.pp.neighbors(filtered_cells)
    filtered_cells = perform_clustering(filtered_cells, resolution=resolution)
    
    # Create ranking key based on cell type
    rank_key = f"rank_genes_{cell_type_base.lower()}"
    
    # Rank genes for the filtered cells
    filtered_cells = rank_genes(filtered_cells, key_added=rank_key)
    
    # Get markers and create marker-specific sub-AnnData if needed
    markers = get_rag(False)
    markers = extract_genes(markers)
    
    filtered_markers, marker_list = create_marker_anndata(filtered_cells, markers)
    
    # Rank marker genes
    marker_key = f"rank_markers_{cell_type_base.lower()}"
    filtered_markers = rank_genes(filtered_markers, key_added=marker_key)
    
    # Copy marker ranking to filtered dataset
    filtered_cells.uns[marker_key] = filtered_markers.uns[marker_key]
    
    # Create dendrogram
    sc.tl.dendrogram(filtered_cells, groupby='leiden')
    
    # Save data
    standardized_name = f"{cell_type_cap} cells"
    save_analysis_results(
        filtered_cells,
        prefix=f"process_cell_data/{standardized_name}",
        save_dotplot=True,
        markers=marker_list
    )
    
    # Extract top genes with preference for marker-specific ranking
    if marker_key in filtered_cells.uns:
        top_genes_df = rank_ordering(filtered_cells, key=marker_key, n_genes=25)
    else:
        top_genes_df = rank_ordering(filtered_cells, key=rank_key, n_genes=25)
    
    # Create gene dictionary
    gene_dict = {}
    for cluster, group in top_genes_df.groupby("cluster"):
        gene_dict[cluster] = list(group["gene"])
    
    # Save processed data
    umap_df = filtered_cells.obs[['UMAP_1', 'UMAP_2', 'leiden', 'cell_type']]
    if 'patient_name' in filtered_cells.obs.columns:
        umap_df['patient_name'] = filtered_cells.obs['patient_name']
        
    fname = f'{standardized_name}_adata_processed.pkl'
    with open(fname, 'wb') as file:
        pickle.dump(umap_df, file)
    
    return gene_dict, filtered_cells

def label_clusters(annotation_result, cell_type, adata):
    """Label clusters with consistent cell type handling."""
    # Standardize cell type names
    cell_type_base = cell_type.split()[0]
    cell_type_cap = cell_type_base.capitalize()
    standardized_cell_type = f"{cell_type_cap} cells"
    
    try:
        adata = adata.copy()
        
        # Parse annotation mapping
        start_idx = annotation_result.find("{")
        end_idx = annotation_result.rfind("}") + 1
        str_map = annotation_result[start_idx:end_idx]
        map2 = ast.literal_eval(str_map)
        map2 = {str(key): value for key, value in map2.items()}
        
        # Apply annotations - different handling for overall cells vs specific cell types
        if cell_type_cap.lower() == "overall":
            # For overall cells, directly apply annotations to the main dataset
            adata.obs['cell_type'] = 'Unknown'
            for group, cell_type_value in map2.items():
                adata.obs.loc[adata.obs['leiden'] == group, 'cell_type'] = cell_type_value
                
            # Save annotated data
            save_analysis_results(
                adata,
                prefix=f"umaps/{standardized_cell_type}",
                save_dendrogram=False,
                save_dotplot=False
            )
            
            # Save annotated adata
            fname = f'annotated_adata/{standardized_cell_type}_annotated_adata.pkl'
            with open(fname, "wb") as file:
                pickle.dump(adata, file)
                
        else:
            # For specific cell types, we need to re-cluster first
            # This is typically called after process_cells()
            specific_cells = adata.copy()
            
            # Apply annotations
            specific_cells.obs['cell_type'] = 'Unknown'
            for group, cell_type_value in map2.items():
                specific_cells.obs.loc[specific_cells.obs['leiden'] == group, 'cell_type'] = cell_type_value
            
            # Save annotated data
            save_analysis_results(
                specific_cells,
                prefix=f"umaps/{standardized_cell_type}",
                save_dendrogram=False,
                save_dotplot=False
            )
            
            # Save annotated adata
            fname = f'annotated_adata/{standardized_cell_type}_annotated_adata.pkl'
            with open(fname, "wb") as file:
                pickle.dump(specific_cells, file)
                
            return specific_cells
            
    except (SyntaxError, ValueError) as e:
        print(f"Error in parsing the map: {e}")
        
    return adata

if __name__ == "__main__":
    clear_directory("annotated_adata")
    clear_directory("basic_data")
    clear_directory("umaps")
    clear_directory("process_cell_data")  
    gene_dict, marker_tree, adata = generate_umap()    

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
    marker_tree = get_rag(False)
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
    display_umap(cell_type)
    display_processed_umap(cell_type)
    
    # # print("===============================================================================")
    # # print("Second round")
    # # print("===============================================================================")
    
    
    spec_cell_type = "T cell"
    gene_dict, adata = process_cells(adata, spec_cell_type)
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
    display_umap(spec_cell_type)
    display_processed_umap(spec_cell_type)
    print(spec_annotation_result)