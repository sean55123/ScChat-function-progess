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
from neo4j import GraphDatabase
import scipy

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
    """Display annotated UMAP plot with robust column checking."""
    try:
        cell_type2 = cell_type.split()[0].capitalize() + " cell"
        cell_type = cell_type.split()[0].capitalize() + " cells"
        
        # Try both possible file names
        if os.path.exists(f'umaps/{cell_type}_umap_data.csv'):
            umap_data = pd.read_csv(f'umaps/{cell_type}_umap_data.csv')
        elif os.path.exists(f'umaps/{cell_type2}_umap_data.csv'):
            umap_data = pd.read_csv(f'umaps/{cell_type2}_umap_data.csv')
        else:
            print(f"Warning: Could not find UMAP data for {cell_type} or {cell_type2}")
            return None
        
        # Check for required columns
        required_cols = ["UMAP_1", "UMAP_2", "cell_type"]
        missing_cols = [col for col in required_cols if col not in umap_data.columns]
        if missing_cols:
            print(f"Warning: Missing required columns in UMAP data: {missing_cols}")
            return None
        
        # Create plot parameters based on available columns
        plot_params = {
            "x": "UMAP_1",
            "y": "UMAP_2",
            "color": "cell_type",
            "title": f'{cell_type} UMAP Plot',
            "labels": {"UMAP_1": "UMAP 1", "UMAP_2": "UMAP 2"}
        }
        
        # Add symbol parameter only if patient_name exists
        if "patient_name" in umap_data.columns:
            plot_params["symbol"] = "patient_name"
        
        # Create the plot
        fig = px.scatter(umap_data, **plot_params)
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
    
    except Exception as e:
        print(f"Error in display_processed_umap: {e}")
        return None

def display_umap(cell_type):
    """Display clustering UMAP plot with robust column checking."""
    try:
        display_flag = True
        cell_type = cell_type.split()[0].capitalize() + " cells"
        file_path = f'process_cell_data/{cell_type}_umap_data.csv'
        
        if not os.path.exists(file_path):
            print(f"Warning: Could not find UMAP data at {file_path}")
            return None
            
        umap_data = pd.read_csv(file_path)
        
        # Check for required columns
        required_cols = ["UMAP_1", "UMAP_2", "leiden"]
        missing_cols = [col for col in required_cols if col not in umap_data.columns]
        if missing_cols:
            print(f"Warning: Missing required columns in UMAP data: {missing_cols}")
            return None
        
        # Handle cell type for specific cell types
        if cell_type != "Overall cells" and "cell_type" in umap_data.columns:
            umap_data['original_cell_type'] = umap_data['cell_type']
            umap_data['cell_type'] = 'Unknown'
        
        # Create plot parameters based on available columns
        plot_params = {
            "x": "UMAP_1",
            "y": "UMAP_2",
            "color": "leiden",
            "title": f"{cell_type} UMAP Plot",
            "labels": {"UMAP_1": "UMAP 1", "UMAP_2": "UMAP 2"}
        }
        
        # Add symbol parameter only if patient_name exists
        if "patient_name" in umap_data.columns:
            plot_params["symbol"] = "patient_name"
        
        # Create the plot
        fig = px.scatter(umap_data, **plot_params)
        fig.update_traces(marker=dict(size=5, opacity=0.8))
        fig.update_layout(
            width=1200,
            height=800,
            autosize=True,
            showlegend=False
        )
        
        # Add custom legend
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
        
    except Exception as e:
        print(f"Error in display_umap: {e}")
        return None

def get_url(gene_name):
    gene_name = str(gene_name).upper()
    url_format = "https://www.ncbi.nlm.nih.gov/gene/?term="
    return gene_name+": "+url_format+gene_name

def explain_gene(gene_dict, marker_tree, annotation_result):
    model = ChatOpenAI(
        model="gpt-4o",
        temperature=0
    )
    
    input_msg =(f"Top genes details: {gene_dict}. "
                f"Markers: {marker_tree}. "
                f"Annotation results: {annotation_result}")
    
    messages = [
        SystemMessage(content="""
            You are a bioinformatics researcher that can do the cell annotation.
            The following are the data and its decription that you will receive.
            * 1. Gene list: top 25 cells arragned from the highest to lowest expression levels in each cluster.
            * 2. Marker tree: marker genes that can help annotating cell type. 
            * 3. Cell annotation results.
            
            Basing on the given information you will have to return a gene list with top 3 possible genes that you use to do the cell annotation.
            This mean you have to give a gene list with 5 most possible genes for each cluster that be used for cell annotation.
            
            Provide your output in the following example format:
            {'0': ['gene 1', 'gene 2', ...],'1': ['gene 1', 'gene 2', ...],'2': ['gene 1', 'gene 2', ...],'3': ['gene 1', 'gene 2', ...],'4': ['gene 1', 'gene 2', ...], ...}.
            
            Strictly adhere to follwing rules:
            * 1. Adhere to the dictionary format and do not include any additional words or explanations.
            * 2. The cluster number in the result dictionary must be arranged with raising power.            
            """),
        HumanMessage(content=input_msg)
    ]
    results = model.invoke(messages)
    gene = results.content
    gene = ast.literal_eval(gene)

    url_clusters = {}
    for cluster_id, gene_list in gene.items():
        url_clusters[cluster_id] = [get_url(gene) for gene in gene_list]
        
    input_msg =(f"Annotation results: {annotation_result}"
                f"Top marker genes: {gene}"
                f"URL for top marker genes: {url_clusters}")
    messages = [
        SystemMessage(content="""
            You are a bioinformatics researcher that can do the cell annotation.
            The following are the data and its decription that you will receive.
            * 1. Cell annotation results.
            * 2. Top marker genes that have been use to do cell annotation.
            * 3. URL for top marker genes, which are the references for each marker genes in each cluster.
            
            Basing on the given information you will have to explain why these marker genes have been used to do cell annotation while with reference URL with it.
            This means you will have to expalin why those top marker genes are used to represent cell in cell annotation results.
            After explanation you will have to attach genes' corresponding URL. 
            
            The response is supposed to follow the following format:
            ### Cluster 0: Cells
            - **Gene 1**: Explanation.
            - **Gene 2**: Explanation.
            - **Gene 3**: Explanation.
            - **Gene 4**: Explanation.
            - **Gene 5**: Explanation.

            **URLs**:
            - Gene 1: (URL 1)
            - Gene 2: (URL 2)
            - Gene 3: (URL 3)
            - Gene 4: (URL 4)
            - Gene 5: (URL 5)
            ...
            
            ** All the clusters should be included.
            ** If there is same cell appears in different clusters you can combine them together.
            """),
        HumanMessage(content=input_msg)
    ]    
    results = model.invoke(messages)
    explanation = results.content
    
    return explanation
    
def get_mapping(directory):
    sample_mapping = None
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == 'sample_mapping.json':
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r') as f:
                        sample_mapping = json.load(f)
                    return sample_mapping  # Return immediately when found
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    
    return sample_mapping  # Return None if no valid file is found

def get_h5ad(directory_path, extension):
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(extension):
                return os.path.join(root, file)
    return None

def extract_genes(data):
    genes = []
    if isinstance(data, dict):
        # Check if this is a cell type dictionary with a 'markers' key
        if 'markers' in data:
            # If markers is directly a list of gene names
            if isinstance(data['markers'], list):
                genes.extend(data['markers'])
        
        # Recurse into nested dictionaries
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                genes.extend(extract_genes(value))
                
    elif isinstance(data, list):
        # Process each item in the list
        for item in data:
            genes.extend(extract_genes(item))
            
    return genes

def get_rag():
    """Retrieve marker genes using Neo4j graph database."""
    # Initialize Neo4j connection
    uri = "bolt://localhost:7689"
    driver = GraphDatabase.driver(uri, auth=("neo4j", "37754262"))
    
    # Load specification from JSON
    specification = None
    file_path = "media/specification_graph.json"
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            specification = json.load(file)
    else:
        print("specification not found")
        return "-"
    
    # Extract parameters from specification
    database = specification['database']
    system = specification['system']
    organ = specification['organ']

    # Initialize result structure
    combined_data = {}
    
    try:
        with driver.session(database=database) as session:
            # Get cell types and markers from Neo4j
            query = """
            MATCH (s:System {name: $system})-[:HAS_ORGAN]->(o:Organ {name: $organ})-[:HAS_CELL]->(c:CellType)-[:HAS_MARKER]->(m:Marker)
            RETURN c.name as cell_name, m.markers as marker_list
            """
            result = session.run(query, system=system, organ=organ)
            
            # Process results into expected format
            for record in result:
                cell_name = record["cell_name"]
                marker_genes = record["marker_list"]
                
                if cell_name not in combined_data:
                    combined_data[cell_name] = {"markers": []}
                
                for marker in marker_genes:
                    combined_data[cell_name]["markers"].append(marker)
    except Exception as e:
        print(f"Error accessing Neo4j: {e}")
        return {}
    finally:
        driver.close()
    
    return combined_data

def get_subtypes(cell_type):
    """Get subtypes of a given cell type using Neo4j."""
    uri = "bolt://localhost:7689"
    driver = GraphDatabase.driver(uri, auth=("neo4j", "37754262"))
    
    # Load specification for database info
    specification = None
    file_path = "media/specification_graph.json"
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            specification = json.load(file)
    
    database = specification['database']
    
    # Initialize result structure
    subtypes_data = {}
    
    try:
        with driver.session(database=database) as session:
            # Get subtypes and markers
            query = """
            MATCH (parent:CellType {name: $parent_cell})-[:DEVELOPS_TO]->(c:CellType)-[:HAS_MARKER]->(m:Marker)
            RETURN c.name as cell_name, m.markers as marker_list
            """
            result = session.run(query, parent_cell=cell_type)
            
            # Process results
            for record in result:
                subtype_name = record["cell_name"]
                marker_genes = record["marker_list"]
                
                if subtype_name not in subtypes_data:
                    subtypes_data[subtype_name] = {"markers": []}
          
                for marker in marker_genes:
                    subtypes_data[subtype_name]["markers"].append(marker)
    except Exception as e:
        print(f"Error accessing Neo4j: {e}")
        return {}
    finally:
        driver.close()
    
    return subtypes_data

def unified_cell_type_handler(cell_type):
    """
    Standardizes cell type names with proper handling for special cases.
    
    Parameters:
    -----------
    cell_type : str
        The cell type string to process
    
    Returns:
    --------
    str
        Standardized cell type name suitable for file paths and database matching
    """
    # Dictionary of known cell types with their standardized forms
    known_cell_types = {
        # One-word cell types (already plural, don't add "cells")
        "platelet": "Platelets",
        "platelets": "Platelets",
        "lymphocyte": "Lymphocytes",
        "lymphocytes": "Lymphocytes",
        
        # Three-word cell types (specific capitalization)
        "natural killer cell": "Natural killer cells",
        "natural killer cells": "Natural killer cells",
        "plasmacytoid dendritic cell": "Plasmacytoid dendritic cells",
        "plasmacytoid dendritic cells": "Plasmacytoid dendritic cells"
    }
    
    # Clean and normalize the input
    clean_type = cell_type.lower().strip()
    if clean_type.endswith(' cells'):
        clean_type = clean_type[:-6].strip()
    elif clean_type.endswith(' cell'):
        clean_type = clean_type[:-5].strip()
    
    # Check if it's a known cell type
    if clean_type in known_cell_types:
        return known_cell_types[clean_type]
    
    # Handle based on word count
    words = clean_type.split()
    if len(words) == 1:
        # Check if it's already plural
        if words[0].endswith('s') and not words[0].endswith('ss'):
            # Already plural, just return as is (e.g., "Platelets")
            return words[0].capitalize()
        else:
            # Add "cells" to singular forms (e.g., "Monocyte cells")
            return f"{words[0].capitalize()} cells"
    
    elif len(words) == 2:
        # Special case for cell types like "T cell", "B cell"
        special_first_words = ['t', 'b', 'nk', 'cd4', 'cd8']
        
        if words[0].lower() in special_first_words:
            # Preserve special capitalization
            return f"{words[0].upper()} cells"
        else:
            # Standard two-word handling
            return f"{words[0].capitalize()} {words[1].capitalize()} cells"
    
    elif len(words) >= 3:
        # For three or more words, only capitalize the first word
        return f"{words[0].capitalize()} {' '.join(words[1:])} cells"
    
    # Fallback
    return f"{cell_type} cells"

def standardize_cell_type(cell_type):
    """
    Standardize cell type strings for flexible matching.
    Handles multi-word cell types, singular/plural forms, and capitalization.
    """
    # Clean and normalize the input
    clean_type = cell_type.lower().strip()
    if clean_type.endswith(' cells'):
        clean_type = clean_type[:-6].strip()
    elif clean_type.endswith(' cell'):
        clean_type = clean_type[:-5].strip()
    
    # Return the standardized base form
    return clean_type

def get_possible_cell_types(cell_type):
    """
    Generate all possible forms of a cell type for flexible matching.
    """
    # Get standardized base form
    base_form = standardize_cell_type(cell_type)
    
    # Generate variations with proper capitalization
    result = unified_cell_type_handler(cell_type)
    
    # Create variations based on the correct standardized form
    words = base_form.split()
    possible_types = [base_form]  # Add the base form
    
    # Add the properly standardized result and its variants
    possible_types.append(result)
    
    # Add variations with and without "cell" or "cells"
    if not result.lower().endswith("cells"):
        possible_types.append(f"{result} cells")
    
    if len(words) == 1:
        # For single words, add with/without "s" variations
        if words[0].endswith('s'):
            possible_types.append(words[0][:-1])  # Without 's'
        else:
            possible_types.append(f"{words[0]}s")  # With 's'
    
    # Remove duplicates while preserving order
    return list(dict.fromkeys(possible_types))

def filter_existing_genes(adata, gene_list):
    """Filter genes to only those present in the dataset, handling non-unique indices."""
    if hasattr(adata, 'raw') and adata.raw is not None:
        # Use raw var_names if available
        var_names = adata.raw.var_names
    else:
        var_names = adata.var_names
        
    # Use isin() which handles non-unique indices properly
    existing_genes = [gene for gene in gene_list if gene in var_names]
    return existing_genes

def preprocess_data(adata, sample_mapping=None):
    """Preprocess the AnnData object with consistent steps."""
    # Ensure var_names are unique before processing
    if not adata.var_names.is_unique:
        print("Warning: Gene names are not unique. Making them unique.")
        adata.var_names_make_unique()
    
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
    
    # Ensure raw var_names are unique too
    if hasattr(adata, 'raw') and adata.raw is not None and not adata.raw.var_names.is_unique:
        print("Warning: Raw gene names are not unique. Making them unique.")
        adata.raw.var_names_make_unique()
    
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
        n_neighbors = min(15, int(0.5 * np.sqrt(adata.n_obs)))
        sc.pp.neighbors(adata, n_neighbors=n_neighbors)
    
    return adata

def perform_clustering(adata, resolution=2, random_state=42):
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
    
    # Check if there are enough cells in each group
    if len(adata.obs[groupby].unique()) <= 1:
        print(f"WARNING: Only one group found in {groupby}, cannot perform differential expression")
        # Create empty results to avoid errors
        adata.uns[key_added] = {
            'params': {'groupby': groupby},
            'names': np.zeros((1,), dtype=[('0', 'U50')])
        }
        return adata
    
    try:
        sc.tl.rank_genes_groups(adata, groupby, method=method, n_genes=n_genes, key_added=key_added, use_raw=False)
    except Exception as e:
        print(f"ERROR in rank_genes_groups: {e}")
        print("Falling back to t-test method")
        try:
            sc.tl.rank_genes_groups(adata, groupby, method='t-test', n_genes=n_genes, key_added=key_added, use_raw=False)
        except Exception as e2:
            print(f"ERROR with fallback method: {e2}")
            # Create empty results
            adata.uns[key_added] = {
                'params': {'groupby': groupby},
                'names': np.zeros((1,), dtype=[('0', 'U50')])
            }
    
    return adata

def create_marker_anndata(adata, markers, copy_uns=True, copy_obsm=True):
    """Create a copy of AnnData with only marker genes."""
    # Filter markers to those present in the dataset
    markers = filter_existing_genes(adata, markers)
    markers = list(set(markers))
    
    # Check if we have any markers at all
    if len(markers) == 0:
        print("WARNING: No marker genes found in the dataset!")
        # Return a minimal valid AnnData to avoid errors
        return anndata.AnnData(
            X=scipy.sparse.csr_matrix((adata.n_obs, 0)),
            obs=adata.obs.copy()
        ), []
    
    # Create a new AnnData object with log-transformed data
    if hasattr(adata, 'raw') and adata.raw is not None:
        # Get indices of marker genes in the raw data
        raw_indices = [i for i, name in enumerate(adata.raw.var_names) if name in markers]
        
        # IMPORTANT: Use log-transformed layer if available, otherwise log-transform the raw data
        if hasattr(adata.raw, 'layers') and 'log1p' in adata.raw.layers:
            X = adata.raw.layers['log1p'][:, raw_indices].copy()
        else:
            # Get raw counts
            X = adata.raw.X[:, raw_indices].copy()
            # Log-transform if needed (check if data appears to be counts)
            if scipy.sparse.issparse(X):
                max_val = X.max()
            else:
                max_val = np.max(X)
            if max_val > 100:  # Heuristic to detect if data is not log-transformed
                print("Log-transforming marker data...")
                X = np.log1p(X)
    else:
        # Using main data
        main_indices = [i for i, name in enumerate(adata.var_names) if name in markers]
        X = adata.X[:, main_indices].copy()
    
    # Create the new AnnData object
    var = adata.var.iloc[main_indices].copy() if 'main_indices' in locals() else adata.raw.var.iloc[raw_indices].copy()
    marker_adata = anndata.AnnData(
        X=X,
        obs=adata.obs.copy(),
        var=var
    )
    
    # Copy additional data
    if copy_uns:
        marker_adata.uns = adata.uns.copy()
    
    if copy_obsm:
        marker_adata.obsm = adata.obsm.copy()
    
    if hasattr(adata, 'obsp'):
        marker_adata.obsp = adata.obsp.copy()
    
    return marker_adata, markers

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
def generate_umap(resolution=2):
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
    # sc.tl.pca(adata, svd_solver='arpack')
    
    # Perform clustering
    adata = perform_clustering(adata, resolution=resolution)
    
    # Rank all genes
    adata = rank_genes(adata, groupby='leiden', n_genes=25, key_added='rank_genes_all')
    
    # Get markers and create marker-specific AnnData
    markers = get_rag()
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
        resolution = 1  # Default higher resolution for subtype clustering
    
    # Get all possible variations of the cell type for matching
    possible_types = get_possible_cell_types(cell_type)
    standardized_name = unified_cell_type_handler(cell_type)
    
    # Filter cells based on cell type with flexible matching
    mask = adata.obs['cell_type'].isin(possible_types)
    if mask.sum() == 0:
        print(f"WARNING: No cells found with cell type matching any of these variations: {possible_types}")
        return None, None, None
    
    filtered_cells = adata[mask].copy()
    
    # Perform new clustering on filtered cells
    sc.tl.pca(filtered_cells, svd_solver='arpack')
    sc.pp.neighbors(filtered_cells)
    filtered_cells = perform_clustering(filtered_cells, resolution=resolution)
    
    # Create ranking key based on cell type
    # Use standardized base form for key naming
    base_form = standardize_cell_type(cell_type).replace(' ', '_').lower()
    rank_key = f"rank_genes_{base_form}"
    
    # Rank genes for the filtered cells
    filtered_cells = rank_genes(filtered_cells, key_added=rank_key)
    
    # Get markers and create marker-specific sub-AnnData if needed
    markers = get_subtypes(cell_type)
    markers_tree = markers.copy()
    markers_list = extract_genes(markers)
    
    # Check if we have enough markers
    if not markers_list:
        print(f"WARNING: No marker genes found for {standardized_name}. Using general ranking.")
        # Use general ranking instead
        top_genes_df = rank_ordering(filtered_cells, key=rank_key, n_genes=25)
        
        # Create gene dictionary
        gene_dict = {}
        for cluster, group in top_genes_df.groupby("cluster"):
            gene_dict[cluster] = list(group["gene"])
        
        # Create dendrogram
        sc.tl.dendrogram(filtered_cells, groupby='leiden')
        
        # Save data
        save_analysis_results(
            filtered_cells,
            prefix=f"process_cell_data/{standardized_name}",
            save_dotplot=False
        )
        
        # Save processed data
        umap_df = filtered_cells.obs[['UMAP_1', 'UMAP_2', 'leiden', 'cell_type']]
        if 'patient_name' in filtered_cells.obs.columns:
            umap_df['patient_name'] = filtered_cells.obs['patient_name']
            
        fname = f'{standardized_name}_adata_processed.pkl'
        with open(fname, 'wb') as file:
            pickle.dump(umap_df, file)
        
        return gene_dict, filtered_cells, markers_tree
    
    # Rest of the function remains the same, but using standardized_name instead of 
    # the previous f"{cell_type_cap} cells" format for file paths and variable names
    
    # Create marker-specific AnnData
    filtered_markers, marker_list = create_marker_anndata(filtered_cells, markers_list)
    
    # Skip marker-specific ranking if fewer than 5 markers were found
    if len(marker_list) < 5:
        print(f"WARNING: Only {len(marker_list)} marker genes found. Using general gene ranking.")
        # Use general ranking instead
        top_genes_df = rank_ordering(filtered_cells, key=rank_key, n_genes=25)
    else:
        try:
            # Rank marker genes
            marker_key = f"rank_markers_{base_form}"
            filtered_markers = rank_genes(filtered_markers, key_added=marker_key)
            
            # Copy marker ranking to filtered dataset
            filtered_cells.uns[marker_key] = filtered_markers.uns[marker_key]
            
            # Extract top genes with preference for marker-specific ranking
            top_genes_df = rank_ordering(filtered_markers, key=marker_key, n_genes=25)
        except Exception as e:
            print(f"ERROR in marker-specific ranking: {e}")
            print("Falling back to general ranking")
            # Use general ranking in case of any error
            top_genes_df = rank_ordering(filtered_cells, key=rank_key, n_genes=25)
    
    # Create gene dictionary
    gene_dict = {}
    for cluster, group in top_genes_df.groupby("cluster"):
        gene_dict[cluster] = list(group["gene"])
    
    # Create dendrogram
    sc.tl.dendrogram(filtered_cells, groupby='leiden')
    
    # Save data
    save_analysis_results(
        filtered_cells,
        prefix=f"process_cell_data/{standardized_name}",
        save_dotplot=len(marker_list) >= 5,
        markers=marker_list
    )
    
    # Save processed data
    umap_df = filtered_cells.obs[['UMAP_1', 'UMAP_2', 'leiden', 'cell_type']]
    if 'patient_name' in filtered_cells.obs.columns:
        umap_df['patient_name'] = filtered_cells.obs['patient_name']
        
    fname = f'{standardized_name}_adata_processed.pkl'
    with open(fname, 'wb') as file:
        pickle.dump(umap_df, file)
    
    return gene_dict, filtered_cells, markers_tree

def label_clusters(annotation_result, cell_type, adata):
    """Label clusters with consistent cell type handling."""
    # Standardize cell type names
    standardized_name = unified_cell_type_handler(cell_type)
    base_form = standardize_cell_type(cell_type).lower()
    
    try:
        adata = adata.copy()
        
        # Parse annotation mapping
        start_idx = annotation_result.find("{")
        end_idx = annotation_result.rfind("}") + 1
        str_map = annotation_result[start_idx:end_idx]
        map2 = ast.literal_eval(str_map)
        map2 = {str(key): value for key, value in map2.items()}
        
        # Apply annotations - different handling for overall cells vs specific cell types
        if base_form == "overall":
            # For overall cells, directly apply annotations to the main dataset
            adata.obs['cell_type'] = 'Unknown'
            for group, cell_type_value in map2.items():
                adata.obs.loc[adata.obs['leiden'] == group, 'cell_type'] = cell_type_value
                
            # Save annotated data
            save_analysis_results(
                adata,
                prefix=f"umaps/{standardized_name}",
                save_dendrogram=False,
                save_dotplot=False
            )
            
            # Save annotated adata
            fname = f'annotated_adata/{standardized_name}_annotated_adata.pkl'
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
                prefix=f"umaps/{standardized_name}",
                save_dendrogram=False,
                save_dotplot=False
            )
            
            # Save annotated adata
            fname = f'annotated_adata/{standardized_name}_annotated_adata.pkl'
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
    clear_directory("figures") 
    # gene_dict, marker_tree, adata = generate_umap()  
    # pd.to_pickle(marker_tree, "marker_tree.pkl")
    # pd.to_pickle(gene_dict, "gene_dict.pkl")

    # input_msg =(f"Top genes details: {gene_dict}. "
    #             f"Markers: {marker_tree}. ")
    
    # messages = [
    #     SystemMessage(content="""
    #         You are a bioinformatics researcher that can do the cell annotation.
    #         The following are the data and its decription that you will receive.
    #         * 1. Gene list: top 25 cells arragned from the highest to lowest expression levels in each cluster.
    #         * 2. Marker tree: marker genes that can help annotating cell type.           
            
    #         Identify the cell type for each cluster using the following markers in the marker tree.
    #         This means you will have to use the markers to annotate the cell type in the gene list. 
    #         Provide your result as the most specific cell type that is possible to be determined.
            
    #         Provide your output in the following example format:
    #         Analysis: group_to_cell_type = {'0': 'Cell type','1': 'Cell type','2': 'Cell type','3': 'Cell type','4': 'Cell type', ...}.
            
    #         Strictly adhere to follwing rules:
    #         * 1. Adhere to the dictionary format and do not include any additional words or explanations.
    #         * 2. The cluster number in the result dictionary must be arranged with raising power.
    #         """),
    #     HumanMessage(content=input_msg)
    # ]
    # model = ChatOpenAI(
    #     model="gpt-4o",
    #     temperature=0
    # )
    # results = model.invoke(messages)
    # annotation_result = results.content
    # print(annotation_result)


    # cell_type = "Overall cells"
    # adata = label_clusters(annotation_result, cell_type, adata)
    # print(adata.obs["cell_type"])
    # display_umap(cell_type)
    # display_processed_umap(cell_type)
    # pd.to_pickle(marker_tree, "marker_tree.pkl")
    # pd.to_pickle(gene_dict, "gene_dict.pkl")
    # pd.to_pickle(adata, "adata.pkl")

    # print("\n" + "="*50 + "\n")
    # print("Start to do ")
    # print("\n" + "="*50 + "\n")
    
    annotation_result = ['Platelets', 'Plasmacytoid dendritic cells', 'Natural killer cells']
    
    for cell_type in annotation_result:
        with open("gene_dict.pkl", "rb") as file:
            gene_dict = pd.read_pickle(file)
        with open("marker_tree.pkl", "rb") as file:
            marker_tree = pd.read_pickle(file)    
        with open("adata.pkl", "rb") as file:
            adata = pd.read_pickle(file)
        gene_dict, adata, marker_tree = process_cells(adata, cell_type)
        
        model = ChatOpenAI(
            model="gpt-4o",
            temperature=0
        )
        input_msg =(f"Top genes details: {gene_dict}. "
                    f"Markers: {marker_tree}. ")
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
        annotation_result = results.content
        adata = label_clusters(annotation_result, cell_type, adata)
        # display_umap(spec_cell_type)
        # display_processed_umap(spec_cell_type)
        
        adata.obs["cell_type"].to_csv(f"{cell_type}_result.csv")
    