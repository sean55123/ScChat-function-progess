import pickle 
import json
import os
import time
import re
import openai
from functions.Tools_reasoning import function_descriptions
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
from langsmith import utils
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
from langgraph.func import entrypoint, task
from langgraph.graph import add_messages
from typing import TypedDict, Dict, List, Any
import ast

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
    global_top_genes_df, global_mean_expression, global_expression_proportion = calculate_cluster_statistics(adata, "overall")
    
    agent_content = """
        You are a bioinformatics researcher that recognizes the relationship between a cell and its subsets basing on the provide information. 
        As you receive input messages, you will record the cells and their markers' genes, as well as their subsets, and turn them into a readable format. 

        Purpose and Goals:
        * Accurately identify and record cells, their marker genes, and their subsets from input messages.
        * Organize the recorded information into a clear and readable format.
        * A clear relationship between root cell (the ancestor) and their subsets (child) should be provided.
        * Do not include any additional words or explanations.
        """

    # Annotation_results = openai.chat.completions.create(
    #     model="gpt-4o",
    #     messages=[
    #         {"role": "system", "content": agent_content},
    #         {"role": "user", "content": str(marker_tree)},
    #     ],
    # )
    # marker_tree =  Annotation_results.choices[0].message.content
    
    gene_dict = {}
    for cluster, group in global_top_genes_df.groupby("cluster"):
        gene_dict[cluster] = list(group["gene"])
        
    # summary2 = (f"Top genes details: {gene_dict}. "
    #             f"markers_tree: {str(marker_tree)}. ")
    # final_summary = summary2
    return gene_dict, marker_tree

if __name__ == "__main__":
    class State(TypedDict):
        Genedict: Dict[str, List[str]]  
        Marker_tree: Dict[str, List[str]]  
        Annotation_results: Dict[str, str]
        Non_leaf_clusters: Dict[str, str]
        Leaf_clusters: Dict[str, str]
        Non_leaf_genes: Dict[str, List[str]]
        Leaf_counter: int
        
    model = ChatOpenAI(model="gpt-4o", temperature=0)
    
    @task()
    def data_preprocess(state: State):
        # gene_dict, marker_tree = generate_umap()
        with open("test_gene_dict.pkl", "rb") as file:
            gene_dict = pd.read_pickle(file)
        marker_tree = get_rag_and_markers(False)
        state['Genedict'] = gene_dict
        state['Marker_tree'] = marker_tree
        return state
    
    @entrypoint()
    def annotation_agent(state: State):
        state['Leaf_counter'] = 0
        state = data_preprocess(state)
        print(state['Genedict'].result().values())
        time.sleep(20)
        gene_dict = state['Genedict']
        
        maker_tree = state['Marker_tree']
        input_msg = (f"Top genes details: {gene_dict}. "
                     f"markers_tree: {maker_tree}. ")
        input_msg = str(input_msg)
        messages = [
            SystemMessage(content="""
                You are a bioinformatics researcher that can do the cell annotation.
                Identify the cell type for each cluster using the following markers in the marker tree, arranged from highest to lowest expression levels. 
                This means you should consider the first several markers in the list to be more important. 
                Provide your result as the most specific cell type that is possible to be determined.
                
                Markers: <marker_tree>
                Provide your output in the following example format:
                Analysis: group_to_cell_type = {'0': 'Cell type','1': 'Cell type','2': 'Cell type','3': 'Cell type','4': 'Cell type', ...}.
                
                Strictly adhere to follwing rules:
                * 1. Adhere to the dictionary format and do not include any additional words or explanations.
                * 2. The cluster number in the result dictionary must be arranged with raising power.
                """),
            HumanMessage(content=input_msg)
        ]
        msg = model.invoke(messages)
        state['Annotation_results'] = msg.content
        return state

    annotation_agent.name = "annotation_agent"
    
    @task()
    def leaf_checker(state: State):
        state['Leaf_counter'] += 1
        annotation_results = state['Annotation_results']
        marker_tree = state['Marker_tree']
        gene_dict = state['Genedict']
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

        state['Non_leaf_clusters'] = non_leaf_clusters
        state['Leaf_clusters'] = leaf_clusters
        state['Non_leaf_genes'] = non_leaf_gene
        return state
    
    @entrypoint()    
    def leaf_search_agent(state: State):
        leaf_checker(state)
        input_msg = (f"Marker tree: {state['Marker_tree']}",
                     f"Awaiting cell: {state['Non_leaf_clusters']}",
                     f"Gene list: {state['Non_leaf_genes']}")
        input_msg = str(input_msg)
        messages = [
            SystemMessage(content="""
                You are a bioinformatics researcher that can do cell annotation basing on the given information.
                The following is the data you will receive and its description,
                1. Marker tree: marker genes that can help annotating cell tyep. 
                2. Awaiting list: a cluster paired with a leaf cells required further annotation.
                3. Gene list: top 25 genes for the corresponding cluster in the awaiting list.
                
                Notation explanation:
                We call the cell that belong to any of the subsets "root cells".
                The childs of root cell are all called leaf cells.
                Leaf cell without any subsets in it called "end leaf cells".
                
                The main goal is to figure out if the cells in the awaiting list can be further recognized as their leaf cell.
                This means you need to do cell annotation to the clusters in the awaiting list, and check if they should be annotated to their leaf cell or stay in current solution.
                
                Markers: <marker_tree>
                Provide your output in the following example format:
                Analysis: group_to_cell_type = {'0': 'Cell type','1': 'Cell type','2': 'Cell type','3': 'Cell type','4': 'Cell type', ...}.
                
                Strictly adhere to follwing rules:
                * 1. Adhere to the dictionary format and do not include any additional words or explanations.
                * 2. The cluster number in the result dictionary must be arranged with raising power.
                """),
            HumanMessage(content=input_msg)
        ]
        msg = model.invoke(messages)
        new_leaf = msg.content.split("=", 1)[1].strip().rstrip(".")
        new_leaf = {**new_leaf, **state['Leaf_clusters']}
        new_leaf = {k: v for k, v in sorted(new_leaf.items(), key=lambda item: int(item[0]))}
        new_leaf = f"Analysis: group_to_cell_type = {new_leaf}"
        state['Annotation_results'] = new_leaf
        return state

    leaf_search_agent.name = "leaf_search_agent"
       
    annotation_supervisor = create_supervisor(
        agents=[annotation_agent, leaf_search_agent],
        model=model,
        prompt="""
        You are a bioinformatics researcher also a leader of cell annotation group.
        The following are the agents and their descriptions that are assigned to you.
        annotation_agent: this agent is really helpful when dealing with data preprocessing and cell annotation.
        leaf_search_agent: this agent can do leaf searching to make the results from annotation agent more precise.
        
        Workflow:
        *1. Once you are asked to assist to do cell annotation, use annotation agent.
        *2. As the cell preprocess and cell annotation is finished, use leaf_search_agent to do leaf search repeatly as it meet the threshold.
        *3. The threshold for leaf search is as the value state['Leaf_counter'] equals to 2.
            This means you have to check the value of state['Leaf_counter] everytime before you use leaf_search_agent.
        *4. As the threshold is met, retrieve the result from state['Annotation_results'].
        
        The final results should contain the information from state['Annotation_results'] and do not include any additional words or explanations.
        """
    )
    
    annotation_group = annotation_supervisor.compile()
    result = annotation_group.invoke({
        "messages": [{
            "role": "user",
            "content": "Help me do the cell annotation"
        }]
    })
    
    for m in result["messages"]:
        m.pretty_print()