import os
import json
import openai
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import scanpy as sc
import pandas as pd
import pickle
import scvi
from scvi.model import SCVI
import plotly.express as px
import plotly.graph_objects as go
import ast
import matplotlib
import warnings
import numpy as np
import shutil
import regex as re
import gseapy as gp
import base64
import requests
from Tools import function_descriptions

# Set basic configuration
sc.set_figure_params(dpi=100)
scvi._settings.seed = 42
sc.settings.seed = 42
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

class ScChat:
    def __init__(self):
        load_dotenv()
        openai.api_key = ""
        # State variables (previously global)
        self.current_adata = None
        self.base_annotated_adata = None
        self.adata = None
        self.resolution = 0.5
        self.sample_mapping = None
        self.SGP = None
        self.first_try = True
        self.clear_data = True
        self.display_flag = False
        self.function_flag = False
        self.cluster_summary = {}
        # For storing intermediate computed statistics
        self.global_top_genes_df = None
        self.global_mean_expression = None
        self.global_expression_proportion = None
        # For storing GSEA output data
        self.out_df = None

    # ====================================================================================
    #                                    Utility methods
    # ====================================================================================
    def clear_directory(self, directory_path):
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

    def find_and_load_sample_mapping(self, directory):
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file == 'sample_mapping.json':
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r') as f:
                        self.sample_mapping = json.load(f)
                    # print(f"'sample_mapping.json' found and loaded from {file_path}")
                    return self.sample_mapping
        # If the file wasn't found
        return None
    
    def find_file_with_extension(self, directory_path, extension):
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.endswith(extension):
                    return os.path.join(root, file)
        return None
    
    def get_rag_and_markers(self, tag):
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
                    if cell_type not in combined_data:
                        combined_data[cell_type] = {'genes': []}
                    combined_data[cell_type]['genes'].extend([marker['gene'] for marker in cell_data['markers']])
        with open("testop.txt", "w") as fptr:
            fptr.write(json.dumps(combined_data, indent=4))
        return combined_data
    
    def filter_existing_genes(self, adata, gene_list):
        existing_genes = [gene for gene in gene_list if gene in adata.raw.var_names]
        return existing_genes

    # ====================================================================================
    #                                    Visualization
    # ====================================================================================
    def display_cell_population_change(self, cell_type):
        # Load the CSV file
        self.display_flag = True
        cell_type = cell_type.split()[0].capitalize() + " cells"
        filename = f'schatbot/cell_population_change/{cell_type}_cell_population_change.csv'
        cell_counts = pd.read_csv(filename)
        # Create the Plotly plot
        fig = px.bar(
            cell_counts,
            x="patient_name",
            y="percentage",
            color="cell_type",
            title=f"Cell Population Change",
            labels={"patient_name": "Patient Name", "percentage": "Percentage of Cell Type"}
        )
        # Update layout for better visualization
        fig.update_layout(
            width=1200,  # Set the width of the plot
            height=800,  # Set the height of the plot
            autosize=True,
            showlegend=True  # Show the legend
        )
        # Convert the plot to JSON for frontend display
        fig_json = fig.to_json()
        return fig_json

    def display_umap(self, cell_type):
        self.display_flag = True
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
            title="T Cells UMAP Plot",
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
        fig_json = fig.to_json()
        return fig_json

    def display_processed_umap(self, cell_type):
        self.display_flag = True
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
        fig_json = fig.to_json()
        return fig_json

    def display_dotplot(self):
        dot_plot_data = pd.read_csv("basic_data/dot_plot_data.csv")
        fig = px.scatter(
            dot_plot_data,
            x='gene',
            y='leiden',
            size='expression',
            color='expression',
            title='Dot Plot',
            labels={'gene': 'Gene', 'leiden': 'Cluster', 'expression': 'Expression Level'},
            color_continuous_scale='Blues'
        )
        fig.update_traces(marker=dict(opacity=0.8))
        fig.update_layout(
            width=1200,
            height=800,
            autosize=True,
        )
        return fig.to_json()

    def display_gsea_dotplot(self):
        # Manually define the correct column names based on your CSV's structure
        column_names = ['Term', 'fdr', 'es', 'nes', 'Rank Metric', 'Enrichment Score']
        # Load the data with the correct column names
        dot_plot_data = pd.read_csv("gsea_plots/gsea_plot_data.csv", header=None, names=column_names, skiprows=1)
        # Drop rows with missing values in 'Rank Metric' and 'Enrichment Score'
        dot_plot_data = dot_plot_data.dropna(subset=['Rank Metric', 'Enrichment Score'])
        # Convert necessary columns to numeric types
        dot_plot_data['Rank Metric'] = pd.to_numeric(dot_plot_data['Rank Metric'])
        dot_plot_data['Enrichment Score'] = pd.to_numeric(dot_plot_data['Enrichment Score'])
        # Filter out invalid sizes (non-positive numbers) or set a minimum size value
        dot_plot_data['Enrichment Score'] = dot_plot_data['Enrichment Score'].apply(lambda x: max(abs(x), 1))
        fig = px.scatter(
            dot_plot_data,
            x='Rank Metric',
            y='Enrichment Score',
            size='Enrichment Score',
            color='Enrichment Score',
            title='GSEA Dot Plot',
            labels={'Rank Metric': 'Rank Metric', 'Enrichment Score': 'Enrichment Score'},
            color_continuous_scale='Blues'
        )
        fig.update_traces(marker=dict(opacity=0.8))
        fig.update_layout(
            width=1200,
            height=800,
            autosize=True,
            xaxis=dict(title="Rank Metric"),
            yaxis=dict(title="Enrichment Score")
        )
        return fig.to_json()

    def display_cell_type_composition(self):
        import plotly.figure_factory as ff
        dendrogram_data = pd.read_csv("basic_data/dendrogram_data.csv")
        fig = ff.create_dendrogram(dendrogram_data.values, orientation='left')
        fig.update_layout(title='Dendrogram', xaxis_title='Distance', yaxis_title='Clusters')
        fig.update_layout(
            width=1200,
            height=800,
            autosize=True,
        )
        return fig.to_json()


    # ====================================================================================
    #                   Preprocessing - annotation & statistics calculation
    # ====================================================================================
    def generate_umap(self):
        matplotlib.use('Agg')
        path = self.find_file_with_extension("media", ".h5ad")
        if not path:
            return ".h5ad file isn't given, unable to generate UMAP."
        self.adata = sc.read_h5ad(path)
        self.current_adata = self.adata
        # Data preprocessing
        sc.pp.filter_cells(self.adata, min_genes=100)
        sc.pp.filter_genes(self.adata, min_cells=3)
        self.adata.var['mt'] = self.adata.var_names.str.startswith('MT-')
        sc.pp.calculate_qc_metrics(self.adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
        self.adata = self.adata[self.adata.obs.pct_counts_mt < 20]
        self.adata.layers['counts'] = self.adata.X.copy()  # used by scVI-tools
        # Normalization
        sc.pp.normalize_total(self.adata, target_sum=1e4)
        sc.pp.log1p(self.adata)
        self.adata.raw = self.adata
        self.find_and_load_sample_mapping("media")
        if self.sample_mapping:
            sc.pp.highly_variable_genes(self.adata, n_top_genes=3000, subset=True, layer='counts', flavor="seurat_v3", batch_key="Sample")
        else:
            sc.pp.highly_variable_genes(self.adata, n_top_genes=3000, subset=True, layer='counts', flavor="seurat_v3")
        if self.sample_mapping:
            SCVI.setup_anndata(self.adata, layer="counts", categorical_covariate_keys=["Sample"],
                                 continuous_covariate_keys=['pct_counts_mt', 'total_counts'])
            model = SCVI.load(dir_path="schatbot/glioma_scvi_model", adata=self.adata)
            latent = model.get_latent_representation()
            self.adata.obsm['X_scVI'] = latent
            self.adata.layers['scvi_normalized'] = model.get_normalized_expression(library_size=1e4)
        if self.sample_mapping:
            sc.pp.neighbors(self.adata, use_rep='X_scVI')
        else:
            sc.pp.neighbors(self.adata)
        sc.tl.umap(self.adata, random_state=42)
        sc.tl.leiden(self.adata, resolution=self.resolution, random_state=42)
        if self.sample_mapping:
            self.adata.obs['patient_name'] = self.adata.obs['Sample'].map(self.sample_mapping)
        umap_df = self.adata.obsm['X_umap']
        self.adata.obs['UMAP_1'] = umap_df[:, 0]
        self.adata.obs['UMAP_2'] = umap_df[:, 1]
        base_markers = self.get_rag_and_markers(False)
        markers = []
        for cell_type, cell_data in base_markers.items():
            markers += cell_data['genes']
        markers = self.filter_existing_genes(self.adata, markers)
        markers = list(set(markers))
        statistic_data = sc.get.obs_df(self.adata, keys=['leiden'] + markers, use_raw=True)
        statistic_data.set_index('leiden', inplace=True)
        pd_mean_expression = pd.DataFrame(statistic_data.groupby('leiden').mean())
        pd_mean_expression.to_csv("basic_data/mean_expression.csv")
        pd_mean_expression.to_json("basic_data/mean_expression.json")
        expression_proportion = statistic_data.gt(0).groupby('leiden').mean()
        pd_expression_proportion = pd.DataFrame(expression_proportion)
        pd_expression_proportion.to_csv("basic_data/expression_proportion.csv")
        pd_expression_proportion.to_json("basic_data/expression_proportion.json")
        if self.sample_mapping is None:
            sc.tl.dendrogram(self.adata, groupby='leiden')
            with plt.rc_context({'figure.figsize': (10, 10)}):
                sc.pl.dotplot(self.adata, markers, groupby='leiden', swap_axes=True, use_raw=True,
                              standard_scale='var', dendrogram=True, color_map="Blues", save="dotplot.png")
                plt.close()
        else:
            sc.tl.dendrogram(self.adata, groupby='leiden', use_rep='X_scVI')
            with plt.rc_context({'figure.figsize': (10, 10)}):
                sc.pl.dotplot(self.adata, markers, groupby='leiden', swap_axes=True, use_raw=True,
                              standard_scale='var', dendrogram=True, color_map="Blues", save="dotplot.png")
                plt.close()
        self.adata.obs['cell_type'] = 'Unknown'
        self.adata.obs[['UMAP_1', 'UMAP_2', 'leiden', 'patient_name']].to_csv("basic_data/Overall cells_umap_data.csv", index=False)
        self.adata.obs[['UMAP_1', 'UMAP_2', 'leiden', 'patient_name']].to_csv("process_cell_data/Overall cells_umap_data.csv", index=False)
        dot_plot_data = statistic_data.reset_index().melt(id_vars='leiden', var_name='gene', value_name='expression')
        dot_plot_data.to_csv("basic_data/dot_plot_data.csv", index=False)
        dendrogram_data = self.adata.uns['dendrogram_leiden']
        pd_dendrogram_linkage = pd.DataFrame(dendrogram_data['linkage'], columns=['source', 'target', 'distance', 'count'])
        pd_dendrogram_linkage.to_csv("basic_data/dendrogram_data.csv", index=False)
        rag_data = self.get_rag_and_markers(True)
        summary = f"UMAP analysis completed. Data summary: {self.adata}, " \
                    f"RAG Data : {str(rag_data)}. " \
                f"Cell counts details are provided. " \
                f"Additional data file generated: preface.txt."
        retrieve_stats_summary = self.retrieve_stats()
        final_summary = f"{summary} {retrieve_stats_summary}"
        self.current_adata = self.adata
        return final_summary

    def label_clusters(self, cell_type):
        # Copy the current adata to work on
        adata2 = self.adata.copy()
        standardized_cell_type3 = cell_type.split()[0].capitalize()
        standardized_cell_type2 = cell_type.split()[0].capitalize() + " cell"
        standardized_cell_type = cell_type.split()[0].capitalize() + " cells"
        try:
            start_idx = self.cluster_summary[cell_type].find("{")
            end_idx = self.cluster_summary[cell_type].rfind("}") + 1
            str_map = self.cluster_summary[cell_type][start_idx:end_idx]
            map2 = ast.literal_eval(str_map)
            map2 = {str(key): value for key, value in map2.items()}
            if standardized_cell_type == "Overall cells" or standardized_cell_type2 == "Overall cell":
                adata2.obs['cell_type'] = 'Unknown'
                for group, cell_type in map2.items():
                    adata2.obs.loc[adata2.obs['leiden'] == group, 'cell_type'] = cell_type
                adata2.obs[['UMAP_1', 'UMAP_2', 'leiden', 'patient_name', 'cell_type']].to_csv(
                    f'umaps/{standardized_cell_type}_annotated_umap_data.csv', index=False)
                self.annotated_adata = adata2.copy()
                fname = f'annotated_adata/Overall cells_annotated_adata.pkl'
                with open(fname, "wb") as file:
                    pickle.dump(self.annotated_adata, file)
                self.base_annotated_adata = adata2
            else:
                adata3 = self.base_annotated_adata.copy()
                specific_cells = adata3[adata3.obs['cell_type'].isin([standardized_cell_type])].copy()
                if specific_cells.shape[0] == 0:
                    specific_cells = adata3[adata3.obs['cell_type'].isin([standardized_cell_type2])].copy()
                if specific_cells.shape[0] == 0:
                    specific_cells = adata3[adata3.obs['cell_type'].isin([standardized_cell_type3])].copy()
                sc.tl.pca(specific_cells, svd_solver='arpack')
                sc.pp.neighbors(specific_cells, random_state=42)
                sc.tl.umap(specific_cells, random_state=42)
                sc.tl.leiden(specific_cells, resolution=self.resolution, random_state=42)
                specific_cells.obs['cell_type'] = 'Unknown'
                for group, cell_type in map2.items():
                    specific_cells.obs.loc[specific_cells.obs['leiden'] == group, 'cell_type'] = cell_type
                specific_cells.obs[['UMAP_1', 'UMAP_2', 'leiden', 'patient_name', 'cell_type']].to_csv(
                    f'umaps/{standardized_cell_type}_annotated_umap_data.csv', index=False)
                self.annotated_adata = specific_cells.copy()
                fname = f'annotated_adata/{standardized_cell_type}_annotated_adata.pkl'
                with open(fname, "wb") as file:
                    pickle.dump(self.annotated_adata, file)
        except (SyntaxError, ValueError) as e:
            print(f"Error in parsing the map: {e}")
        return "Repeat 'Annotation of clusters is complete'"

    def extract_top_genes_stats(self, adata, groupby='leiden', n_genes=25):
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

    def calculate_cluster_statistics(self, adata, category, n_genes=25):
        # Adding
        base_markers = self.get_rag_and_markers(False)
        markers = []
        for cell_type, cell_data in base_markers.items():
            markers += cell_data['genes']
        markers = self.filter_existing_genes(adata, markers)
        markers = list(set(markers))
        sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon', n_genes=n_genes)
        top_genes_df = self.extract_top_genes_stats(adata, groupby='leiden', n_genes=25)
        if self.sample_mapping:
            sc.tl.dendrogram(adata, groupby='leiden', use_rep='X_scVI')
        else:
            sc.tl.dendrogram(adata, groupby='leiden')
        marker_expression = sc.get.obs_df(adata, keys=['leiden'] + markers, use_raw=True)
        marker_expression.set_index('leiden', inplace=True)
        mean_expression = marker_expression.groupby('leiden').mean()
        expression_proportion = marker_expression.gt(0).groupby('leiden').mean()
        self.global_top_genes_df = top_genes_df
        self.global_mean_expression = mean_expression
        self.global_expression_proportion = expression_proportion
        return self.global_top_genes_df, self.global_mean_expression, self.global_expression_proportion

    def retrieve_stats(self):
        with open("basic_data/mean_expression.json", 'r') as file:
            mean_expression = json.load(file)
        with open("basic_data/expression_proportion.json", 'r') as file:
            expression_proportion = json.load(file)
        self.calculate_cluster_statistics(self.adata, 'overall')
        markers = self.get_rag_and_markers(False)
        markers = ', '.join(markers)
        explanation = "Please analyze the clustering statistics and classify each cluster based on the following data: Top Genes:Mean Expression: Expression Proportion: , based on statistical data: 1. top_genes_df: 25 top genes expression within each clusters, with it's p_val, p_val_adj, and logfoldchange; 2. mean_expression of the marker genes: specific marker genes mean expression within each cluster; 3. expression_proportion of the marker genes: every cluster each gene expression fraction within each cluster, and give back the mapping dictionary in the format like this group_to_cell_type = {'0': 'Myeloid cells','1': 'T cells','2': 'Myeloid cells','3': 'Myeloid cells','4': 'T cells'} without further explanation or comment.  I only want the summary map in the response, do not give me any explanation or comment or repeat my input, i dont want any redundant information other than the summary map"
        mean_expression_str = ", ".join([f"{k}: {v}" for k, v in mean_expression.items()])
        expression_proportion_str = ", ".join([f"{k}: {v}" for k, v in expression_proportion.items()])
        summary = (f"Explanation: {explanation}. "
                   f"Mean expression data: {mean_expression_str}. "
                   f"Expression proportion data: {expression_proportion_str}. "
                   f"Top genes details: {self.global_top_genes_df}. "
                   f"markers: {markers}. ")
        return summary

    # ====================================================================================
    #                              Specific cell type processing
    # ====================================================================================
    def process_cells(self, cell_type):
        self.processed_cell_name = cell_type
        adata2 = self.base_annotated_adata.copy()
        cell_type_split = cell_type.split()[0]
        cell_type_cap = cell_type_split.capitalize()
        cell_type_full = cell_type_cap + " cells"
        cell_type_alt = cell_type_cap + " cell"
        filtered_cells = adata2[adata2.obs['cell_type'].isin([cell_type_full])].copy()
        if filtered_cells.shape[0] == 0:
            filtered_cells = adata2[adata2.obs['cell_type'].isin([cell_type_alt])].copy()
        if filtered_cells.shape[0] == 0:
            filtered_cells = adata2[adata2.obs['cell_type'].isin([cell_type_cap])].copy()
        sc.tl.pca(filtered_cells, svd_solver='arpack')
        sc.pp.neighbors(filtered_cells)
        sc.tl.umap(filtered_cells)
        sc.tl.leiden(filtered_cells, resolution=self.resolution)
        umap_data = filtered_cells.obsm['X_umap']
        umap_df = pd.DataFrame(umap_data, columns=['UMAP_1', 'UMAP_2'])
        umap_df['cell_type'] = filtered_cells.obs['cell_type'].values
        umap_df['patient_name'] = filtered_cells.obs['patient_name'].values
        umap_df['leiden'] = filtered_cells.obs['leiden'].values
        umap_df.to_csv(f'process_cell_data/{cell_type_full}_umap_data.csv', index=False)
        explanation = (f"Please analyze the {cell_type_full} clustering statistics and classify each cluster based on the following data into more in depth cell types, "
                       f"based on statistical data: we prepare 1. {cell_type_full}_top_genes_df: 25 top genes expression within each clusters, with its p_val, p_val_adj, and logfoldchange; "
                       f"2. {cell_type_full}_mean_expression of the marker genes: specific marker genes mean expression within each cluster; "
                       f"3. {cell_type_full}_expression_proportion of the marker genes: every cluster each gene expression fraction within each cluster, and give back the mapping dictionary in the python dictionary format string corresponding to string without further explanation or comment.  "
                       "I only want the summary map in the response, do not give me any explanation or comment or repeat my input, i dont want any redundant information other than the summary map.")
        self.calculate_cluster_statistics(filtered_cells, cell_type_full)
        cell_markers = self.get_rag_and_markers(False)
        cell_markers_str = ', '.join(str(cell_markers))
        summary2 = (f"Explanation: {explanation}, "
                    f"{cell_type_full} key marker genes include: {cell_markers_str}. "
                    f"{cell_type_full} top genes {self.global_top_genes_df}"
                    f"{cell_type_full} mean expression {str(self.global_mean_expression)}"
                    f"{cell_type_full} expression proportion {str(self.global_expression_proportion)}")
        fname = f'{cell_type_full}_adata_processed.pkl'
        with open(fname, 'wb') as file:
            pickle.dump(umap_df, file)
        return summary2

    def calculate_cell_population_change(self, cell_type):
        cell_type_full = cell_type.split()[0].capitalize() + " cells"
        if cell_type_full == 'Overall cells':
            with open('annotated_adata/Overall cells_annotated_adata.pkl', 'rb') as file:
                cpc_adata = pd.read_pickle(file)
        else:
            fname = f'annotated_adata/{cell_type_full}_annotated_adata.pkl'
            with open(fname, 'rb') as file:
                cpc_adata = pd.read_pickle(file)
        cell_counts = cpc_adata.obs.groupby(['patient_name', 'cell_type']).size().reset_index(name='counts')
        total_counts = cell_counts.groupby('patient_name')['counts'].transform('sum')
        cell_counts['percentage'] = (cell_counts['counts'] / total_counts) * 100
        output_filename = f'schatbot/cell_population_change/{cell_type_full}_cell_population_change.csv'
        cell_counts.to_csv(output_filename, index=False)
        summary = "Can you tell me the cell population change for each cell type / patient from this data? do not tell me how to do it, just tell me"
        summary2 = f"Explanation: {summary}, {cell_counts.to_string(index=False)}"
        return summary2

    def sample_differential_expression_genes_comparison(self, cell_type, sample_1, sample_2):
        cell_type_cap = cell_type.split()[0].capitalize()
        cell_type_full = cell_type_cap + " cells"
        cell_type_alt = cell_type_cap + " cell"
        with open(f'annotated_adata/Overall cells_annotated_adata.pkl', 'rb') as file:
            adata2 = pd.read_pickle(file)
        filtered_cells = adata2[adata2.obs['cell_type'].isin([cell_type_full])].copy()
        if filtered_cells.shape[0] == 0:
            filtered_cells = adata2[adata2.obs['cell_type'].isin([cell_type_alt])].copy()
        if filtered_cells.shape[0] == 0:
            filtered_cells = adata2[adata2.obs['cell_type'].isin([cell_type_cap])].copy()
        adata_filtered = filtered_cells[filtered_cells.obs['patient_name'].isin([sample_1, sample_2])].copy()
        unique_patients = adata_filtered.obs['patient_name'].astype(str).unique()
        if sample_1 not in unique_patients or sample_2 not in unique_patients:
            return f"Error: One or both patients ({sample_1}, {sample_2}) not found in the dataset for cell type '{cell_type_full}'."
        sc.tl.rank_genes_groups(adata_filtered, groupby='patient_name', groups=[sample_2], reference=sample_1, method='wilcoxon')
        results_post = {
            'genes': adata_filtered.uns['rank_genes_groups']['names'][sample_2],
            'logfoldchanges': adata_filtered.uns['rank_genes_groups']['logfoldchanges'][sample_2],
            'pvals': adata_filtered.uns['rank_genes_groups']['pvals'][sample_2],
            'pvals_adj': adata_filtered.uns['rank_genes_groups']['pvals_adj'][sample_2]
        }
        df_post = pd.DataFrame(results_post)
        significant_genes_post = df_post[df_post['pvals_adj'] < 0.05]
        significant_genes_post = significant_genes_post[abs(significant_genes_post['logfoldchanges']) > 1]
        significant_genes_post.to_csv('SGP.csv', index=False)
        summary = (
            f"Reference Sample: {sample_1}, Comparison Sample: {sample_2}\n"
            "Explanation: DO NOT GIVE PYTHON CODE. JUST COMPARE AND EXPLAIN. This function is designed to perform a differential gene expression analysis for a specified cell type between two sample conditions (pre and post-treatment or two different conditions). "
            "The reference patient condition is patient_1. The differential expression analysis is performed by comparing the gene expression levels of sample_2 against those of sample_1. Provide a comparison with normal formatting.\n"
            "Significant Genes Data: \n"
            f"{str(significant_genes_post)}\n"
            "Explanation of attributes:\n"
            "Genes: The names of the genes analyzed, providing insight into which genes are tested for differential expression between the two conditions.\n"
            "Log Fold Changes: Values showing how gene expression levels differ between the two conditions. Positive values indicate upregulation in sample_2, and negative values indicate downregulation in sample_2.\n"
            "P-values: These values help determine the statistical significance of the observed changes in gene expression. Lower p-values suggest that the changes are less likely to have occurred by chance.\n"
            "Adjusted P-values: These values provide a more stringent measure of significance by controlling for the false discovery rate. Significant adjusted p-values (e.g., < 0.05) indicate that the changes in gene expression are statistically robust even after adjusting for multiple comparisons."
        )
        return summary

    # ====================================================================================
    #                                     GSEA related
    # ====================================================================================
    def gsea_analysis(self):
        try:
            self.SGP = pd.read_csv('SGP.csv')
            print("SGP.csv loaded successfully")
        except Exception as e:
            print(f"Error loading SGP.csv: {e}")
            return json.dumps({"status": "error", "message": f"Error loading SGP.csv: {e}"})
        try:
            significant_genes_post = self.SGP
            significant_genes_post['rank'] = -np.log10(significant_genes_post.pvals_adj) * significant_genes_post.logfoldchanges
            significant_genes_post = significant_genes_post.sort_values('rank', ascending=False).reset_index(drop=True)
            ranking = significant_genes_post[['genes', 'rank']]
            gene_list = ranking['genes'].str.strip().to_list()
            print("Gene list prepared")
        except Exception as e:
            print(f"Error preparing gene list: {e}")
            return json.dumps({"status": "error", "message": f"Error preparing gene list: {e}"})
        try:
            libraries = gp.get_library_name()
            pre_res = gp.prerank(rnk=ranking, 
                                 gene_sets=["KEGG_2021_Human", "GO_Biological_Process_2023", "GO_Molecular_Function_2023", 
                                            "GO_Cellular_Component_2023", "Reactome_2022", "MSigDB_Hallmark_2020", 
                                            "MSigDB_Oncogenic_Signatures", "Cancer_Cell_Line_Encyclopedia", 
                                            "Human_Phenotype_Ontology", "Disease_Signatures_from_GEO_down_2014", 
                                            "Disease_Signatures_from_GEO_up_2014", "Disease_Perturbations_from_GEO_down", 
                                            "Disease_Perturbations_from_GEO_up"],
                                 seed=6)
            print("GSEA analysis completed")
        except Exception as e:
            print(f"Error in GSEA analysis: {e}")
            return json.dumps({"status": "error", "message": f"Error in GSEA analysis: {e}"})
        out = []
        for term in pre_res.results:
            fdr = pre_res.results[term]['fdr']
            es = pre_res.results[term]['es']
            nes = pre_res.results[term]['nes']
            if fdr <= 0.05:
                out.append([term, fdr, es, nes])
        try:
            self.out_df = pd.DataFrame(out, columns=['Term', 'fdr', 'es', 'nes']).sort_values('fdr').reset_index(drop=True)
            print("Filtered significant terms")
        except Exception as e:
            print(f"Error filtering significant terms: {e}")
            return json.dumps({"status": "error", "message": f"Error filtering significant terms: {e}"})
        try:
            os.makedirs('gsea_plots', exist_ok=True)
            print("Directory for plots ensured")
        except Exception as e:
            print(f"Error ensuring plot directory: {e}")
            return json.dumps({"status": "error", "message": f"Error ensuring plot directory: {e}"})
        terms_to_plot = self.out_df['Term']
        try:
            axs = pre_res.plot(terms=terms_to_plot, show_ranking=False, legend_kws={'loc': (1.05, 0)})
            plt.title("GSEA Enrichment Scores for Significant Terms (FDR ≤ 0.05)")
            plt.xlabel("Rank in Ordered Dataset")
            plt.ylabel("Enrichment Score (ES)")
            plt.savefig('gsea_plots/enrichment_scores.png')
            plt.close()
            print("Enrichment scores plot saved")
        except Exception as e:
            print(f"Failed to plot all terms together: {e}")
        try:
            plt.figure(figsize=(10, 6))
            scatter = plt.scatter(
                x=self.out_df['nes'],
                y=self.out_df['Term'],
                s=(self.out_df['es'].abs() * 500),
                c=self.out_df['fdr'],
                cmap='RdBu_r',
                alpha=0.7
            )
            cbar = plt.colorbar(scatter)
            cbar.set_label('FDR')
            plt.title('GSEA Dot Plot')
            plt.xlabel('Normalized Enrichment Score (NES)')
            plt.ylabel('Term')
            plt.tight_layout()
            plt.savefig("gsea_plots/gsea_dot_plot.png")
            plt.close()
            print("GSEA dot plot saved")
        except Exception as e:
            print(f"Error saving GSEA dot plot: {e}")
        try:
            output_file = 'gsea_plots/gsea_plot_data.csv'
            if not self.out_df.empty:
                all_output_data = []
                for index, row in self.out_df.iterrows():
                    term_to_graph = row['Term']
                    term_details = pre_res.results[term_to_graph].copy()
                    rank_metric = pre_res.ranking
                    es_profile = term_details['RES']
                    gsea_data = pd.DataFrame({
                        'Rank Metric': rank_metric,
                        'Enrichment Score': es_profile
                    })
                    term_info = {
                        'Term': term_to_graph,
                        'fdr': term_details['fdr'],
                        'es': term_details['es'],
                        'nes': term_details['nes'],
                        'Rank Metric': '',
                        'Enrichment Score': ''
                    }
                    term_info_df = pd.DataFrame([term_info])
                    output_df = pd.concat([term_info_df, gsea_data], ignore_index=True)
                    all_output_data.append(output_df)
                final_output_df = pd.concat(all_output_data, ignore_index=True)
                final_output_df.to_csv(output_file, index=False)
                print(f"GSEA plot data saved to '{output_file}'")
            else:
                print("No significant gene sets found with FDR <= 0.05.")
        except Exception as e:
            print(f"Error saving GSEA plot data: {e}")
        try:
            ranking_gene_list = significant_genes_post[['genes', 'rank']]
            ranking_gene_list.to_csv('ranking_gene_list.csv', index=False)
            print("Ranking gene list saved to 'ranking_gene_list.csv'")
        except Exception as e:
            print(f"Error saving ranking gene list: {e}")
        response = {
            "status": "success",
            "message": "GSEA analysis completed successfully",
            "enrichment_scores_plot": 'gsea_plots/enrichment_scores.png',
            "dot_plot": 'gsea_plots/gsea_dot_plot.png',
            "gsea_plot_data": 'gsea_plot_data.csv',
            "ranking_gene_list": 'ranking_gene_list.csv'
        }
        response_json = json.dumps(response)
        return response_json

    def safe_filename(self, term):
        term_safe = re.sub(r'[^a-zA-Z0-9_]', '_', term)
        return term_safe[:50]

    def read_image(self):
        def encode_image(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        image_path = self.find_file_with_extension("media", ".jpg")
        if not image_path:
            return "No image found in the folder."
        base64_image = encode_image(image_path)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai.api_key}"
        }
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What’s in this image?"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 300
        }
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        try:
            return response.json()['choices'][0]['message']['content']
        except KeyError:
            return "Error in processing the image."
        
    # ====================================================================================
    #                                    Chat-related
    # ====================================================================================
    def start_chat2_web(self, user_input, conversation_history):
        self.function_flag = False
        self.display_flag = False
        scchat_context1 = f"""
        You are a chatbot for helping in Single Cell RNA Analysis, you can call the functions generate_umap, process_cells, label_clusters, display_umap, display_processed_umap and more multiple times. DO NOT FORGET THIS. respond with a greeting.
        """
        if self.first_try and self.clear_data:
            self.clear_directory("annotated_adata")
            self.clear_directory("basic_data")
            self.clear_directory("umaps")
            self.clear_directory("process_cell_data")
            research_context_path = self.find_file_with_extension("media", ".txt")
            if research_context_path:
                with open(research_context_path, "r") as rcptr:
                    research_context = rcptr.read()
                    conversation_history.append({"role": "user", "content": research_context})
            conversation_history.append({"role": "user", "content": scchat_context1})
            self.first_try = False
        conversation_history.append({"role": "user", "content": user_input})
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=conversation_history,
            functions=function_descriptions,
            function_call="auto",
            temperature=0.2,
            top_p=0.4
        )
        output = response.choices[0].message
        main_flag = False
        if response and output.function_call:
            main_flag = True
        if not main_flag:
            fin_response = openai.chat.completions.create(
                model="gpt-4o",
                messages=conversation_history,
                functions=function_descriptions,
                function_call="auto",
                temperature=0.2,
                top_p=0.4
            )
            output = fin_response.choices[0].message
            ai_response = output.content
        if output.function_call and main_flag:
            function_name = output.function_call.name
            function_args = output.function_call.arguments
            self.function_flag = True
            try:
                function_response = "Function did not execute."
                print(f"Making a function call to: {function_name}")
                if function_args is None:
                    function_response = getattr(self, function_name)()
                else:
                    function_args = json.loads(function_args)
                    print("======================================================")
                    print(type(function_args))
                    print("======================================================")
                    print(f"Parsed function arguments: {function_args}")
                    function_response = getattr(self, function_name)(**function_args)
                    
                if self.display_flag:
                    return function_response, conversation_history, self.display_flag
                
                if function_name == "label_clusters":
                    function_result_message = {"role": "assistant", "content": "Annotation is complete."}
                    conversation_history.append(function_result_message)
                    final_response = "Annotation is complete."
                else:
                    function_result_message = {"role": "user", "content": function_response}
                    conversation_history.append(function_result_message)
                    new_response = openai.chat.completions.create(
                        model="gpt-4o",
                        messages=conversation_history,
                        temperature=0,
                        top_p=0.4
                    )
                    final_response = new_response.choices[0].message.content if new_response.choices[0] else "Interesting"
                    conversation_history.append({"role": "assistant", "content": final_response})
                
                if function_name == "generate_umap":
                    self.cluster_summary["overall cells"] = final_response
                elif function_name == "process_cells":
                    self.cluster_summary[self.processed_cell_name] = final_response
                
                self.function_flag = False
                return final_response, conversation_history, self.display_flag
            except KeyError as e:
                print(f"Function {function_name} not found: {e}")
                return f"Function {function_name} not found.", conversation_history, self.display_flag
        else:
            conversation_history.append({"role": "assistant", "content": ai_response})
            return ai_response, conversation_history, self.display_flag


if __name__ == "__main__":
    scchat = ScChat()
    conversation_history = [
        {"role": "user", "content": ""}
    ]

    user_input = "Generate UMAP"
    print("Received message: ", user_input)
    final_response, conversation_history, display_flag = scchat.start_chat2_web(
        user_input=user_input,
        conversation_history=conversation_history
    )
    
    print("Response from API:", final_response)
    # ========================================================================
    
    user_input = "Label the cluster for overall cells."
    print("Received message: ", user_input)
    final_response, conversation_history, display_flag = scchat.start_chat2_web(
        user_input=user_input,
        conversation_history=conversation_history
    )
    
    print("Response from API:", final_response)
    # # ========================================================================
    
    # user_input = "Display the UMAP for overall cell."
    # print("Received message: ", user_input)
    # final_response, conversation_history, display_flag = scchat.start_chat2_web(
    #     user_input=user_input,
    #     conversation_history=conversation_history
    # )
    
    # print("Response from API:", final_response)

    # ========================================================================
    # user_input = "Filter out T cells and generate UMAP again."
    # print("Received message: ", user_input)
    # final_response, conversation_history, display_flag = scchat.start_chat2_web(
    #     user_input=user_input,
    #     conversation_history=conversation_history
    # )
    
    # print("Response from API:", final_response)
    # # ========================================================================
    
    # user_input = "Display the annotated UMAP for overall cells."
    # print("Received message: ", user_input)
    # final_response, conversation_history, display_flag = scchat.start_chat2_web(
    #     user_input=user_input,
    #     conversation_history=conversation_history
    # )
    
    # print("Response from API:", final_response)
    # # ========================================================================
    
    # user_input = "Label the cluster for T cells."
    # print("Received message: ", user_input)
    # final_response, conversation_history, display_flag = scchat.start_chat2_web(
    #     user_input=user_input,
    #     conversation_history=conversation_history
    # )
    
    # print("Response from API:", final_response)
    # ========================================================================
    
    # user_input = "What are the cell population chagnes in T cells before and after treatment?"
    # print("Received message: ", user_input)
    # final_response, conversation_history, display_flag = scchat.start_chat2_web(
    #     user_input=user_input,
    #     conversation_history=conversation_history
    # )
    
    # print("Response from API:", final_response)
    # # ========================================================================
    
    # user_input = "Explain the reason of annotation while providing me with the verified marker gene URL."
    # print("Received message: ", user_input)
    # final_response, conversation_history, display_flag = scchat.start_chat2_web(
    #     user_input=user_input,
    #     conversation_history=conversation_history
    # )
    
    # print("Response from API:", final_response)