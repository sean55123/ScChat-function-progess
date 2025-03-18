import pandas as pd
import scanpy as sc
import os 
import json
from scvi.model import SCVI
import anndata

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

def find_and_load_sample_mapping(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == 'sample_mapping.json':
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    sample_mapping = json.load(f)

    return sample_mapping

def find_file_with_extension(directory_path, extension):
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(extension):
                return os.path.join(root, file)
    return None

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

def filter_existing_genes(adata, gene_list):
    existing_genes = [gene for gene in gene_list if gene in adata.raw.var_names]
    return existing_genes

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


path = find_file_with_extension("media", ".h5ad")
adata = sc.read_h5ad(path)


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
sc.tl.leiden(adata, resolution=0.5, random_state=42)
if sample_mapping:
    adata.obs['patient_name'] = adata.obs['Sample'].map(sample_mapping)
umap_df = adata.obsm['X_umap']
adata.obs['UMAP_1'] = umap_df[:, 0]
adata.obs['UMAP_2'] = umap_df[:, 1]
markers = get_rag_and_markers(False)
markers = extract_genes(markers)
markers = filter_existing_genes(adata, markers)
markers = list(set(markers))  

adata_markers = anndata.AnnData(
    X=adata.raw[:, markers].X.copy(),
    obs=adata.obs.copy(),
    var=adata.raw[:, markers].var.copy()
)
sc.tl.rank_genes_groups(adata_markers, groupby='leiden', method='wilcoxon', use_raw=False, n_genes=50)
sc.pl.rank_genes_groups(adata_markers, groupby='leiden')