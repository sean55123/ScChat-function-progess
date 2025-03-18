function_descriptions = [
    {
        "name": "generate_umap",
        "description": "Used to Generate UMAP for unsupervised clustering for RNA analysis. Generates a UMAP visualization based on the given RNA sequencing data",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "display_dotplot",
        "description": "Displays the dotplot for the sample",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "display_cell_type_composition",
        "description": "Displays the cell type composition for the sample",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "read_image",
        "description": "Reads and processes an image from the 'media' folder and returns a description of what's in the image.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "process_cells",
        "description": "process cells",
        "parameters": {
            "type": "object",
            "properties": {
                "cell_type": {
                    "type": "string",
                    "description": "the cell type"
                }
            },
            "required": ["cell_type"],
        },
    },
    {
        "name": "display_umap",
        "description": "displays umap that is NOT annotated. This function should be called whenever the user asks for a umap that is not annotated. In the case that the user does not specify cell type, use overall cells. This function can be called multiple times. This function should not be called when asked to GENERATE umap.",
        "parameters": {
            "type": "object",
            "properties": {
                "cell_type": {
                    "type": "string",
                    "description": "the cell type"
                }
            },
            "required": ["cell_type"],
        },
    },
    {
        "name": "display_processed_umap",
        "description": "displays umap that IS annotated. This function should be called whenever the user asks for a umap that IS annotated. In the case that the user does not specify cell type, use overall cells. This function can be called multiple times. This function should not be called when asked to GENERATE umap.",
        "parameters": {
            "type": "object",
            "properties": {
                "cell_type": {
                    "type": "string",
                    "description": "the cell type"
                }
            },
            "required": ["cell_type"],
        },
    },
    {
        "name": "display_cell_population_change",
        "description": "displays cell population change graph",
        "parameters": {
            "type": "object",
            "properties": {
                "cell_type": {
                    "type": "string",
                    "description": "the cell type"
                }
            },
            "required": ["cell_type"],
        },
    },
    {
        "name": "sample_differential_expression_genes_comparison",
        "description": "Function is designed to perform a differential gene expression analysis for a specified cell type between two patient conditions (pre and post-treatment or two different conditions)",
        "parameters": {
            "type": "object",
            "properties": {
                "cell_type": {
                    "type": "string",
                    "description": "The type of cell to be compared"
                },
                "sample_1": {
                    "type": "string",
                    "description": "Identifier for the first patient"
                },
                "sample_2": {
                    "type": "string",
                    "description": "Identifier for the second patient"
                }
            },
            "required": ["cell_type", "sample_1", "sample_2"]
        }
    },
    {
        "name": "calculate_cell_population_change",
        "description": "calculate_cell population change for percentage changes in cell populations or samples before and after treatment. This calculation can be done for any cell type. This is to see the changes in the population before and after treatment.",
        "parameters": {
            "type": "object",
            "properties": {
                "cell_type": {
                    "type": "string",
                    "description": "the cell type"
                }
            },
            "required": ["cell_type"],
        },
    },
    {
        "name": "gsea_analysis",
        "description": "Performs Gene Set Enrichment Analysis (GSEA) on a dataset of significant genes. This function ranks the genes based on their adjusted p-values and log-fold changes, performs the GSEA analysis using multiple gene set libraries, filters the results for significant terms (FDR ≤ 0.05), and generates several output files including enrichment score plots, a dot plot, and CSV files with the GSEA results and ranked gene list.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "label_clusters",
        "description": "This function can be called multiple times. this function is to label and or annotate clusters. It can be done for any type of cells that is mentioned by the user. If the user does not mention the cell type use overall cells. This function can be called multiple times.",
        "parameters": {
            "type": "object",
            "properties": {
                "cell_type": {
                    "type": "string",
                    "description": "the cell type"
                }
            },
            "required": ["cell_type"],
        },
    },
]