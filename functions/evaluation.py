import pandas as pd
import pronto
import pickle
import os
import re

# --- Configuration ---
ontology_file_path = '/Users/seanchiu/Desktop/scChat development/compare/cl.owl'  # Update this path if needed
pickle_file_path = 'cell_ontology_lookup.pkl'
csv_file_path = 'combined_pbmc3k.csv'
output_file_path = 'cell_type_comparison_results.csv'

# --- Helper Functions ---
def load_map_pickle(file_path):
    """Load the cell name to CL ID lookup map from a pickle file."""
    print(f"Loading map from {file_path}...")
    try:
        with open(file_path, 'rb') as f:
            loaded_map = pickle.load(f)
        print("Map loaded successfully.")
        return loaded_map
    except FileNotFoundError:
        print(f"Error: Pickle file not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading map: {e}")
        return None

def build_reverse_lookup(name_to_id_map):
    """Build a reverse lookup from CL IDs to cell names."""
    id_to_name_map = {}
    for name, cl_id in name_to_id_map.items():
        if cl_id not in id_to_name_map:
            id_to_name_map[cl_id] = []
        id_to_name_map[cl_id].append(name)
    return id_to_name_map

def normalize_cell_name(name):
    """Normalize cell type names for better matching."""
    if name is None or not isinstance(name, str):
        return ""
    
    # Convert to lowercase
    name = name.lower().strip()
    
    # Handle common variations
    name = name.replace('-positive', '+').replace('-negative', '-')
    name = name.replace(' positive ', ' + ').replace(' negative ', ' - ')
    
    # Standardize "cell" vs "cells" endings
    if name.endswith(' cells'):
        name = name[:-1]  # Remove the 's' to make it "cell"
    elif not name.endswith(' cell') and not name.endswith('+') and not name.endswith('-'):
        name = name + ' cell'  # Add "cell" if it doesn't end with "cell", "+", or "-"
    
    return name

def check_if_predicted_is_parent(predicted_cl_id, ground_truth_cl_id, ontology):
    """Check if predicted cell type is a parent of ground truth cell type."""
    # If either ID is not found, we can't determine the relationship
    if predicted_cl_id not in ontology or ground_truth_cl_id not in ontology:
        return False
    
    # Get the terms
    predicted_term = ontology[predicted_cl_id]
    ground_truth_term = ontology[ground_truth_cl_id]
    
    # Case 1: Direct match (same term)
    if predicted_cl_id == ground_truth_cl_id:
        return True
    
    # Case 2: Check if predicted is a parent of ground truth
    # Get all ancestors of the ground truth term (including self)
    ancestors = ground_truth_term.superclasses(distance=None)
    
    # Check if predicted term is among the ancestors
    return predicted_term in ancestors

def get_cl_id_for_cell_name(cell_name, name_to_id_map, try_variations=True):
    """
    Try to find a CL ID for a cell name, with enhanced plural handling and variation matching.
    """
    # Guard against None
    if cell_name is None:
        return None
    
    # Try the original name first (exact match)
    cl_id = name_to_id_map.get(cell_name)
    if cl_id:
        return cl_id
    
    # Try the name as provided after normalization
    normalized_name = normalize_cell_name(cell_name)
    cl_id = name_to_id_map.get(normalized_name)
    if cl_id:
        return cl_id
    
    # Try lowercased version
    lower_name = cell_name.lower()
    cl_id = name_to_id_map.get(lower_name)
    if cl_id:
        return cl_id
    
    # Try lowercased version without special characters or extra spaces
    clean_name = re.sub(r'[^\w\s]', ' ', lower_name)
    clean_name = re.sub(r'\s+', ' ', clean_name).strip()
    cl_id = name_to_id_map.get(clean_name)
    if cl_id:
        return cl_id
    
    # Enhanced plural handling - try singular form 
    if cell_name.endswith('s'):
        singular_form = cell_name[:-1]
        cl_id = name_to_id_map.get(singular_form)
        if cl_id:
            return cl_id
        
        # Try lowercase singular
        cl_id = name_to_id_map.get(singular_form.lower())
        if cl_id:
            return cl_id
        
        # Try normalized singular
        if normalized_name.endswith('s'):
            singular_normalized = normalized_name[:-1]
            cl_id = name_to_id_map.get(singular_normalized)
            if cl_id:
                return cl_id
    
    # If still not found and try_variations is True, try more variations
    if cl_id is None and try_variations and isinstance(cell_name, str):
        # Try variations with different cases
        variants = [
            cell_name.upper(),                       # ALL CAPS
            cell_name.title(),                       # Title Case
            cell_name.capitalize(),                  # First letter capitalized
        ]
        
        # Try with/without cell suffix
        base_variants = [
            normalized_name,                                   # Normalized (lowercase, standardized)
            normalized_name[:-5].strip() if normalized_name.endswith(' cell') else normalized_name,  # Without "cell"
            normalized_name + ' cell' if not normalized_name.endswith(' cell') else normalized_name, # With "cell"
            normalized_name.rstrip(' cell') + ' cells'         # With "cells" plural
        ]
        variants.extend(base_variants)
        
        # Handle specific plural cases for common cell types
        if cell_name == "Platelets":
            variants.append("platelet")
            variants.append("blood platelet")
            variants.append("blood platelets")
        elif cell_name == "Monocytes":
            variants.append("monocyte")
        elif cell_name == "Mononuclear phagocytes":
            variants.append("mononuclear phagocyte")
            variants.append("phagocyte")
            variants.append("phagocytes")
        
        # Try CD variations (uppercase CD vs lowercase cd)
        if 'cd' in normalized_name.lower():
            cd_variants = [
                normalized_name.replace('cd', 'CD'),
                normalized_name.replace('CD', 'cd')
            ]
            variants.extend(cd_variants)
        
        # Try T-cell vs T cell variations
        if 't cell' in normalized_name.lower() or 't-cell' in normalized_name.lower():
            t_variants = [
                normalized_name.replace('t cell', 't-cell'),
                normalized_name.replace('t-cell', 't cell'),
                normalized_name.replace('T cell', 'T-cell'),
                normalized_name.replace('T-cell', 'T cell')
            ]
            variants.extend(t_variants)
        
        # Try alpha-beta vs alpha beta variations
        if 'alpha-beta' in normalized_name.lower() or 'alpha beta' in normalized_name.lower():
            ab_variants = [
                normalized_name.replace('alpha-beta', 'alpha beta'),
                normalized_name.replace('alpha beta', 'alpha-beta')
            ]
            variants.extend(ab_variants)
        
        # Try positive/negative variations
        if '+' in normalized_name or '-' in normalized_name:
            pm_variants = [
                normalized_name.replace('+', '-positive,'),
                normalized_name.replace('-', '-negative,'),
                normalized_name.replace('-positive,', '+'),
                normalized_name.replace('-negative,', '-')
            ]
            variants.extend(pm_variants)
        
        # Check all variants
        for variant in variants:
            cl_id = name_to_id_map.get(variant)
            if cl_id:
                return cl_id
    
    # If we've tried everything and still can't find it, return None
    return None

# --- Main Workflow ---
def main():
    # 1. Load the lookup map
    name_to_id_map = load_map_pickle(pickle_file_path)
    if name_to_id_map is None:
        print("Could not load the lookup map. Exiting.")
        return
    
    # 2. Build a reverse lookup for getting names from IDs
    id_to_name_map = build_reverse_lookup(name_to_id_map)
    
    # 3. Load the ontology
    print(f"Loading ontology from {ontology_file_path}...")
    try:
        ontology = pronto.Ontology(ontology_file_path)
        print("Ontology loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Ontology file not found at {ontology_file_path}")
        return
    except Exception as e:
        print(f"Error loading ontology: {e}")
        return
    
    # 4. Read the CSV file
    print(f"Reading CSV file from {csv_file_path}...")
    try:
        df = pd.read_csv(csv_file_path)
        print(f"CSV loaded successfully with {len(df)} rows.")
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_file_path}")
        return
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
    
    # 5. Add columns for CL IDs and results
    df['ground_truth_cl_id'] = None
    df['predicted_cl_id'] = None
    df['is_predicted_parent'] = False
    df['parent_match_type'] = 'Unknown'  # Will be 'Direct Match', 'Parent', 'No Relationship', or 'Unknown'
    
    # Keep track of cell types not found in the ontology
    not_found_ground_truth = set()
    not_found_predicted = set()
    
    # Print some sample entries from the lookup map for debugging
    print("\nSample entries from lookup map:")
    sample_keys = list(name_to_id_map.keys())[:5]
    for key in sample_keys:
        print(f" - '{key}' -> {name_to_id_map[key]}")
    
    # Try looking up the problematic cell types directly
    problematic_cells = [
        "central memory CD4-positive, alpha-beta T cell",
        "CD8-positive, alpha-beta T cell",
        "CD4-positive, alpha-beta T cell",
        "B cell, CD19-positive",
        "naive thymus-derived CD4-positive, alpha-beta T cell",
        "Platelets",
        "Monocytes",
        "Mononuclear phagocytes"
    ]
    print("\nTesting lookup for known problematic cell types:")
    for cell in problematic_cells:
        cl_id = get_cl_id_for_cell_name(cell, name_to_id_map)
        print(f" - '{cell}' -> {cl_id}")
    
    # 6. Process each row
    print("Processing rows...")
    for idx, row in df.iterrows():
        if idx % 500 == 0:  # Print progress every 500 rows
            print(f"Processing row {idx}/{len(df)}...")
            
        # Get cell type names
        ground_truth = row['ground_truth_cell_type']
        predicted = row['predicted_cell_type']
        
        # Try to get CL IDs
        ground_truth_cl_id = get_cl_id_for_cell_name(ground_truth, name_to_id_map)
        predicted_cl_id = get_cl_id_for_cell_name(predicted, name_to_id_map)
        
        # Store the CL IDs
        df.at[idx, 'ground_truth_cl_id'] = ground_truth_cl_id
        df.at[idx, 'predicted_cl_id'] = predicted_cl_id
        
        # Track not found cell types
        if ground_truth_cl_id is None and ground_truth:
            not_found_ground_truth.add(ground_truth)
        if predicted_cl_id is None and predicted:
            not_found_predicted.add(predicted)
        
        # Check relationship
        if ground_truth_cl_id and predicted_cl_id:
            # Case: Direct match (same cell type)
            if ground_truth_cl_id == predicted_cl_id:
                df.at[idx, 'is_predicted_parent'] = True
                df.at[idx, 'parent_match_type'] = 'Direct Match'
            else:
                # Check if predicted is a parent of ground truth
                is_parent = check_if_predicted_is_parent(
                    predicted_cl_id, ground_truth_cl_id, ontology
                )
                df.at[idx, 'is_predicted_parent'] = is_parent
                df.at[idx, 'parent_match_type'] = 'Parent' if is_parent else 'No Relationship'
        else:
            # Case: One or both cell types not found in ontology
            df.at[idx, 'parent_match_type'] = 'Unknown'
    
    # 7. Save detailed results
    print(f"Saving detailed results to {output_file_path}...")
    df.to_csv(output_file_path, index=False)
    print("Detailed results saved successfully.")
    
    # 7b. Create and save binary results file
    binary_output_path = 'cell_type_binary_results.csv'
    print(f"Saving binary results to {binary_output_path}...")
    
    # Create a new dataframe with just the cell_id and binary result
    binary_df = pd.DataFrame()
    binary_df['cell_id'] = df['cell_id']
    
    # Set to 1 for Direct Match or Parent, 0 for No Relationship
    # Rows with Unknown parent_match_type (where one or both cell types weren't found) will get 0
    binary_df['is_correct'] = ((df['parent_match_type'] == 'Direct Match') | 
                              (df['parent_match_type'] == 'Parent')).astype(int)
    
    # Save the binary results
    binary_df.to_csv(binary_output_path, index=False)
    print("Binary results saved successfully.")
    
    # 8. Generate summary
    total_rows = len(df)
    direct_matches = sum(df['parent_match_type'] == 'Direct Match')
    parent_matches = sum(df['parent_match_type'] == 'Parent')
    no_relationships = sum(df['parent_match_type'] == 'No Relationship')
    unknowns = sum(df['parent_match_type'] == 'Unknown')
    
    print("\n--- Summary ---")
    print(f"Total rows analyzed: {total_rows}")
    print(f"Direct matches: {direct_matches} ({direct_matches/total_rows*100:.2f}%)")
    print(f"Parent relationships: {parent_matches} ({parent_matches/total_rows*100:.2f}%)")
    print(f"No relationships: {no_relationships} ({no_relationships/total_rows*100:.2f}%)")
    print(f"Unknown (not found in ontology): {unknowns} ({unknowns/total_rows*100:.2f}%)")
    
    total_correct = direct_matches + parent_matches
    print(f"Total correct predictions (direct match or parent): {total_correct} ({total_correct/total_rows*100:.2f}%)")
    
    # Display cell types not found in the ontology (up to a limit)
    max_to_show = 10
    if not_found_ground_truth:
        print(f"\nGround truth cell types not found in ontology ({len(not_found_ground_truth)} unique types):")
        for ct in list(not_found_ground_truth)[:max_to_show]:
            print(f" - {ct}")
        if len(not_found_ground_truth) > max_to_show:
            print(f"   ... and {len(not_found_ground_truth) - max_to_show} more")
    
    if not_found_predicted:
        print(f"\nPredicted cell types not found in ontology ({len(not_found_predicted)} unique types):")
        for ct in list(not_found_predicted)[:max_to_show]:
            print(f" - {ct}")
        if len(not_found_predicted) > max_to_show:
            print(f"   ... and {len(not_found_predicted) - max_to_show} more")

if __name__ == "__main__":
    main()