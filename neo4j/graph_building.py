import pandas as pd
from neo4j import GraphDatabase
import math
from collections import defaultdict, deque

# Read the Excel file (adjust the file path if needed)
df = pd.read_excel('singlecell_marker_collections_Kunming.xlsx', sheet_name='Mus musculus-marker')

# Remove duplicate cell type entries where coarseCellType equals cellTypes
# If the values are identical (after stripping whitespace), set cellTypes to None to prevent self-referencing nodes
df.loc[df['coarseCellType'].fillna('').str.strip() == df['cellTypes'].fillna('').str.strip(), 'coarseCellType'] = None

# Build a directed graph from coarseCellType to cellTypes
graph = defaultdict(set)
all_nodes = set()

for index, row in df.iterrows():
    parent = row["coarseCellType"]
    child = row["cellTypes"]
    if pd.isna(parent) or pd.isna(child):
        continue
    # Clean up string values if needed
    parent = str(parent).strip()
    child = str(child).strip()
    graph[parent].add(child)
    all_nodes.add(parent)
    all_nodes.add(child)

# Identify nodes that never appear as a child – these are the roots (Level 1)
children_set = set(child for children in graph.values() for child in children)
roots = [node for node in all_nodes if node not in children_set]

# Use BFS to assign levels: Level 1 for roots, Level = parent's level + 1 for children.
levels = {}
queue = deque()
for root in roots:
    levels[root] = 1
    queue.append(root)

while queue:
    current = queue.popleft()
    current_level = levels[current]
    for child in graph.get(current, []):
        new_level = current_level + 1
        if child not in levels or new_level < levels[child]:
            levels[child] = new_level
            queue.append(child)

# Extract system, organ, and marker associations from the DataFrame
systems = set()
system_to_organs = defaultdict(set)
organ_to_cells = defaultdict(set)
cell_to_markers = defaultdict(set)

for index, row in df.iterrows():
    # Check if system, organ, and marker columns exist and are not NaN
    if pd.isna(row.get("filepath")) or pd.isna(row.get("organ")) or pd.isna(row.get("canonicalMarkers")):
        continue
    system_val = str(row["filepath"]).strip()
    organ_val = str(row["organ"]).strip()
    marker_val = str(row["canonicalMarkers"]).strip().lower()

    # Use the cell hierarchy columns
    coarse = str(row["coarseCellType"]).strip() if not pd.isna(row["coarseCellType"]) else None
    detailed = str(row["cellTypes"]).strip() if not pd.isna(row["cellTypes"]) else None

    systems.add(system_val)
    system_to_organs[system_val].add(organ_val)
    if coarse:
        organ_to_cells[organ_val].add(coarse)
    if detailed:
        # Split the marker string by commas and add each non-empty, normalized marker to the set
        markers = {m.strip().lower() for m in marker_val.split(",") if m.strip()}
        cell_to_markers[detailed] = cell_to_markers[detailed].union(markers)


def create_cell_node(tx, name, level):
    query = f"""
    MERGE (c:CellType {{name: $name}})
    SET c.level = $level
    SET c:{'Level' + str(level)}
    """
    tx.run(query, name=name, level=level)

def create_relationship(tx, parent, child):
    # Create a relationship representing the developmental hierarchy
    tx.run(
        "MATCH (p:CellType {name: $parent}), (c:CellType {name: $child}) "
        "MERGE (p)-[:DEVELOPS_TO]->(c)",
        parent=parent, child=child
    )

def create_system_node(tx, name):
    tx.run("MERGE (s:System {name: $name})", name=name)

def create_organ_node(tx, name):
    tx.run("MERGE (o:Organ {name: $name})", name=name)

def create_marker_node(tx, name):
    normalized_name = name.strip().lower()
    tx.run("MERGE (m:Marker {name: $normalized_name})", normalized_name=normalized_name)

def create_system_organ_relationship(tx, system, organ):
    tx.run("MATCH (s:System {name: $system}), (o:Organ {name: $organ}) MERGE (s)-[:HAS_ORGAN]->(o)", system=system, organ=organ)

def create_organ_cell_relationship(tx, organ, cell):
    tx.run("MATCH (o:Organ {name: $organ}), (c:CellType {name: $cell}) MERGE (o)-[:HAS_CELL]->(c)", organ=organ, cell=cell)

def create_cell_marker_relationship(tx, cell, marker):
    normalized_marker = marker.strip().lower()
    tx.run(
        "MATCH (c:CellType {name: $cell}), (m:Marker {name: $normalized_marker}) MERGE (c)-[:HAS_MARKER]->(m)",
        cell=cell, normalized_marker=normalized_marker
    )

def create_marker_node(tx, cell, markers, name):
    # Create a single aggregated marker node for a cell with a list of unique markers
    tx.run("MERGE (am:Marker {name: $name}) SET am.markers = $markers",
           name=name, markers=markers)

def create_cell_to_marker_relationship(tx, cell, name):
    # Link the cell node to the aggregated marker node
    tx.run("MATCH (c:CellType {name: $cell}), (am:Marker {name: $name}) MERGE (c)-[:HAS_MARKER]->(am)",
           cell=cell, name=name)

with driver.session(database="mouse") as session:
    # Create nodes with the computed level for cell types
    for cell, level in levels.items():
        session.write_transaction(create_cell_node, cell, level)
    
    # Create relationships based on the original cell hierarchy graph
    for parent, children in graph.items():
        for child in children:
            session.write_transaction(create_relationship, parent, child)
            
    # Create system nodes and link them to organ nodes
    for system in systems:
        session.write_transaction(create_system_node, system)
    
    for system, organs in system_to_organs.items():
        for organ in organs:
            session.write_transaction(create_organ_node, organ)
            session.write_transaction(create_system_organ_relationship, system, organ)
    
    # Link organ nodes to the corresponding coarse cell types
    for organ, cells in organ_to_cells.items():
        for cell in cells:
            session.write_transaction(create_organ_cell_relationship, organ, cell)
    
    # Create aggregated marker nodes and link them to detailed cell types
    for cell, markers in cell_to_markers.items():
        aggregated_name = cell + "_aggregated_marker"
        session.write_transaction(create_marker_node, cell, list(markers), aggregated_name)
        session.write_transaction(create_cell_to_marker_relationship, cell, aggregated_name)

driver.close()