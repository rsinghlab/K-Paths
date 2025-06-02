import json
import logging
import os
import re
import networkx as nx
import pandas as pd
from datasets import Dataset

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Helper function to convert dictionary keys to integers where applicable
def convert_keys_to_int(d):
    return {int(k) if k.isdigit() else k: v for k, v in d.items()}

# Load JSON safely
def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

# Load and process dataset mappings
def load_and_process_mappings(dataset_name, paths, hetionet_path):
    logger.info("Processing nodes")
    def preprocess_B_to_dict(B):
        return {f"{item['kind']}::{item['identifier']}": item["name"] for item in B}

    def create_new_dict(A, B_dict):
        return {key: B_dict.get(value) for key, value in A.items() if value in B_dict}

    # Load mappings
    kg_entities = {value: key for key, value in load_json(paths[dataset_name]["bkg_entity2id"]).items()}
    hetionet_nodes = load_json(hetionet_path)["nodes"]
    B_dict = preprocess_B_to_dict(hetionet_nodes)
    
    # Create entity-to-name mapping
    entities_to_name_dict = create_new_dict(kg_entities, B_dict)

    # Load drug node-to-ID mapping
    drug_data = load_json(paths[dataset_name]["node2id"])
    drug_bank_ids = {value: key for key, value in drug_data.items()}

    # Load drug information
    if dataset_name == "drugbank":
        drug_info = pd.read_json(paths[dataset_name]["drug_info"]).T
        # drug_info = pd.read_json(paths[dataset_name]["drug_info"])
        drug_info.reset_index(inplace=True)
        drug_info.columns = [
            "id",
            "name",
            "name_postfix_input_ids",
            "description",
            "sent_list",
            "sent_tokenized_list",
            "prefix_sent_tokenized_list",
        ]
        drug_dict = pd.Series(drug_info.name.values, drug_info.id.values).to_dict()
        
    elif dataset_name == "ddinter":
        drug_info = pd.read_csv(paths[dataset_name]["drug_info"])
        drug_dict = pd.Series(
            drug_info.name.values, drug_info.drugbank_id.values
        ).to_dict()
        
    else:
        drug_dict = load_json(paths[dataset_name]["drug_info"])

    # Create final mapping
    drugs_to_name_dict = {key: drug_dict.get(value) for key, value in drug_bank_ids.items() if value in drug_dict}
    return {**drugs_to_name_dict, **entities_to_name_dict}

# Build graph from file
def build_graph_from_file(file_path, dataset, add_reverse_edges=False, offset=0):
    logger.info(f"Loading graph; Reverse Edges = {add_reverse_edges}")
    dataset_tuples = dataset.map(
        lambda x: {"tuple": (x["drug1_id"], x["label_idx"] + offset, x["drug2_id"])},
        remove_columns=dataset.column_names,
    )
    dataset_tuples_set = set([tuple(x) for x in dataset_tuples["tuple"]])

    G = nx.DiGraph()
    with open(file_path, "r") as f:
        header = f.readline()
        if not header.strip().startswith("entitya"):
            f.seek(0)
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            entitya_id, entityb_id, relation_id = map(int, parts)
            if (entitya_id, relation_id, entityb_id) in dataset_tuples_set:
                continue
            G.add_edges_from([(entitya_id, entityb_id, {"relation": relation_id})])
            if add_reverse_edges:
                G.add_edges_from([(entityb_id, entitya_id, {"relation": relation_id + offset})])

    logger.info(f"Graph built with {len(G.nodes)} nodes and {len(G.edges)} edges")
    return G

# Path finding functions
# NOTE!!!
# Filtering for diverse relations is performed based on relation IDs.
# This means that when selecting paths, the function ensures that 
# the same relation ID does not appear multiple times in a path 
# to maintain diversity.
#
# However, this filtering does NOT:
# 1. Detect or remove redundancy caused by inverse relations or if asymmetric relations occur.
#    - Some relations are inherently one-way, such as "inhibits" vs. "is inhibited by".
#    - Since filtering is only based on relation IDs, it does not check whether
#      an asymmetric relation is meaningful in a given path.
#
# As a result, paths may still contain redundancy due to inverse relations
# or asymmetric relations being treated the same way as symmetric ones.
#
# Potential Fix:
# - Consider grouping inverse relations together and selecting only one.
# - Implement additional logic to check asymmetric relations explicitly.

def find_Kpaths(G, drug_a, drug_b, max_length=3, max_paths=10):
    logger.info("Finding K diverse paths")
    if not G.has_node(drug_a) and not G.has_node(drug_b):
        return []
    elif G.has_node(drug_a) and not G.has_node(drug_b):
        return one_node_in_graph(G, drug_a, max_paths)
    elif not G.has_node(drug_a) and G.has_node(drug_b):
        return one_node_in_graph(G, drug_b, max_paths)
    else:
        return two_nodes_in_graph(G, drug_a, drug_b, max_length, max_paths)
        
def find_neighbors_only(G, drug_a, drug_b,max_paths=10):
    logger.info("Finding neighbors only")
    if not G.has_node(drug_a) and not G.has_node(drug_b):
        return []
    elif G.has_node(drug_a) and not G.has_node(drug_b):
        return one_node_in_graph_no_filter(G, drug_a, max_paths)
    elif not G.has_node(drug_a) and G.has_node(drug_b):
        return one_node_in_graph_no_filter(G, drug_b, max_paths)
    else:
        return two_nodes_in_graph_neigbors_only(G, drug_a, drug_b, max_paths)

def find_Kpaths_no_filter(G, drug_a, drug_b, max_length=3, max_paths=10):
    logger.info("Finding K paths without diversity")
    if not G.has_node(drug_a) and not G.has_node(drug_b):
        return []
    elif G.has_node(drug_a) and not G.has_node(drug_b):
        return one_node_in_graph_no_filter(G, drug_a, max_paths)
    elif not G.has_node(drug_a) and G.has_node(drug_b):
        return one_node_in_graph_no_filter(G, drug_b, max_paths)
    else:
        return two_nodes_in_graph_no_filter(G, drug_a, drug_b, max_length, max_paths)



# Updated one_node_in_graph function with filtering step
def one_node_in_graph(G, drug_id, max_paths=10):
    """
    Finds direct connections from a single drug node in the graph while avoiding duplicate relation paths.
    
    Parameters:
    G (networkx.Graph): The graph.
    drug_id (str): The drug node to search from.
    max_paths (int): The maximum number of unique paths to return.
    
    Returns:
    list: A list of paths [(drug_id, relation, neighbor)].
    """
    neighbors = list(G.neighbors(drug_id))
    
    # Track seen relations to avoid duplicates
    relations_seen = set()
    all_paths = []

    for neighbor in neighbors:
        edge_data = G.get_edge_data(drug_id, neighbor)
        relation_id = edge_data.get("relation", None)

        # Avoid duplicate relations
        if relation_id not in relations_seen:
            relations_seen.add(relation_id)
            all_paths.append([(drug_id, relation_id, neighbor)])
        
        # Stop if max_paths is reached
        if len(all_paths) >= max_paths:
            break

    return all_paths

def two_nodes_in_graph(G, drug_a, drug_b, max_length=3, max_paths=10):
    # no path between drug_a and drug_b return empty list
    """
    Return up to `max_paths` shortest simple paths (≤ `max_length` edges)
    between `drug_a` and `drug_b`, with out duplicate
    relation sequences.
    """
    if not nx.has_path(G, drug_a, drug_b):
        return one_node_in_graph(G, drug_a, max_paths) + one_node_in_graph(
            G, drug_b, max_paths
        )

    all_paths = []
    relations_seen = set()
    paths_generator = nx.shortest_simple_paths(G, drug_a, drug_b)
    # https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.simple_paths.shortest_simple_paths.html
    for path in paths_generator:
        if len(path) - 1 > max_length or len(all_paths) >= max_paths:
            break
        path_relations = []
        _tuples = []
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]
            edge_data = G.get_edge_data(u, v)
            relation_id = edge_data.get("relation", None)
            _tuples.append((u, relation_id, v))
            path_relations.append(relation_id)

        relations_tuple = tuple(path_relations)
        if relations_tuple not in relations_seen:
            relations_seen.add(relations_tuple)
            all_paths.append(_tuples)

    return all_paths



def two_nodes_in_graph_no_filter(G, drug_a, drug_b,
                                 max_length: int = 3,
                                 max_paths:  int = 10):
    """
    Return up to `max_paths` shortest simple paths (≤ `max_length` edges)
    between `drug_a` and `drug_b`, without filtering out duplicate
    relation sequences.
    """
    # If the drugs aren’t connected, fall back to direct neighbours only
    if not nx.has_path(G, drug_a, drug_b):
        return (
            one_node_in_graph(G, drug_a, max_paths) +
            one_node_in_graph(G, drug_b, max_paths)
        )

    all_paths      = []
    paths_generator = nx.shortest_simple_paths(G, drug_a, drug_b)

    for path in paths_generator:
        # Stop once a path exceeds the allowed hop count
        if len(path) - 1 > max_length:
            break
        # Stop after gathering the requested number of paths
        if len(all_paths) >= max_paths:
            break

        # Convert the node list into the (u, relation_id, v) tuples you need
        tuples = []
        for u, v in zip(path[:-1], path[1:]):
            edge_data   = G.get_edge_data(u, v) or {}
            relation_id = edge_data.get("relation")
            tuples.append((u, relation_id, v))

        all_paths.append(tuples)

    return all_paths


def one_node_in_graph_no_filter(G, drug_id, max_paths: int = 10):
    """
    Return up to `max_paths` direct neighbor relations for `drug_id`
    with **no** filtering for duplicate relation types.
    """
    all_paths = []

    for neighbor in G.neighbors(drug_id):
        if len(all_paths) >= max_paths:        # stop once limit reached
            break
        edge_data   = G.get_edge_data(drug_id, neighbor) or {}
        relation_id = edge_data.get("relation")
        all_paths.append([(drug_id, relation_id, neighbor)])

    return all_paths
    
def two_nodes_in_graph_neigbors_only(G, drug_a, drug_b, max_paths: int = 10):
    """
    Collects at most half of `max_paths` direct‑neighbor relations for
    `drug_a` and the same number for `drug_b` (5 each when max_paths=10).

    Unlike the earlier version, this function:
      • does **not** try to connect the two drugs via intermediate nodes  
      • does **not** filter out duplicate relation types

    Parameters
    ----------
    G : networkx.Graph
        The graph.
    drug_a, drug_b : str
        The two drug nodes that are *already known* to be in `G`.
    max_paths : int, default 10
        Overall cap on the number of (single‑edge) paths returned.

    Returns
    -------
    list
        A list in the form
        [
          [(drug_a, relation_id, neighbor_1)],
          … (up to 5 for drug_a) …
          [(drug_b, relation_id, neighbor_1)],
          … (up to 5 for drug_b) …
        ]
    """
    # How many neighbors to keep per drug (e.g. 5 if max_paths == 10)
    per_node_limit = max_paths // 2 or 1

    all_paths = []

    for drug_id in (drug_a, drug_b):
        count = 0
        for neighbor in G.neighbors(drug_id):
            if count >= per_node_limit:
                break
            edge_data = G.get_edge_data(drug_id, neighbor) or {}
            relation_id = edge_data.get("relation")
            all_paths.append([(drug_id, relation_id, neighbor)])
            count += 1

    # In case max_paths is odd, trim to the exact limit
    return all_paths[:max_paths]


def remove_leakage(row, interaction_dict, label_col, drug1_col, drug2_col):
    """
    Removes leakage in path strings by identifying and removing direct interactions.
    Handles optional suffixes (e.g., "Compound", "Disease") and works with reversed order.
    """
    # If label is not in interaction_dict, return the path_str unchanged
    if row[label_col] not in interaction_dict:
        logger.info(f"Label {row[label_col]} not found in interaction_dict. Returning original path_str.")
        return row["path_str"]

    # Extract and escape the interaction template dynamically
    interaction_template = interaction_dict[row[label_col]]
    # logger.info(interaction_template)
    interaction_template = re.escape(interaction_template.replace("{u}", "").replace("{v}", "").strip())

    # Create regex patterns for drug-disease interactions
    drug_pattern = rf"{re.escape(row[drug1_col])}(?:\s*\(.*?\))?"
    disease_pattern = rf"{re.escape(row[drug2_col])}(?:\s*\(.*?\))?"

    # Define regex patterns for both interaction orders
    patterns = [
        re.compile(rf"{drug_pattern}\s*{interaction_template}\s*{disease_pattern}", re.IGNORECASE),
        re.compile(rf"{disease_pattern}\s*{interaction_template}\s*{drug_pattern}", re.IGNORECASE)
    ]

    
    # Process each line in path_str
    cleaned_lines = []
    for line in row["path_str"].split("\n"):
        cleaned_line = line.strip()

        # Apply regex patterns directly
        for pattern in patterns:
            cleaned_line = pattern.sub("", cleaned_line).strip()

        if cleaned_line:  # Only add non-empty lines
            cleaned_lines.append(cleaned_line)

    # Join cleaned lines into final output
    return "\n".join(cleaned_lines)



def build_relations_dict(dataset_name, relations_path, add_reverse_edges=False):
    """
    Builds a dictionary mapping relation IDs to names for a given dataset.

    Args:
        dataset_name (str): Name of the dataset (e.g., "drugbank", "pharmaDB").
        relations_path (str): Path to the JSON file containing relations.
        add_reverse_edges (bool): Whether to include reversed relations.

    Returns:
        tuple: (relation_id_to_name dictionary, offset)
    """
    
    logger.info("Processing relations")

    # Load JSON from file
    relations_dict = load_json(relations_path)

    # Extract dataset-specific relations
    dataset_relations = relations_dict.get(dataset_name, {})

    # Load required relation mappings
    common_relation_id_to_name = dataset_relations.get("common_relation_id_to_name", {})
    dataset_specific_rel_to_name = dataset_relations.get(f"{dataset_name}_relation_id_to_name", {})

    # Find overlapping keys
    common_keys = set(common_relation_id_to_name.keys())
    dataset_keys = set(dataset_specific_rel_to_name.keys())
    duplicate_keys = common_keys & dataset_keys

    # Identify conflicting values
    conflicting_keys = {k for k in duplicate_keys if common_relation_id_to_name[k] != dataset_specific_rel_to_name[k]}

    if conflicting_keys:
        logger.info(f"Warning: Conflicting relation IDs detected and will be overwritten: {conflicting_keys}")

    # Merge relations (common + dataset-specific)
    relation_id_to_name = {**common_relation_id_to_name, **dataset_specific_rel_to_name}

    # Compute offset (for reverse edges)
    offset = len(relation_id_to_name)

    # Handle reversed relations (if needed)
    if add_reverse_edges:
        reversed_common_rel = dataset_relations.get("reversed_common_relation_id_to_name", {})
        reversed_dataset_rel_to_name = dataset_relations.get(f"reversed_{dataset_name}_relation_id_to_name", {})

        # Merge reversed relations
        combined = {**reversed_common_rel, **reversed_dataset_rel_to_name}

        # Apply offset to avoid ID conflicts
        reversed_with_offset = {int(k) + offset: v for k, v in combined.items()}
        relation_id_to_name.update(reversed_with_offset)
        
    # Debugging Output
   
    logger.info(f"Offset for {dataset_name}: {offset}")
    logger.info(f"Number of Relations for {dataset_name}:{len(dataset_specific_rel_to_name)}")
    

    return relation_id_to_name, offset, dataset_specific_rel_to_name



def load_and_process_dataset(dataset_name, split, paths, debug=False):
    """
    Loads and processes the dataset before graph construction.

    Args:
        dataset_name (str): The dataset name (e.g., "drugbank").
        split (str): "train" or "test".
        paths (dict): Dictionary containing dataset paths.
        debug (bool): Whether to load a small subset for debugging.

    Returns:
        Dataset: The processed dataset with renamed and mapped columns.
    """
    logger.info(f"Loading the dataset; Split = {split}")

    # Get the file path based on dataset name and split
    file_path = paths[dataset_name].get(f"{split}_set")
    logger.info(f"File path for dataset '{dataset_name}' and split '{split}': {file_path}")
    
    
    # Force string type in case file_path is accidentally a tuple
    if isinstance(file_path, tuple):
        file_path = file_path[0]


    # Determine file type based on extension
    file_extension = os.path.splitext(file_path)[-1].lower()

    if file_extension == ".json":
        try:
            dataset = Dataset.from_json(file_path)
        except ValueError:
            # If JSON Lines format fails, try loading as a standard JSON array
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):  # Ensure it's an array of objects
                    dataset = Dataset.from_list(data)
                else:
                    raise ValueError("JSON file must be in JSON Lines format or an array of objects.")
    
    elif file_extension == ".csv":
        dataset = Dataset.from_csv(file_path)
    elif file_extension == ".parquet":
        dataset = Dataset.from_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

    # Load only a small subset for debugging
    if debug:
        dataset = dataset.select(range(min(5, len(dataset))))  

    # Process different datasets based on dataset_name
    def process_drugbank_dataset(dataset):
        test_columns_pairs = {
            "Drug1_ID": "drug1_db",
            "Drug2_ID": "drug2_db",
            "Y": "label_idx",
            "effect": "label",
        }
        train_column_pairs = {
            "description_y": "drug1_desc",
            "description_y_drug2": "drug2_desc",
            "Drug1_ID": "drug1_db",
            "Drug2_ID": "drug2_db",
            "Y": "label_idx",
            "effect": "label",
        }
        

        columns_pairs = train_column_pairs if split == "train" else test_columns_pairs
        for old_col, new_col in columns_pairs.items():
            if old_col in dataset.column_names:
                dataset = dataset.rename_column(old_col, new_col)

        with open(paths[dataset_name]["node2id"], "r") as file:
            drug_data = json.load(file)

        dataset = dataset.map(
            lambda example: {
                "drug1_id": drug_data.get(example["drug1_db"], -1),
                "drug2_id": drug_data.get(example["drug2_db"], -1),
            }
        )

        return dataset
    


    def process_ddinter_dataset(dataset):
        logger.info(f"Initial dataset columns: {dataset.column_names}")
        # rename columns
        label_to_idx = {"minor": 0, "moderate": 1, "major": 2}
        columns_pairs = {
            "Drug1_ID": "drug1_db",
            "Drug2_ID": "drug2_db",
            "output": "label",
        }
        for k, v in columns_pairs.items():
            dataset = dataset.rename_column(k, v)
            
            
        if split == "train":
            # convert to tuples "('DDInter644', 'DDInter885')" -> ('DDInter644', 'DDInter885')
            dataset = dataset.map(
                lambda example: {
                    "drug_pair": list(eval(example["drug_pair"])),
                }
            )
            # preprocess the label
            dataset = dataset.map(
                lambda example: {
                    "label": re.sub(
                        r"##Answer:", "", example["label"], flags=re.I
                    ).strip(),
                }
            )

        dataset = dataset.map(
            lambda example: {
                "label_idx": label_to_idx.get(example["label"].strip().lower(), -1),
            }
        )

        with open(paths[dataset_name]["node2id"], "r") as file:
            drug_data = json.load(file)

        dataset = dataset.map(
            lambda example: {
                "drug1_id": drug_data.get(example["drug1_db"], -1),
                "drug2_id": drug_data.get(example["drug2_db"], -1),
            }
        )
        # logger.info(f"Initial dataset columns2: {dataset.column_names}")
        return dataset

    def process_pharmaDB_dataset(dataset):
        label_to_idx = {"disease-modifying": 0, "palliates": 1, "non indications": 2}
        columns_pairs = {
            "drugbank_id": "drug1_db",
            "doid_id": "drug2_db",
            "category": "label",
        }
        for old_col, new_col in columns_pairs.items():
            if old_col in dataset.column_names:
                dataset = dataset.rename_column(old_col, new_col)

        dataset = dataset.map(
            lambda example: {
                "label_idx": label_to_idx.get(example["label"].strip().lower(), -1),
            }
        )

        with open(paths[dataset_name]["node2id"], "r") as file:
            drug_data = json.load(file)

        dataset = dataset.map(
            lambda example: {
                "drug1_id": drug_data.get(example["drug1_db"], -1),
                "drug2_id": drug_data.get(example["drug2_db"], -1),
            }
        )

        return dataset

    # Call the correct processing function
    if dataset_name == "drugbank":
        dataset = process_drugbank_dataset(dataset)
        columns_to_keep = [
            "drug1_db",
            "drug2_db",
            "drug1_id",
            "drug2_id",
            "drug_pair",
            "drug1_name",
            "drug2_name",
            "drug1_desc",
            "drug2_desc",
            "label",
            "label_idx",
        ]
    elif dataset_name == "ddinter":
        dataset = process_ddinter_dataset(dataset)
        # keep only the following columns
        columns_to_keep = [
            "drug1_db",
            "drug2_db",
            "drug1_id",
            "drug2_id",
            "drug_pair",
            "drug1_name",
            "drug2_name",
            "drug1_desc",
            "drug2_desc",
            "label",
            "label_idx",
        ]
    elif dataset_name == "pharmaDB":
        dataset = process_pharmaDB_dataset(dataset)
        columns_to_keep = [
        "drug1_db", "drug2_db", "drug1_id", "drug2_id", "drug_name",
        "drug_desc", "disease_name", "disease_desc", "label", "effect",
        "Y", "label_idx"]

    else:
        raise ValueError(f"Dataset {dataset_name} not recognized")
        
    # Load only a small subset for debugging
    if debug:
        dataset = dataset.select(range(min(5, len(dataset))))
        logger.info(f"DEBUG MODE: Using a small subset of the dataset ({len(dataset)} samples)")
    
    
    dataset = dataset.remove_columns(
        [col for col in dataset.column_names if col not in columns_to_keep]
    )

    # Log final dataset columns
    logger.info("Dataset loaded and processed")
    logger.info(f"Final dataset columns: {dataset.column_names}")

    return dataset
