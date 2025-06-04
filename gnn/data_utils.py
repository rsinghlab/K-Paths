import os
import json
import logging
import numpy as np
import torch
import pandas as pd

def load_node_mapping(node_file, entity_drug_file):
    """
    Loads a combined node-to-ID mapping from drug and gene JSON files.

    Args:
        node_file (str): Path to the JSON file containing drug nodes.
        entity_drug_file (str): Path to the JSON file containing gene nodes.

    Returns:
        dict: Combined {node_name: node_id} mapping.
    """
    with open(node_file, 'r') as f:
        node_to_id_drugs = json.load(f)
    with open(entity_drug_file, 'r') as f:
        node_to_id_genes = json.load(f)

    node_to_id = {**node_to_id_drugs, **node_to_id_genes}
    logging.info(f"Loaded node_to_id mapping with {len(node_to_id)} nodes.")
    return node_to_id

def load_hetionet_graph(triplet_file):
    """
    Loads Hetionet triplets from a text file.

    Args:
        triplet_file (str): Path to file with tab-separated (head, tail, relation_id).

    Returns:
        List[List[int]]: List of [head_id, tail_id, relation_id] triplets.
    """
    if not os.path.exists(triplet_file):
        logging.warning(f"Triplet file {triplet_file} does not exist. No hetionet triplets loaded.")
        return []

    hetionet_triplets_raw = np.loadtxt(triplet_file, dtype=int)  # head_id tail_id relation_type
    hetionet_triplets = hetionet_triplets_raw.tolist()
    logging.info(f"Loaded {len(hetionet_triplets)} hetionet triplets.")
    return hetionet_triplets

def no_hetionet(triplet_file):
    """
    Dummy function for disabling Hetionet loading.

    Returns:
        list: Empty list of triplets.
    """
    logging.warning("Hetionet graph loading is disabled.")
    return []

def get_all_drug_ids(train_df, test_df):
    """
    Extracts all unique drug IDs from train and test sets.

    Args:
        train_df (pd.DataFrame): Training data.
        test_df (pd.DataFrame): Test data.

    Returns:
        set: Unique drug identifiers.
    """
    drugs1 = set(train_df['Drug1_ID'])
    drugs2 = set(train_df['Drug2_ID'])
    drugs_test1 = set(test_df['Drug1_ID'])
    drugs_test2 = set(test_df['Drug2_ID'])
    all_drug_ids = drugs1.union(drugs2).union(drugs_test1).union(drugs_test2)
    logging.info(f"Total unique drug IDs extracted from datasets: {len(all_drug_ids)}")
    return all_drug_ids

def get_train_drug_ids(train_df):
    """
    Extracts unique drug IDs from the training dataset.

    Args:
        train_df (pd.DataFrame): Training data.

    Returns:
        set: Unique drug identifiers.
    """
    drugs1 = set(train_df['Drug1_ID'])
    drugs2 = set(train_df['Drug2_ID'])
    train_drug_ids = drugs1.union(drugs2)
    logging.info(f"Total unique drug IDs extracted from training data: {len(train_drug_ids)}")
    return train_drug_ids

def update_node_to_id_with_missing_drugs(node_to_id, drug_ids):
    """
    Adds missing drug IDs to the node_to_id mapping.

    Args:
        node_to_id (dict): Existing {node_name: node_id} mapping.
        drug_ids (set): Drug IDs to be checked and added.

    Returns:
        dict: Updated node_to_id mapping.
    """
    node_to_id_keys = set(node_to_id.keys())
    missing_drugs = drug_ids - node_to_id_keys
    if missing_drugs:
        max_node_id = max(node_to_id.values())
        for drug_id in missing_drugs:
            max_node_id += 1
            node_to_id[drug_id] = max_node_id
        logging.info(f"Added {len(missing_drugs)} missing drug IDs to node_to_id.")
    else:
        logging.info("No missing drug IDs to add to node_to_id.")
    return node_to_id


def get_descriptions_drugbank(df):
    """
    Extracts drug descriptions from DrugBank-formatted DataFrame.

    Args:
        df (pd.DataFrame): Dataset with Drug1_ID, Drug2_ID, and descriptions.

    Returns:
        dict: {Drug_ID: Description} mapping.
    """
    drugs1 = df[['Drug1_ID', 'Drug1_Description']].rename(
        columns={'Drug1_ID': 'Drug_ID', 'Drug1_Description': 'Drug_Description'}
    )
    drugs2 = df[['Drug2_ID', 'Drug2_Description']].rename(
        columns={'Drug2_ID': 'Drug_ID', 'Drug2_Description': 'Drug_Description'}
    )

    all_drugs = pd.concat([drugs1, drugs2], ignore_index=True)
    all_drugs = all_drugs.drop_duplicates(subset='Drug_ID')
    all_drugs['Drug_Description'] = all_drugs['Drug_Description'].fillna('').astype(str)

    drug_descriptions = dict(zip(all_drugs['Drug_ID'], all_drugs['Drug_Description']))
    logging.info(f"Extracted descriptions for {len(drug_descriptions)} drugs.")
    return drug_descriptions

def get_descriptions_ddinter(df):
    """
    Extracts drug descriptions from DrugBank-formatted DataFrame.

    Args:
        df (pd.DataFrame): Dataset with Drug1_ID, Drug2_ID, and descriptions.

    Returns:
        dict: {Drug_ID: Description} mapping.
    """
    drugs1 = df[['Drug1_ID', 'drug1_desc']].rename(
        columns={'Drug1_ID': 'Drug_ID', 'drug1_desc': 'Drug_Description'}
    )
    drugs2 = df[['Drug2_ID', 'drug2_desc']].rename(
        columns={'Drug2_ID': 'Drug_ID', 'drug2_desc': 'Drug_Description'}
    )

    all_drugs = pd.concat([drugs1, drugs2], ignore_index=True)
    all_drugs = all_drugs.drop_duplicates(subset='Drug_ID')
    all_drugs['Drug_Description'] = all_drugs['Drug_Description'].fillna('').astype(str)

    drug_descriptions = dict(zip(all_drugs['Drug_ID'], all_drugs['Drug_Description']))
    logging.info(f"Extracted descriptions for {len(drug_descriptions)} drugs.")
    return drug_descriptions


def get_descriptions_pdb(df):
    """
    Extracts descriptions for drugs and diseases from a PDB-formatted DataFrame.

    Args:
        df (pd.DataFrame): Dataset with drugbank_id, doid_id, and their names.

    Returns:
        dict: {entity_id: description} mapping for drugs and diseases.
    """
    drugs = df[['drugbank_id', 'drug_name']].rename(
        columns={'drugbank_id': 'entity_id', 'drug_name': 'description'}
    )
    diseases = df[['doid_id', 'disease_name']].rename(
        columns={'doid_id': 'entity_id', 'disease_name': 'description'}
    )

    combined = pd.concat([drugs, diseases], ignore_index=True).drop_duplicates(subset='entity_id')
    combined['description'] = combined['description'].fillna('').astype(str)

    descriptions = dict(zip(combined['entity_id'], combined['description']))
    logging.info(f"Extracted descriptions for {len(descriptions)} entities (drugs + diseases).")
    return descriptions


def map_Y_to_relation_id_drugbank(triplets, relation_to_id, node_to_id, max_relations, is_test=False):
    """
    Maps triplets to (head_idx, tail_idx, relation_id) using DrugBank-specific rules.

    Args:
        triplets (List[List[str]]): Input triplets [h, t, y].
        relation_to_id (dict): Mapping from relation name to integer ID.
        node_to_id (dict): Mapping from node names to indices.
        max_relations (int): Maximum allowed relation ID.
        is_test (bool): If True, skip relation checks.

    Returns:
        List[List[int]]: Mapped triplets [h_idx, t_idx, relation_id].
    """
    mapped_edges = []
    for h, t, y in triplets:
        h_idx = node_to_id.get(h)
        t_idx = node_to_id.get(t)
        y_id = relation_to_id.get(y)

        if h_idx is None or t_idx is None:
            logging.warning(f"Skipping triplet with missing node mapping: {h}, {t}, {y}")
            continue

        if y_id is None or (not is_test and y_id >= max_relations):
            logging.warning(f"Invalid or out-of-range Y value '{y}', skipping triplet {h, t, y}")
            continue

        mapped_edges.append([h_idx, t_idx, y_id])
    return mapped_edges

def map_Y_to_relation_id_pdb(triplets, relation_to_id, node_to_id, num_classes, is_test=False):
    mapped_triplets = []
    skipped_triplets = []

    for triplet in triplets:
        try:
            h, t, Y = triplet

            if not is_test:
                relation_id = relation_to_id.get(Y)
            else:
                relation_id = -1 

            h_idx = node_to_id.get(h)
            t_idx = node_to_id.get(t)

            if h_idx is None or t_idx is None or relation_id is None:
                skipped_triplets.append(triplet)
                continue

            mapped_triplets.append([h_idx, t_idx, relation_id])

        except ValueError:
            skipped_triplets.append(triplet)
            continue

    if skipped_triplets:
        logging.warning(f"Skipped {len(skipped_triplets)} triplets due to missing data or mapping errors.")
    return mapped_triplets

def map_Y_to_relation_id_ddinter(triplets, relation_to_id, node_to_id, is_test=False):
    """
    Maps 'Y' in triplets to relation_ids and converts Drug IDs to node indices.

    :param triplets: list of [Drug1_ID, Drug2_ID, Y] or [Drug1_ID, Drug2_ID]
    :param relation_to_id: dict, {relation_name: relation_id, ...}
    :param node_to_id: dict, {node_name: node_id, ...}
    :param is_test: bool, if True, treat triplets as test set without Y
    :return: list of [Drug1_idx, Drug2_idx, relation_id] or [Drug1_idx, Drug2_idx]
    """
    mapped_triplets = []
    skipped_triplets = []

    for triplet in triplets:
        if is_test:
            if len(triplet) < 2:
                logging.warning(f"Invalid test triplet format: {triplet}")
                skipped_triplets.append(triplet)
                continue
            h, t = triplet[:2]
            Y = None
        else:
            if len(triplet) < 3:
                logging.warning(f"Invalid training triplet format: {triplet}")
                skipped_triplets.append(triplet)
                continue
            h, t, Y = triplet[:3]

        h_idx = node_to_id.get(h, None)
        t_idx = node_to_id.get(t, None)
        if h_idx is None or t_idx is None:
            skipped_triplets.append(triplet)
            logging.warning(f"Drug ID '{h}' or '{t}' not found in node_to_id mapping, skipping triplet {triplet}")
            continue

        if is_test:
            relation_id = -1  
        else:
            relation_id = relation_to_id.get(Y, None)
            if relation_id is None:
                skipped_triplets.append(triplet)
                logging.warning(f"Relation '{Y}' not found in relation_to_id mapping, skipping triplet {triplet}")
                continue

        mapped_triplets.append([h_idx, t_idx, relation_id])

    if skipped_triplets:
        logging.info(f"Total skipped triplets: {len(skipped_triplets)}")
    return mapped_triplets


def generate_embeddings(node_to_id, drug_descriptions, embedding_save_path, tokenizer, roberta_model, device):
    """
    Generates embeddings for all nodes: drugs using RoBERTa and genes as random vectors.

    Args:
        node_to_id (dict): Mapping from node names to indices.
        drug_descriptions (dict): {Drug_ID: Description}.
        embedding_save_path (str): Path to save the embeddings.
        tokenizer: HuggingFace tokenizer for the RoBERTa model.
        roberta_model: HuggingFace RoBERTa model.
        device (torch.device): Target device for computation.

    Returns:
        torch.Tensor: Embeddings of shape [num_nodes, embedding_dim].
    """
    embedding_dim = 768  # Projection size
    projection_layer = torch.nn.Linear(roberta_model.config.hidden_size, embedding_dim).to(device)

    max_node_idx = max(node_to_id.values())
    total_nodes = max_node_idx + 1

    logging.info(f"Total nodes (max node index): {total_nodes}")

    embeddings = torch.zeros((total_nodes, embedding_dim), device=device)

    logging.info("Generating embeddings for drugs and genes...")
    for node_name, node_idx in node_to_id.items():
        if node_name.startswith("DB"):  # Drug nodes
            description = drug_descriptions.get(node_name, "")
            if description:
                inputs = tokenizer(
                    description,
                    return_tensors="pt",
                    truncation=True,
                    padding="max_length",
                    max_length=512
                ).to(device)
                with torch.no_grad():
                    outputs = roberta_model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0)
                embedding = projection_layer(embedding) 
                embeddings[node_idx] = embedding
            else:
                embeddings[node_idx] = torch.zeros(embedding_dim, device=device)
        elif node_name.startswith("Gene::"):  # Gene nodes
            embeddings[node_idx] = torch.randn(embedding_dim, device=device)
        else:
            logging.warning(f"Unknown node type for {node_name}. Using random embedding.")
            embeddings[node_idx] = torch.randn(embedding_dim, device=device)

    return embeddings

def generate_random_embeddings(node_to_id, embedding_dim, device):
    """
    Generates random embeddings for all nodes.

    Args:
        node_to_id (dict): Mapping from node names to indices.
        embedding_dim (int): Size of each embedding.
        device (torch.device): Device to store the tensor.

    Returns:
        torch.Tensor: Random embeddings of shape [num_nodes, embedding_dim].
    """
    max_node_idx = max(node_to_id.values())
    total_nodes = max_node_idx + 1

    logging.info(f"Generating random embeddings for {total_nodes} nodes...")

    # Create random embeddings for all nodes
    embeddings = torch.randn((total_nodes, embedding_dim), device=device)

    logging.info("Random embeddings generated successfully.")
    return embeddings

def load_embeddings(embedding_save_path, device):
    """
    Loads saved embeddings from disk.

    Args:
        embedding_save_path (str): Path to saved file (dict with "embeddings" key).
        device (torch.device): Target device for loading.

    Returns:
        torch.Tensor: Loaded embedding matrix.
    """
    embeddings_dict = torch.load(embedding_save_path, map_location='cpu')
    if "embeddings" not in embeddings_dict:
        raise KeyError("'embeddings' key not found in the saved file.")
    embeddings = embeddings_dict["embeddings"].to(device)
    logging.info(f"'embeddings' tensor loaded from {embedding_save_path} and moved to {device}.")
    return embeddings

def save_sorted_embeddings(embeddings, node_to_id, drug_descriptions, save_path, indices_txt_path="drug_indices.txt"):
    """
    Saves embeddings with drugs sorted to the front of the tensor, plus index metadata.

    Args:
        embeddings (torch.Tensor): Embedding matrix.
        node_to_id (dict): Mapping of node names to indices.
        drug_descriptions (dict): {Drug_ID: Description}.
        save_path (str): Path to save .pt file.
        indices_txt_path (str): Path to write sorted drug index mapping.
    """
    drug_names = [name for name in node_to_id if name.startswith("DB")]
    drug_indices = [node_to_id[name] for name in drug_names]
    drug_embeddings = embeddings[drug_indices]

    sorted_indices = np.argsort(drug_names)
    sorted_drug_names = [drug_names[i] for i in sorted_indices]
    sorted_drug_embeddings = drug_embeddings[sorted_indices]

    full_embeddings = embeddings.clone() 
    full_embeddings[:len(sorted_drug_embeddings)] = sorted_drug_embeddings 

    start_end_indices = {
        name: (i, i) for i, name in enumerate(sorted_drug_names)
    }

    with open(indices_txt_path, "w") as f:
        for name, (start, end) in start_end_indices.items():
            f.write(f"{name}\t{start}\t{end}\n")
    logging.info(f"Start and end indices saved to {indices_txt_path}")

    data_to_save = {
        "embeddings": full_embeddings,  # Full embeddings with sorted drugs at the top
        "sorted_drug_embeddings": sorted_drug_embeddings,  # Sorted drug embeddings only
    }
    torch.save(data_to_save, save_path)
    logging.info(f"Saved embeddings with sorted drugs and metadata to {save_path}")

def create_edge_tensors(triplets, device):
    """
    Converts list of triplets to PyTorch tensors.

    Args:
        triplets (List[List[int]]): [head_idx, tail_idx, relation_id].
        device (torch.device): Target device.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: edge_index [2, num_edges], edge_type [num_edges].
    """
    edge_index = []
    edge_type = []

    for triplet in triplets:
        h, t, r = triplet
        edge_index.append([h, t])
        edge_type.append(r)

    if len(edge_index) == 0:
        logging.warning("No edges found for the given triplets.")
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        edge_type = torch.empty((0,), dtype=torch.long, device=device)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long, device=device).t().contiguous()
        edge_type = torch.tensor(edge_type, dtype=torch.long, device=device)

    return edge_index, edge_type

def check_edge_indices(edge_index, edge_name, num_nodes):
    """
    Checks whether edge indices are within the allowed node range.

    Args:
        edge_index (torch.Tensor): Tensor of shape [2, num_edges].
        edge_name (str): Label used in logging.
        num_nodes (int): Total number of valid node indices.

    Logs:
        Warnings or errors if the indices are out of range.
    """
    if edge_index.numel() == 0:
        logging.warning(f"{edge_name} has no edges.")
        return
    max_node_idx = edge_index.max().item()
    min_node_idx = edge_index.min().item()
    logging.info(f"{edge_name}: min node index = {min_node_idx}, max node index = {max_node_idx}")
    if max_node_idx >= num_nodes:
        logging.error(f"{edge_name} contains node index {max_node_idx} >= num_nodes {num_nodes}.")

