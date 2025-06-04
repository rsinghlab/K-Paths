import os
import json
import random
import pandas as pd
import torch
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import argparse
import logging

from transformers import RobertaTokenizer, RobertaModel

from data_utils import (
    load_node_mapping,
    load_hetionet_graph,
    update_node_to_id_with_missing_drugs,
    create_edge_tensors,
    get_drug_descriptions,
    generate_embeddings
)
from model import RGCN, EdgeClassifier
from train_eval_utils import train_model

def parse_args():
    parser = argparse.ArgumentParser(description="RGCN Training")
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--train_file_path', type=str, required=True, help='Path to the training dataset CSV file')
    parser.add_argument('--hetionet_triplet_file', type=str, required=True, help='Path to the BKG triplet file')
    parser.add_argument('--node_file', type=str, required=True, help='Path to the node-to-ID mapping file')
    parser.add_argument('--entity_drug_file', type=str, required=True, help='Path to the entity-to-drug mapping file')
    parser.add_argument('--embedding_dim', type=int, default=768, help='Dimension of node embeddings')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Dimension of hidden layers in the RGCN model')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='Weight decay for the optimizer')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=400, help='Patience for early stopping')
    parser.add_argument('--max_samples_per_epoch', type=int, default=50000, help='Max samples per epoch')
    parser.add_argument('--num_samples_per_class', type=int, default=50000, help='Number of samples per class')
    parser.add_argument('--log_file', type=str, default="output/rgcn_training.log", help='Path to the log file')
    parser.add_argument('--model_save_path', type=str, default="output/trained_model.pt", help='Path to save the trained model')
    parser.add_argument('--use_text_embeddings', action='store_true', help='Use text-based embeddings for drugs')
    return parser.parse_args()

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_rgcn_model(args):
    """
    Train an RGCN model ensuring that training relations are not overwritten by hetionet relations.
    Relations:
        - Training Relations: Dynamically assigned based on unique Y values in training data
        - hetionet Relations: Appended after training relations with unique IDs
    """
    set_seed(args.seed)

    # Set up logging
    logging.basicConfig(
        filename=args.log_file,
        filemode='w',  
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load training data
    try:
        train_df = pd.read_csv(args.train_file_path, dtype={"Drug1_ID": str, "Drug2_ID": str})
        logging.info(f"Loaded training data with {len(train_df)} records from {args.train_file_path}")
    except Exception as e:
        logging.error(f"Failed to load training data: {e}")
        raise

    # Handle NaN values (if present)
    num_nans_train = train_df['Y'].isnull().sum()
    if num_nans_train > 0:
        logging.warning(f"Found {num_nans_train} NaN values in 'Y' column of training data. These rows will be removed.")
        train_df = train_df.dropna(subset=['Y'])
        logging.info(f"Training data after removing NaNs: {len(train_df)} records.")
    else:
        logging.info("No NaN values found in 'Y' column of training data.")

    # 4) Convert 'Y' column to int
    try:
        train_df['Y'] = train_df['Y'].astype(int)
        logging.info("Converted 'Y' column in training data to integer type.")
    except ValueError as e:
        logging.error(f"Error converting 'Y' column to int in training data: {e}")
        # Identify problematic rows
        problematic_rows = train_df[~train_df['Y'].astype(str).str.match(r'^\d+$')]
        logging.error(f"Problematic rows in 'Y' column:\n{problematic_rows}")
        raise

    # 5) Load and process Hetionet triplets
    try:
        hetionet_triplets = load_hetionet_graph(args.hetionet_triplet_file)
        logging.info(f"Loaded {len(hetionet_triplets)} hetionet triplets from {args.hetionet_triplet_file}")
    except Exception as e:
        logging.error(f"Failed to load hetionet triplets: {e}")
        raise

    # Convert Hetionet triplets to DataFrame 
    try:
        bkg_df = pd.DataFrame(hetionet_triplets, columns=["Drug1_ID", "Drug2_ID", "Y"])
        logging.info("Converted hetionet triplets to DataFrame.")
    except Exception as e:
        logging.error(f"Failed to process hetionet triplets: {e}")
        raise

    # Handle NaN values (if present in Hetionet)
    num_nans_bkg = bkg_df['Y'].isnull().sum()
    if num_nans_bkg > 0:
        logging.warning(f"Found {num_nans_bkg} NaN values in 'Y' column of BKG triplets. These rows will be removed.")
        bkg_df = bkg_df.dropna(subset=['Y'])
        logging.info(f"BKG data after removing NaNs: {len(bkg_df)} records.")
    else:
        logging.info("No NaN values found in 'Y' column of BKG triplets.")

    # Convert 'Y' column to int in Hetionet triplets
    try:
        bkg_df["Y"] = bkg_df["Y"].astype(int)
        logging.info("Converted 'Y' column in BKG data to integer type.")
    except ValueError as e:
        logging.error(f"Error converting 'Y' column to int in BKG data: {e}")
        problematic_bkg = bkg_df[~bkg_df['Y'].astype(str).str.match(r'^\d+$')]
        logging.error(f"Problematic rows in 'Y' column of BKG data:\n{problematic_bkg}")
        raise

    # Load node mapping
    try:
        node_to_id = load_node_mapping(args.node_file, args.entity_drug_file)
        logging.info(f"Loaded node mapping from {args.node_file} and {args.entity_drug_file}")
    except Exception as e:
        logging.error(f"Failed to load node mappings: {e}")
        raise

    # Collect all node IDs from training set and Hetionet graph
    train_nodes = set(train_df['Drug1_ID']).union(train_df['Drug2_ID'])
    bkg_nodes = set(bkg_df['Drug1_ID']).union(bkg_df['Drug2_ID'])
    all_nodes = train_nodes.union(bkg_nodes)
    logging.info(f"Total unique nodes from training and BKG data: {len(all_nodes)}")

    # Update node_to_id to include missing nodes 
    node_to_id = update_node_to_id_with_missing_drugs(node_to_id, all_nodes)
    num_nodes = max(node_to_id.values()) + 1
    logging.info(f"Updated node_to_id mapping to include all nodes. Total nodes: {num_nodes}")

    # 12) Save updated node_to_id
    try:
        with open("node_to_id.json", "w") as f:
            json.dump(node_to_id, f)
        logging.info(f"Saved updated node_to_id mapping to 'node_to_id.json'")
    except Exception as e:
        logging.error(f"Failed to save updated node_to_id mapping: {e}")
        raise

    # Filter rare classes before splitting
    # Calculate class counts
    class_counts = train_df['Y'].value_counts()
    # Identify classes with at least 2 samples
    valid_classes = class_counts[class_counts >= 2].index
    # Filter training data to include only valid classes
    initial_train_size = len(train_df)
    train_df = train_df[train_df['Y'].isin(valid_classes)]
    final_train_size = len(train_df)
    removed_classes = set(class_counts.index) - set(valid_classes)
    if removed_classes:
        logging.warning(f"Removed classes with fewer than 2 samples: {removed_classes}")
        logging.info(f"Filtered training data from {initial_train_size} to {final_train_size} records.")
    else:
        logging.info("No classes with fewer than 2 samples found.")

    # Split training data (train set + Hetionet) into train and validation 
    try:
        train_split, val_split = train_test_split(
            train_df,
            test_size=0.2,
            random_state=args.seed,
            stratify=train_df['Y']
        )
        logging.info(f"Split training data into {len(train_split)} training and {len(val_split)} validation records.")
    except ValueError as e:
        logging.error(f"Error during train_test_split: {e}")
        raise

    # Validate split 
    train_relations = set(train_split['Y'])
    val_relations = set(val_split['Y'])
    missing_relations = val_relations - train_relations
    if missing_relations:
        logging.warning(f"Relation types missing from training: {sorted(missing_relations)}")
        # Move samples from val to train to ensure all relations are present in training
        for relation in sorted(missing_relations):
            sample = val_split[val_split['Y'] == relation].iloc[0]
            train_split = pd.concat([train_split, sample.to_frame().T])
            val_split = val_split.drop(sample.name)
        logging.info(f"Revised split sizes: {len(train_split)} training and {len(val_split)} validation records.")
    else:
        logging.info("All validation relations are present in the training set.")

    # Generate relation_to_id mapping
    # Assign training relations first
    sorted_train_relations = sorted(train_relations)
    relation_to_id = {y: idx for idx, y in enumerate(sorted_train_relations)}
    logging.info(f"Assigned relation_to_id for training relations: {relation_to_id}")

    # Offset for Hetionet relations
    offset = len(relation_to_id)
    # Get unique Hetionet relations
    hetionet_relations = sorted(set(bkg_df['Y']))
    # Assign unique IDs to Hetionet relations, appending after training relations
    for y in hetionet_relations:
        if y in relation_to_id:
            # Overlapping relation: assign a new unique ID
            new_y = y + offset
            relation_to_id[new_y] = len(relation_to_id)
            # Update 'Y' in bkg_df to reflect the new relation ID
            bkg_df.loc[bkg_df['Y'] == y, 'Y'] = new_y
            logging.info(f"Overlapping hetionet relation {y} assigned new ID {new_y}")
        else:
            # Non-overlapping relation: assign next available ID
            relation_to_id[y] = len(relation_to_id)
            logging.info(f"hetionet relation {y} assigned ID {relation_to_id[y]}")

    num_relations = len(relation_to_id)
    logging.info(f"Generated relation_to_id mapping with {num_relations} unique relations.")

    # Create train_relation_to_id Mapping
    train_relation_to_id = {y: idx for idx, y in enumerate(sorted_train_relations)}
    logging.info(f"Created train_relation_to_id mapping: {train_relation_to_id}")

    # Load drug descriptions (if using text embeddings)
    drug_descriptions = None
    if args.use_text_embeddings:
        try:
            # Combine training split and Hetionet data for descriptions
            combined_train_bkg_df = pd.concat([train_split, bkg_df], ignore_index=True)
            drug_descriptions = get_drug_descriptions(combined_train_bkg_df)
            logging.info(f"Extracted descriptions for {len(drug_descriptions)} drugs.")
        except Exception as e:
            logging.error(f"Failed to get drug descriptions: {e}")
            raise

    # Load RoBERTa tokenizer and model
    roberta_tokenizer, roberta_model_instance = None, None
    if args.use_text_embeddings:
        try:
            roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
            roberta_model_instance = RobertaModel.from_pretrained("roberta-base").to(device)
            logging.info("Loaded RoBERTa tokenizer and model for text-based embeddings.")
        except Exception as e:
            logging.error(f"Failed to load RoBERTa tokenizer/model: {e}")
            raise

    # Generate embeddings
    try:
        combined_train_bkg_df = pd.concat([train_split, bkg_df], ignore_index=True)
        embeddings = generate_embeddings(
            node_to_id=node_to_id,
            drug_descriptions=drug_descriptions,
            tokenizer=roberta_tokenizer,
            roberta_model=roberta_model_instance,
            device=device,
            use_text_embeddings=args.use_text_embeddings
        )  # shape: [num_nodes, embedding_dim]
        logging.info(f"Generated embeddings with shape: {embeddings.shape}")
    except Exception as e:
        logging.error(f"Failed to generate embeddings: {e}")
        raise

    # Determine which embeddings to train: only for training drug IDs
    train_drug_ids = set(train_split['Drug1_ID']).union(train_split['Drug2_ID'])

    # Initialize mask: True = frozen, False = trainable
    frozen_mask = torch.ones(num_nodes, dtype=torch.bool, device=device)

    # Mark training drugs as trainable (i.e., not frozen)
    for drug_id in train_drug_ids:
        idx = node_to_id.get(drug_id)
        if idx is not None:
            frozen_mask[idx] = False

    # Split embeddings into frozen and trainable sets
    frozen_embeddings = embeddings[frozen_mask].detach()
    trainable_embeddings = embeddings[~frozen_mask]  # will be updated

    logging.info(f"Froze {frozen_embeddings.shape[0]} embeddings, {trainable_embeddings.shape[0]} embeddings are trainable.")

    # Map edges for training and validation 
    def map_edges_for_classification(df, relation_to_id, classification_relation_to_id=None):
        """
        Maps Drug IDs to their integer IDs and relations to class IDs.
        If classification_relation_to_id is provided, maps relations to classification labels.
        Otherwise, uses the provided relation_to_id.
        Skips edges with missing Drug IDs or unseen relations.
        """
        mapped = []
        for drug1, drug2, y in df[['Drug1_ID', 'Drug2_ID', 'Y']].values.tolist():
            if classification_relation_to_id:
                if y in classification_relation_to_id:
                    if drug1 in node_to_id and drug2 in node_to_id:
                        mapped.append([node_to_id[drug1], node_to_id[drug2], classification_relation_to_id[y]])
                    else:
                        logging.warning(f"Drug IDs ({drug1}, {drug2}) not found in node_to_id. Skipping edge.")
                else:
                    # Ignore hetionet relations in classification
                    continue
            else:
                if y in relation_to_id:
                    if drug1 in node_to_id and drug2 in node_to_id:
                        mapped.append([node_to_id[drug1], node_to_id[drug2], relation_to_id[y]])
                    else:
                        logging.warning(f"Drug IDs ({drug1}, {drug2}) not found in node_to_id. Skipping edge.")
                else:
                    logging.warning(f"Relation '{y}' not recognized. Skipping edge ({drug1}, {drug2}, {y}).")
        return mapped

    # Map Training Edges for Graph (including hetionet)
    combined_train_bkg_df = pd.concat([train_split, bkg_df], ignore_index=True)
    mapped_train_edges = map_edges_for_classification(combined_train_bkg_df, relation_to_id)
    train_edge_index, train_edge_type = create_edge_tensors(mapped_train_edges, device)
    logging.info(f"Mapped {len(mapped_train_edges)} training edges for the graph.")

    # Map Classification Training Edges (only training relations)
    mapped_class_train_edges = map_edges_for_classification(train_split, relation_to_id, train_relation_to_id)
    train_class_edge_index, train_class_edge_type = create_edge_tensors(mapped_class_train_edges, device)
    logging.info(f"Mapped {len(mapped_class_train_edges)} training edges for classification.")

    # Map Classification Validation Edges (only training relations)
    mapped_class_val_edges = map_edges_for_classification(val_split, relation_to_id, train_relation_to_id)
    val_class_edge_index, val_class_edge_type = create_edge_tensors(mapped_class_val_edges, device)
    logging.info(f"Mapped {len(mapped_class_val_edges)} validation edges for classification.")

    # 22) Create Combined Edge Tensors (Training + Validation) for Graph
    combined_edge_index = train_edge_index  # Only training edges (including hetionet) are used for the graph
    combined_edge_type = train_edge_type
    logging.info("Created combined edge tensors for the graph.")

    # Initialize RGCN model
    model = RGCN(
        num_nodes=num_nodes,
        embedding_dim=embeddings.shape[1],
        hidden_dim=args.hidden_dim,
        num_relations=num_relations  # All relations for graph
    ).to(device)
    logging.info("Initialized RGCN model.")

    # Assign embeddings to RGCN model
    final_embeddings = embeddings.clone()
    final_embeddings[frozen_mask] = frozen_embeddings
    final_embeddings[~frozen_mask] = trainable_embeddings
    model.embeddings.weight.data = final_embeddings
    logging.info("Assigned combined embeddings to RGCN model.")

    # Initialize EdgeClassifier 
    edge_classifier = EdgeClassifier(
        node_embedding_dim=args.hidden_dim,
        hidden_dim=128,
        num_classes=len(train_relation_to_id)  # Only training relations
    ).to(device)
    logging.info("Initialized EdgeClassifier for training relations only.")

    # Optimizer and scheduler
    trainable_params = [
        {"params": model.parameters()},
        {"params": trainable_embeddings}
    ]
    optimizer = AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5, verbose=True)
    logging.info("Configured optimizer and scheduler.")

    #  Train
    try:
        train_model(
            model=model,
            edge_classifier=edge_classifier,
            optimizer=optimizer,
            scheduler=scheduler,
            combined_train_edge_index=combined_edge_index,  # entire adjacency for RGCN
            combined_train_edge_type=combined_edge_type,
            train_edge_index=train_class_edge_index,  # classification train edges
            train_edge_type=train_class_edge_type,    # classification train labels
            val_edge_index=val_class_edge_index,      # classification val edges
            val_edge_type=val_class_edge_type,        # classification val labels
            num_classes=len(train_relation_to_id),    # number of training relation classes
            epochs=args.epochs,
            log_file="output/training_log.txt",
            device=device,
            patience=args.patience,
            max_samples_per_epoch=args.max_samples_per_epoch,
            num_samples_per_class=args.num_samples_per_class
        )
        logging.info("Training completed successfully.")
    except Exception as e:
        logging.error(f"An error occurred during training: {e}")
        raise

    # Save model
    try:
        torch.save({
            'model_state_dict': model.state_dict(),
            'edge_classifier_state_dict': edge_classifier.state_dict(),
            'relation_to_id': relation_to_id,               # All relations
            'train_relation_to_id': train_relation_to_id,   # Only training relations
            'node_to_id': node_to_id                        # Node mapping
        }, args.model_save_path)
        logging.info(f"Model and mappings saved to {args.model_save_path}")
    except Exception as e:
        logging.error(f"Failed to save the model: {e}")
        raise


def main():
    args = parse_args()
    train_rgcn_model(args)

if __name__ == "__main__":
    main()
