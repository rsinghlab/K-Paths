import os
import json
import random
import logging
import argparse
import numpy as np
import pandas as pd
import torch
import warnings
from transformers import RobertaTokenizer, RobertaModel
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

warnings.filterwarnings("ignore", category=FutureWarning, message="You are using torch.load")

from data_utils import load_hetionet_graph, create_edge_tensors, update_node_to_id_with_missing_drugs, generate_embeddings, get_drug_descriptions
from model import RGCN, EdgeClassifier

def parse_args():
    parser = argparse.ArgumentParser(description="RGCN Evaluation Script")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    # Paths
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained checkpoint (.pt) file")
    parser.add_argument("--node_to_id_path", type=str, required=True, help="Path to the original node_to_id JSON used during training")
    parser.add_argument("--train_file_path", type=str, required=True, help="The original training edges CSV (train + maybe dev, etc.)")
    parser.add_argument("--hetionet_triplet_file", type=str, required=True, help="The complete BKG used during training")
    parser.add_argument("--test_file", type=str, required=True, help="The test set CSV with new test drugs (columns: Drug1_ID, Drug2_ID, Y?, Drug1_Name, Drug2_Name, Drug1_Description, Drug2_Description)")
    parser.add_argument("--test_bkg_file", type=str, default=None, help="(Optional) Additional edges for test. If it’s truly a subset of the complete BKG, you might skip this.")
    
    # Output Paths
    parser.add_argument("--output_predictions", type=str, default="test_predictions.csv", help="Path to save the prediction CSV")
    parser.add_argument("--eval_log_file", type=str, default="output/evaluation_metrics.txt", help="Path to save evaluation logs")
    # parser.add_argument("--eval_results_file", type=str, default="output/evaluation_results.txt", help="Path to save evaluation results in the results file")
    parser.add_argument("--log_file", type=str, default="output/rgcn_inference.log", help="Path to save inference logs")

    # Model/embedding hyperparameters
    parser.add_argument("--embedding_dim", type=int, default=768, help="Dimension of node embeddings")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Dimension of hidden layers in the RGCN model")
    parser.add_argument("--use_text_embeddings", action="store_true", help="Generate text embeddings for newly added nodes. Otherwise random.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the evaluation on (e.g., 'cuda', 'cpu')")
    
    return parser.parse_args()

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_checkpoint(model_path: str, device: torch.device):
    """Load the model checkpoint."""
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model checkpoint file '{model_path}' does not exist.")
    ckpt = torch.load(model_path, map_location=device)
    return ckpt

def load_node_to_id(node_to_id_path: str):
    """Load node_to_id mapping from JSON."""
    if not os.path.isfile(node_to_id_path):
        raise FileNotFoundError(f"node_to_id file '{node_to_id_path}' does not exist.")
   
    with open(node_to_id_path, "r") as f:
        node_to_id = json.load(f)
        logging.info(f"Successfully loaded node_to_id with {len(node_to_id)} nodes.")
    return node_to_id

def load_relations(ckpt: dict):
    """Load relation mappings from the checkpoint."""
    if 'relation_to_id' not in ckpt:
        raise KeyError("relation_to_id not found in the checkpoint.")
    relation_to_id = ckpt['relation_to_id']
    num_relations = len(relation_to_id)
    
    if 'train_relation_to_id' not in ckpt:
        raise KeyError("train_relation_to_id not found in the checkpoint.")
    train_relation_to_id = ckpt['train_relation_to_id']
    num_train_relations = len(train_relation_to_id)
    
    return relation_to_id, num_relations, train_relation_to_id, num_train_relations

def load_train_data(train_file_path: str, hetionet_triplet_file: str):
    """Load training data and graph."""
    if not os.path.isfile(train_file_path):
        raise FileNotFoundError(f"Training file '{train_file_path}' does not exist.")
    train_df = pd.read_csv(train_file_path, dtype={"Drug1_ID": str, "Drug2_ID": str})
    
    hetionet_triplets = load_hetionet_graph(hetionet_triplet_file)
    bkg_df = pd.DataFrame(hetionet_triplets, columns=["Drug1_ID", "Drug2_ID", "Y"])
    bkg_df["Y"] = bkg_df["Y"].astype(int)
    
    combined_train_bkg_df = pd.concat([train_df, bkg_df], ignore_index=True)
    combined_train_bkg_df["Y"] = combined_train_bkg_df["Y"].astype(int)
    
    return train_df, combined_train_bkg_df

def load_test_data(test_file: str):
    """Load test data."""
    if not os.path.isfile(test_file):
        raise FileNotFoundError(f"Test file '{test_file}' does not exist.")
    test_df = pd.read_csv(test_file, dtype={"Drug1_ID": str, "Drug2_ID": str})
    if "Y" not in test_df.columns:
        test_df["Y"] = -1  # Placeholder for unknown labels
    return test_df


def load_test_bkg(test_bkg_file: str, train_relation_to_id: dict, node_to_id: dict, device: torch.device):
    """Load and map test BKG edges, ensuring only training relations are included."""
    if test_bkg_file and os.path.exists(test_bkg_file):
        test_bkg_triplets = load_hetionet_graph(test_bkg_file)
        test_bkg_df = pd.DataFrame(test_bkg_triplets, columns=["Drug1_ID", "Drug2_ID", "Y"])
        test_bkg_df["Y"] = test_bkg_df["Y"].astype(int)
        logging.info(f"Loaded test BKG with {len(test_bkg_triplets)} triplets.")
        
        # Map test BKG triplets using train_relation_to_id to ensure only training relations are included
        mapped_test_bkg_edges = []
        for _, row in test_bkg_df.iterrows():
            y = row["Y"]
            if y in train_relation_to_id:
                if row["Drug1_ID"] in node_to_id and row["Drug2_ID"] in node_to_id:
                    mapped_test_bkg_edges.append([
                        node_to_id[row["Drug1_ID"]],
                        node_to_id[row["Drug2_ID"]],
                        train_relation_to_id[y]
                    ])
                else:
                    logging.warning(f"Drug IDs ({row['Drug1_ID']}, {row['Drug2_ID']}) not found in node_to_id. Skipping test BKG edge.")
            else:
                # Skip triplet with unseen relation and log a warning
                logging.warning(f"Relation '{y}' in test BKG not found in training relations. Skipping triplet ({row['Drug1_ID']}, {row['Drug2_ID']}, {y}).")
        
        # Convert to tensors
        test_bkg_edge_index, test_bkg_edge_type = create_edge_tensors(mapped_test_bkg_edges, device)
        logging.info(f"Mapped {test_bkg_edge_index.shape[1]} test BKG edges after filtering.")
        return test_bkg_edge_index, test_bkg_edge_type
    else:
        logging.info("No test BKG file provided or file does not exist. Proceeding without test BKG.")
        return None, None


def prepare_embeddings(args, node_to_id: dict, drug_descriptions: dict, device: torch.device):
    """Generate or initialize node embeddings."""
    num_nodes = len(node_to_id)
    if args.use_text_embeddings:
        try:
            tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
            roberta_model = RobertaModel.from_pretrained("roberta-base").to(device)
            logging.info("Loaded RoBERTa tokenizer and model for text-based embeddings.")
        except Exception as e:
            logging.error(f"Failed to load RoBERTa tokenizer/model: {e}")
            raise
    else:
        tokenizer, roberta_model = None, None
        logging.info("Using random embeddings as text embeddings are not enabled.")
    
    try:
        if not args.use_text_embeddings:
            # If not using text embeddings, initialize random embeddings
            base_embeddings = torch.randn(num_nodes, args.embedding_dim, device=device)
            logging.info(f"Initialized random embeddings with shape: {base_embeddings.shape}")
        else:
            # Generate text-based embeddings
            base_embeddings = generate_embeddings(
                node_to_id=node_to_id,
                drug_descriptions=drug_descriptions,
                tokenizer=tokenizer,
                roberta_model=roberta_model,
                device=device,
                use_text_embeddings=args.use_text_embeddings
            )  # shape: [num_nodes, embedding_dim]
            logging.info(f"Generated text-based embeddings with shape: {base_embeddings.shape}")
    except Exception as e:
        logging.error(f"Failed to generate embeddings: {e}")
        raise
    
    return base_embeddings


def initialize_model(args, num_nodes: int, base_embeddings: torch.Tensor, num_relations: int, train_relation_to_id: dict, device: torch.device):
    """Initialize RGCN model and EdgeClassifier."""
    model = RGCN(
        num_nodes=num_nodes,
        embedding_dim=base_embeddings.shape[1],
        hidden_dim=args.hidden_dim,
        num_relations=num_relations  # All relations for graph
    ).to(device)
    logging.info("Initialized RGCN model.")
    
    edge_classifier = EdgeClassifier(
        node_embedding_dim=args.hidden_dim,
        hidden_dim=128,
        num_classes=len(train_relation_to_id)  # Only training relations
    ).to(device)
    logging.info("Initialized EdgeClassifier for training relations only.")
    
    return model, edge_classifier


def load_model_states(ckpt: dict, model: RGCN, edge_classifier: EdgeClassifier, device: torch.device, num_nodes: int, base_embeddings: torch.Tensor, train_relation_to_id: dict):
    """Load model and EdgeClassifier states from checkpoint."""
    if 'model_state_dict' in ckpt:
        # Extract embeddings.weight before deleting
        if 'embeddings.weight' in ckpt['model_state_dict']:
            checkpoint_embeddings = ckpt['model_state_dict']['embeddings.weight']
            checkpoint_num_nodes, embedding_dim = checkpoint_embeddings.shape
            logging.info(f"Original number of nodes in training embeddings: {checkpoint_num_nodes}")
            del ckpt['model_state_dict']['embeddings.weight']  
            logging.info("Removed 'embeddings.weight' from checkpoint state_dict.")
        else:
            checkpoint_embeddings = None
            checkpoint_num_nodes = 0
            logging.warning("No 'embeddings.weight' found in checkpoint.")
        
        # Load the rest of the state_dict
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        logging.info("Loaded RGCN model state_dict with strict=False.")
    else:
        logging.error("model_state_dict not found in the checkpoint.")
        raise KeyError("model_state_dict not found in the checkpoint.")
    
    if 'edge_classifier_state_dict' in ckpt:
        edge_classifier.load_state_dict(ckpt['edge_classifier_state_dict'], strict=True)
        logging.info("Loaded EdgeClassifier state_dict.")
    else:
        logging.error("edge_classifier_state_dict not found in the checkpoint.")
        raise KeyError("edge_classifier_state_dict not found in the checkpoint.")
    
    # Assign precomputed embeddings to training nodes and initialize new embeddings
    if checkpoint_embeddings is not None:
        # Ensure embedding dimensions match
        if checkpoint_embeddings.shape[1] != base_embeddings.shape[1]:
            logging.error(f"Embedding dimension mismatch: checkpoint={checkpoint_embeddings.shape[1]}, current={base_embeddings.shape[1]}")
            raise ValueError("Embedding dimension mismatch between checkpoint and current model.")
        
        # Assign existing embeddings
        if checkpoint_num_nodes > num_nodes:
            logging.error("Checkpoint has more nodes than the current node_to_id mapping.")
            raise ValueError("Checkpoint contains more nodes than current node_to_id mapping.")
        
        model.embeddings.weight.data[:checkpoint_num_nodes] = checkpoint_embeddings
        logging.info(f"Assigned {checkpoint_num_nodes} pre-trained embeddings to the model.")
        
        # Initialize new embeddings randomly
        if num_nodes > checkpoint_num_nodes:
            with torch.no_grad():
                # Use the same initialization as used during training
                model.embeddings.weight.data[checkpoint_num_nodes:] = torch.randn(
                    (num_nodes - checkpoint_num_nodes, base_embeddings.shape[1]),
                    device=device
                ) * 0.01  # Standard deviation can be adjusted as needed
            logging.info(f"Initialized {num_nodes - checkpoint_num_nodes} new embeddings randomly.")
    else:
        logging.warning("No pre-trained embeddings found in checkpoint. Initializing all embeddings randomly.")
        with torch.no_grad():
            model.embeddings.weight.data = base_embeddings
        logging.info("Initialized all embeddings randomly.")
    
    # Optional: Validate embedding assignment
    if checkpoint_embeddings is not None and checkpoint_num_nodes > 0:
        if torch.allclose(model.embeddings.weight.data[0], checkpoint_embeddings[0]):
            logging.info("First embedding matches the checkpoint. Assignment successful.")
        else:
            logging.warning("First embedding does not match the checkpoint. Check node_to_id consistency.")


def map_edges_for_graph(df: pd.DataFrame, relation_to_id: dict, node_to_id: dict):
    """Map edges to numerical IDs based on relation_to_id and node_to_id."""
    mapped = []
    for drug1, drug2, y in df[['Drug1_ID', 'Drug2_ID', 'Y']].values.tolist():
        if y in relation_to_id:
            if drug1 in node_to_id and drug2 in node_to_id:
                mapped.append([node_to_id[drug1], node_to_id[drug2], relation_to_id[y]])
            else:
                logging.warning(f"Drug IDs ({drug1}, {drug2}) not found in node_to_id. Skipping edge.")
        else:
            logging.warning(f"Relation '{y}' not recognized. Skipping edge ({drug1}, {drug2}, {y}).")
    return mapped


def append_test_bkg_to_graph(test_bkg_edge_index: torch.Tensor, test_bkg_edge_type: torch.Tensor, final_edge_index: torch.Tensor, final_edge_type: torch.Tensor, device: torch.device):
    """Append test BKG edges to the graph tensors."""
    if test_bkg_edge_index.size(1) > 0:
        final_edge_index = torch.cat([final_edge_index, test_bkg_edge_index], dim=1)
        final_edge_type = torch.cat([final_edge_type, test_bkg_edge_type], dim=0)
        logging.info(f"Appended {test_bkg_edge_index.shape[1]} test BKG edges. Total edges now: {final_edge_index.shape[1]}")
    else:
        logging.info("No valid test BKG edges to append after filtering.")
    return final_edge_index, final_edge_type


def prepare_test_edges(test_df: pd.DataFrame, train_relation_to_id: dict, node_to_id: dict):
    """Prepare test edges for prediction, ensuring only training relations are used."""
    test_pairs = []
    test_labels = []
    valid_indices = []  # To track which rows are valid
    skipped_test_edges = 0  # Counter for skipped test edges

    for idx, row in test_df.iterrows():
        d1, d2, y = row["Drug1_ID"], row["Drug2_ID"], row.get("Y", -1)
        if d1 in node_to_id and d2 in node_to_id:
            if y in train_relation_to_id:
                test_pairs.append([node_to_id[d1], node_to_id[d2]])
                test_labels.append(train_relation_to_id[y])
                valid_indices.append(idx)
            elif y == -1:
                # No label available; prepare for prediction
                test_pairs.append([node_to_id[d1], node_to_id[d2]])
                test_labels.append(-1)  # Placeholder
                valid_indices.append(idx)
            else:
                # Relation not in training relations; skip
                logging.warning(f"Test edge ({d1}, {d2}, {y}) has an unseen relation. Skipping.")
                skipped_test_edges += 1
        else:
            # Log and skip invalid edges
            logging.warning(f"Skipping invalid test edge: {d1}, {d2}, Y={y}")
            skipped_test_edges += 1

    if not test_pairs:
        logging.error("No valid test pairs found after filtering unseen relations. Exiting.")
        raise ValueError("No valid test pairs found.")

    return test_pairs, test_labels, valid_indices, skipped_test_edges


def perform_inference(model: RGCN, edge_classifier: EdgeClassifier, final_edge_index: torch.Tensor, final_edge_type: torch.Tensor, test_edge_index: torch.Tensor, device: torch.device):
    """Perform inference to predict relations on test edges."""
    with torch.no_grad():
        node_embeddings = model(final_edge_index, final_edge_type)
        logits = edge_classifier(node_embeddings, test_edge_index)
        preds = logits.argmax(dim=-1).cpu().numpy()
    return preds


# def save_predictions(test_df: pd.DataFrame, valid_indices: list, preds: np.ndarray, output_predictions: str, train_relation_to_id: dict):
#     """Save predictions to CSV."""
#     test_df.loc[valid_indices, "pred_relation_id"] = preds
#     test_df["pred_relation_id"].fillna(-1, inplace=True)  
#     inverse_train_relation_to_id = {v: k for k, v in train_relation_to_id.items()}
#     test_df["pred_relation"] = test_df["pred_relation_id"].apply(
#         lambda x: inverse_train_relation_to_id.get(x, "Unknown") if x != -1 else "Unknown"
#     )

#     test_df.to_csv(output_predictions, index=False)
#     logging.info(f"Saved predictions to {output_predictions}")


def evaluate_predictions(test_df: pd.DataFrame, train_relation_to_id: dict, eval_results_file: str, eval_log_file: str):
    """Evaluate predictions and save metrics: Accuracy, F1 Macro, and Cohen's Kappa."""
    if "Y" in test_df.columns and (test_df["Y"] != -1).any():
        # Filter test_df to include only valid predictions 
        eval_test_df = test_df[(test_df["Y"] != -1) & (test_df["pred_relation_id"] != -1)]
        if not eval_test_df.empty:
            eval_test_df["true_relation_id"] = eval_test_df["Y"].map(train_relation_to_id)
            # Remove rows where mapping failed 
            eval_test_df = eval_test_df.dropna(subset=["true_relation_id"])
            eval_test_df["true_relation_id"] = eval_test_df["true_relation_id"].astype(int)
            # Prepare arrays
            eval_true = eval_test_df["true_relation_id"].values
            eval_pred = eval_test_df["pred_relation_id"].values
            # Calculate metrics
            accuracy = accuracy_score(eval_true, eval_pred)
            f1_macro = f1_score(eval_true, eval_pred, average='macro')
            kappa = cohen_kappa_score(eval_true, eval_pred)

            # Save evaluation metrics to results file
            with open(eval_results_file, "w") as f:
                logging.info(f"Accuracy: {accuracy:.4f}\n")
                logging.info(f"F1 Macro: {f1_macro:.4f}\n")
                logging.info(f"Cohen's Kappa: {kappa:.4f}\n")
            logging.info(f"Saved evaluation metrics to {eval_results_file}")

            # Log detailed metrics to log file
            with open(eval_log_file, "w") as f:
                f.write(f"Accuracy: {accuracy:.4f}\n")
                f.write(f"F1 Macro: {f1_macro:.4f}\n")
                f.write(f"Cohen's Kappa: {kappa:.4f}\n")
            logging.info(f"Saved detailed evaluation logs to {eval_log_file}")
        else:
            logging.warning("No valid test edges with known labels and within training relations for evaluation.")
    else:
        logging.warning("Test data does not contain any known labels ('Y') for evaluation.")

def eval_rgcn_model(args):
    """
    Evaluate an RGCN model ensuring consistency in relation handling.
    Assumes that all relation types in the test BKG are present in the training BKG.
    """
    set_seed(args.seed)

    # Set up logging
    logging.basicConfig(
        filename=args.log_file,
        filemode='a',  # append
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load the trained checkpoint
    ckpt = load_checkpoint(args.model_path, device)
    logging.info(f"Loaded model checkpoint from {args.model_path}")

    # Load node_to_id mapping
    node_to_id = load_node_to_id(args.node_to_id_path)
    logging.info(f"Loaded node_to_id with {len(node_to_id)} nodes.")

    # Load relation_to_id and train_relation_to_id mapping from the checkpoint
    relation_to_id, num_relations, train_relation_to_id, num_train_relations = load_relations(ckpt)
    logging.info(f"Loaded relation_to_id with {num_relations} relations.")
    logging.info(f"Loaded train_relation_to_id with {num_train_relations} training relations.")

    # Load training data and BKG
    train_df, combined_train_bkg_df = load_train_data(args.train_file_path, args.hetionet_triplet_file)
    logging.info(f"Combined training and BKG data with {len(combined_train_bkg_df)} edges.")

    # Load test data
    test_df = load_test_data(args.test_file)
    logging.info(f"Loaded test data with {len(test_df)} records.")

    # Load drug descriptions
    drug_descriptions = get_drug_descriptions(pd.concat([train_df, test_df], ignore_index=True))
    logging.info(f"Loaded drug descriptions for {len(drug_descriptions)} drugs.")

    # Update node_to_id with test nodes
    train_nodes = set(combined_train_bkg_df["Drug1_ID"]).union(combined_train_bkg_df["Drug2_ID"])
    test_nodes = set(test_df["Drug1_ID"]).union(test_df["Drug2_ID"])
    test_bkg_nodes = set()
    if args.test_bkg_file and os.path.exists(args.test_bkg_file):
        test_bkg_triplets = load_hetionet_graph(args.test_bkg_file)
        test_bkg_df = pd.DataFrame(test_bkg_triplets, columns=["Drug1_ID", "Drug2_ID", "Y"])
        test_bkg_df["Y"] = test_bkg_df["Y"].astype(int)
        test_bkg_nodes = set(test_bkg_df["Drug1_ID"]).union(test_bkg_df["Drug2_ID"])
        logging.info(f"Loaded test BKG with {len(test_bkg_triplets)} triplets.")
    all_inference_nodes = train_nodes.union(test_nodes).union(test_bkg_nodes)
    node_to_id = update_node_to_id_with_missing_drugs(node_to_id, all_inference_nodes)
    num_nodes = len(node_to_id)
    logging.info(f"Updated node_to_id to include inference nodes. Total nodes: {num_nodes}")

    # Build embeddings (text-based or random)
    base_embeddings = prepare_embeddings(args, node_to_id, drug_descriptions, device)

    # Initialize RGCN model and EdgeClassifier
    model, edge_classifier = initialize_model(args, num_nodes, base_embeddings, num_relations, train_relation_to_id, device)

    # Load model and EdgeClassifier states with strict=False
    load_model_states(ckpt, model, edge_classifier, device, num_nodes, base_embeddings, train_relation_to_id)

    # Map training and BKG edges
    mapped_train_edges = map_edges_for_graph(combined_train_bkg_df, relation_to_id, node_to_id)
    final_edge_index, final_edge_type = create_edge_tensors(mapped_train_edges, device)
    logging.info(f"Final adjacency has {final_edge_index.shape[1]} edges.")

    # Append test BKG triplets if provided
    if args.test_bkg_file and os.path.exists(args.test_bkg_file):
        # Load and map test BKG edges
        test_bkg_edge_index, test_bkg_edge_type = load_test_bkg(args.test_bkg_file, train_relation_to_id, node_to_id, device)
        if test_bkg_edge_index is not None and test_bkg_edge_type is not None:
            # Append the test BKG edges
            final_edge_index, final_edge_type = append_test_bkg_to_graph(
                test_bkg_edge_index=test_bkg_edge_index,
                test_bkg_edge_type=test_bkg_edge_type,
                final_edge_index=final_edge_index,
                final_edge_type=final_edge_type,
                device=device
            )
    else:
        logging.info("No test BKG file provided or file does not exist.")

    # Prepare test edges for prediction
    test_pairs, test_labels, valid_indices, skipped_test_edges = prepare_test_edges(test_df, train_relation_to_id, node_to_id)
    logging.info(f"Prepared {len(test_pairs)} test edges for prediction. Skipped {skipped_test_edges} edges.")

    # Convert to tensors
    test_edge_index = torch.tensor(test_pairs, dtype=torch.long).t().contiguous().to(device)
    test_edge_type = torch.tensor(test_labels, dtype=torch.long).to(device)

    # Perform inference
    preds = perform_inference(model, edge_classifier, final_edge_index, final_edge_type, test_edge_index, device)

    # Save predictions
    # save_predictions(test_df, valid_indices, preds, args.output_predictions, train_relation_to_id)

    # Evaluate predictions
    evaluate_predictions(test_df, train_relation_to_id, args.eval_results_file, args.eval_log_file)

    logging.info("Evaluation completed.")


def main():
    args = parse_args()
    eval_rgcn_model(args)


if __name__ == "__main__":
    main()
