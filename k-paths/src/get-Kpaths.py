import os
import logging
import argparse
from functools import partial
from datasets import Dataset
from utils.Kpaths_utils import (
    load_and_process_mappings,
    load_and_process_dataset,
    build_graph_from_file,
    find_Kpaths,
    remove_leakage,
    load_json,
    build_relations_dict,  # Now imported from utils
    find_neighbors_only,
    find_Kpaths_no_filter
)

# Configure Logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Define Paths
PATHS = {
    "drugbank": {
        "inductive": "data/drugbank/drugbank_Augmented_KG.txt",
        "transductive": "",
        "bkg_entity2id": "data/drugbank/BKG_entity2Id.json",
         "drug_info": "data/drugbank/id_to_name_mapping.json",
        "node2id": "data/drugbank/node2id.json",
        "train_set": "data/drugbank/drugbank_train_set.csv",
        "test_set": "data/drugbank/drugbank_test_set.json",
        # "test_set": "data/drugbank/drugbank_val_set.json"
    },
    "ddinter": {
        "inductive": "data/ddinter/ddinter_Augmented_KG.txt",
        "bkg_entity2id": "data/ddinter/BKG_entity2Id.json",
        "drug_info": "data/ddinter/id_to_name_mapping.csv",
        "node2id": "data/ddinter/node2id.json",
        "train_set": "data/ddinter/ddinter_train_set.csv",
        "test_set": "data/ddinter/ddinter_test_set.json",
        # "test_set":"data/ddinter_val_set.json",
    },
    
    "pharmaDB": {
        "inductive": "data/pharmaDB/pharmaDB_Augmented_KG.txt",
        "bkg_entity2id": "data/pharmaDB/BKG_entity2Id.json",
        "drug_info": "data/pharmaDB/id_to_name_mapping.json",
        "node2id": "data/pharmaDB/node2id.json",
        "train_set": "data/pharmaDB/pharmaDB_train_set.csv",
        "test_set": "data/pharmaDB/pharmaDB_test_set.json",
    }
    
}

HETIO_NET_PATH = "data/hetionet/hetionet-v1.0.json"
RELATIONS_PATH = "data/relations_dicts_file.json"

def create_output_file_name(args):
    """Creates an output filename based on arguments."""
    file_name = f"{args.dataset_name}_{args.split}"
    if args.add_reverse_edges:
        file_name += "_add_reverse"
    if args.debug:
        file_name += "_debug"
    return file_name + ".json"

def main(args):
    logger.info(f"Processing dataset: {args.dataset_name}")

    # Load mappings and dataset
    node_id_to_name = load_and_process_mappings(args.dataset_name, PATHS, HETIO_NET_PATH)
    relation_id_to_name, offset, dataset_relations = build_relations_dict(args.dataset_name, RELATIONS_PATH, args.add_reverse_edges)
    dataset = load_and_process_dataset(args.dataset_name, args.split, PATHS, args.debug)

 

    # Build the graph once
    G = build_graph_from_file(PATHS[args.dataset_name]["inductive"], dataset, args.add_reverse_edges, offset)

    
    def normalize_keys(d):
        return {int(k) if str(k).isdigit() else k: v for k, v in d.items()}

    node_id_to_name_norm = normalize_keys(node_id_to_name)
    relation_id_to_name_norm = normalize_keys(relation_id_to_name)
    dataset_relations_norm = normalize_keys(dataset_relations)

    dataset_dict = {idx: value for idx, value in enumerate(dataset_relations_norm.values())}
    

    def wrapped_get_formatted_paths(example, mode="default"):
        """Generates formatted paths for interactions across different datasets."""
        
        """Select path-finding strategy (default=k-paths, no_filter=k-paths without diversity, or neighbors_only)"""
        if mode == "K-paths":
            logger.info("Finding K-paths")
            all_paths = find_Kpaths(G, example["drug1_id"], example["drug2_id"])
        elif mode == "no_filter":
            logger.info("Finding K-paths without filtering")
            all_paths = find_Kpaths_no_filter(G, example["drug1_id"], example["drug2_id"])
        elif mode == "neighbors_only":
            logger.info("Finding direct neighbors between entities only")
            all_paths = find_neighbors_only(G, example["drug1_id"], example["drug2_id"])
        else:
            raise ValueError(f"Unsupported path extraction mode: {mode}")
        
        all_paths_str_list = []

        for path in all_paths:
            tuple_str = []
            for u, r, v in path:
                u_name = node_id_to_name_norm.get(int(u) if str(u).isdigit() else u, f"Node-{u}")
                v_name = node_id_to_name_norm.get(int(v) if str(v).isdigit() else v, f"Node-{v}")
                relation_name = relation_id_to_name_norm.get(int(r) if str(r).isdigit() else r, "related to")
                tuple_str.append((u_name, relation_name, v_name))

            if tuple_str:
                all_paths_str_list.append(tuple_str)

        # Convert to readable string format
        path_str = ""
        for _path in all_paths_str_list:
            for i, (u, r, v) in enumerate(_path):
                if i == 0:
                    path_str += r.format(u=u, v=v)
                else:
                    path_str += f" and {r.format(u=u, v=v)}"
            path_str += "\n"

        logger.info("Removing Leakages")

        filtered_path_str = remove_leakage(
            {
                "path_str": path_str,
                "label": example["label_idx"],
                "drug1name": example.get("drug_name", example.get("drug1_name", "Unknown")),
                "drug2name": example.get("disease_name", example.get("drug2_name", "Unknown")),
            },
            interaction_dict=dataset_dict,
            label_col="label",
            drug1_col="drug1name",
            drug2_col="drug2name",
        )

        return {
            "all_paths": all_paths,
            "all_paths_str": all_paths_str_list,
            "path_str": filtered_path_str,
        }
        

    # Apply function efficiently
    logger.info("Finding paths for all the entries in the dataset")

    # dataset = dataset.map(wrapped_get_formatted_paths, num_proc=12)

    # Wrap function with selected mode
    wrapped_fn = partial(wrapped_get_formatted_paths, mode=args.mode)
    dataset = dataset.map(wrapped_fn, num_proc=12)

    # Save processed dataset
    output_file = os.path.join(args.output_dir, create_output_file_name(args))
    os.makedirs(args.output_dir, exist_ok=True)
    dataset.to_json(output_file)

    logger.info(f"Dataset saved at {output_file}")
    logger.info("Processing completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="drugbank",choices=["ddinter", "drugbank", "pharmaDB"])
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--add_reverse_edges", action="store_true")
    parser.add_argument("--mode", type=str, default="K-paths", choices=["K-paths", "no_filter", "neighbors_only"])
    parser.add_argument("--output_dir", type=str, default="data/paths")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
# run:# python3 k-paths/src/get-Kpaths.py --add_reverse_edges --split test --dataset ddinter --add_reverse_edges --mode K-paths