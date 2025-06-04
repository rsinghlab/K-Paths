import json
import pandas as pd
from pathlib import Path
from utils.subgraph_utils import load_dataframe, extract_triplets_from_paths, modify_relation_and_swap, expand_relations, correct_relations, create_final_subgraph, convert_keys_to_int

# ---------- Define Global Constants ----------
PATHS = {
    "drugbank": { 
        "train": "data/paths/drugbank_train_add_reverse.json",
        "test": "data/paths/drugbank_test_add_reverse.json",
    },
    "ddinter": {
        "train": "data/paths/ddinter_train_add_reverse.json",
        "test": "data/paths/ddinter_test_add_reverse.json",
    },
    
    "pharmaDB": {
        "train": "data/paths/pharmaDB_train_add_reverse.json",
        "test": "data/paths/pharmaDB_test_add_reverse.json",

    }
    
}

REL_DICT_PATH = "data/relations_dicts_file.json"
OUTPUT_DIR = "data/subgraphs"

# ---------- Utility Functions ----------

def load_relation_dict(dataset):
    with open(REL_DICT_PATH, "r") as f:
        data = json.load(f, object_hook=convert_keys_to_int)
    relation_id_to_name = {
        **data[dataset]['common_relation_id_to_name'],
        **data[dataset].get(f"{dataset}_relation_id_to_name", {})
    }
    return data, relation_id_to_name

def get_columns_to_keep(dataset):
    if dataset == 'pharmaDB':
        return [
            'drug1_db', 'drug2_db', 'drug1_id', 'drug2_id',
            'drug_name', 'disease_name', 'label',
            'label_idx', 'all_paths'
        ]
    else:
        return [
            "drug1_db", "drug2_db", "drug1_id", "drug2_id",
            "drug1_name", "drug2_name", "label",
            "label_idx", "all_paths"
        ]

def process_dataset_split(dataset, split):
    print(f"\nProcessing {dataset} - {split}")
    
    data, relation_id_to_name = load_relation_dict(dataset)
    offset = len(relation_id_to_name)
    upperbound = 23 + len(data[dataset].get(f"{dataset}_relation_id_to_name", {})) - 1

    columns_to_keep = get_columns_to_keep(dataset)
    df = load_dataframe(dataset, split, PATHS, columns=columns_to_keep)
    df_triplets = extract_triplets_from_paths(df).drop_duplicates(subset=["source", "relation", "target"]).reset_index(drop=True)

    df_triplets["swapped_source"], df_triplets["swapped_target"], df_triplets["modified_relation"] = zip(
        *df_triplets.apply(modify_relation_and_swap, axis=1, offset=offset)
    )

    df_triplets['relation_description'] = df_triplets['modified_relation'].map(relation_id_to_name)
    description_to_id = {v: k for k, v in relation_id_to_name.items()}

    expanded_df = expand_relations(df_triplets, upper_bound=upperbound, description_to_id=description_to_id).reset_index(drop=True)

    corrected_relation_dict = {
        '{u} (Disease) associated with {v} (Gene)': 9,
        '{u} may increase the anticholinergic activities of {v}': 24
    }
    expanded_df = correct_relations(expanded_df, upper_bound=upperbound, corrected_relation_dict=corrected_relation_dict)

    final_subgraph = create_final_subgraph(expanded_df, dataset, data)

    output_path = Path(OUTPUT_DIR) / f"{dataset}_{split}_subgraph.txt"
    final_subgraph.to_csv(output_path, sep=' ', index=False, header=False)
    print(f"Saved to {output_path}")

# ---------- Main Execution Loop ----------

def main():
    # datasets = ["drugbank", "ddinter", "pharmaDB"]
    # datasets = ["drugbank"]




    for dataset in PATHS.keys():
    # for dataset in datasets:
        for split in ["train", "test"]:
            process_dataset_split(dataset, split)

if __name__ == "__main__":
    main()
#run: python k-paths/src/get-subgraph.py