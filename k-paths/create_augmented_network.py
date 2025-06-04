import os
import json
from pathlib import Path
from utils.augmented_network_utils import create_augmentedKG, load_dataset_file

# ---------- Setup ----------
# BASE_DIR = Path(__file__).resolve().parent
# os.chdir(BASE_DIR)  # Ensure correct working directory

# ---------- Paths ----------
PATHS = {

    "hetionet": {
        "data_relations": "data/hetionet/hetionet_relations.json",
        "reversed_data_relations": "data/hetionet/hetionet_reversed_relations.json"
    },
       
    "drugbank": {
        "node2id": "data/drugbank/node2id.json",
        "bkg_file":"data/drugbank/BKG_file.txt",
        "train_file":"data/drugbank/Drugbank_train.txt",
        "data_relations": "data/drugbank/drugbank_relations.json"
        
    },
    "ddinter": {
        "node2id": "data/ddinter/node2id.json",
        "bkg_file":"data/ddinter/BKG_file.txt",
        "train_file":"data/ddinter/DDinter_train.txt",
        "data_relations": "data/ddinter/ddinter_relations.json"
    },
    "pharmaDB": { 
        "node2id": "data/pharmaDB/node2id.json",
        "bkg_file":"data/pharmaDB/BKG_file.txt",
        "train_file":"data/pharmaDB/pharmaDB_train.txt",
        "data_relations": "data/pharmaDB/pharmaDB_relations.json"
        }
        

        }

# ---------- Main ----------
def main():
    data = {}
    datasets = ["drugbank", "ddinter", "pharmaDB"]
    # datasets = ["pharmaDB"]
    handle_duplicates = True

    # Load Hetionet relations
    hetionet_rel = load_dataset_file("hetionet", "data_relations", PATHS)
    hetionet_reversed = load_dataset_file("hetionet", "reversed_data_relations", PATHS)

    for dataset in datasets:
        print(f"\n▶ Processing {dataset}...")

        # Build augmented knowledge graph
        augmented_kg, relation_map = create_augmentedKG(
            dataset, PATHS,
            hetionet_rel,
            hetionet_reversed,
            handle_duplicate_edges=handle_duplicates
        )

        print(f"Augmented KG for {dataset}: {len(augmented_kg)} edges")
        print(f"Relation types: {len(augmented_kg.relation.value_counts())}")

        # Save KG
        # Save KG
        augmented_kg.to_csv(f"data/{dataset}/{dataset}_Augmented_KG.txt", sep=" ", index=False, header=False)


        # Save mapping
        data[dataset] = relation_map

    # Save all mappings
    with open("data/relations_dicts_file.json", "w") as f:
        json.dump(data, f, indent=2)

    print("\n All augmented KGs and relation mappings saved.")

# ---------- Entry Point ----------
if __name__ == "__main__":
    main()

#run: python k-paths/src/create_augmented_network.py