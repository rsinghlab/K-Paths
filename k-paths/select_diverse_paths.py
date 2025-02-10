import json
import logging
import os
import random
import re
from copy import deepcopy

import networkx as nx
import numpy as np
import pandas as pd
from datasets import Dataset

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Define relation ID to name mapping (this should be replaced with actual data)
common_relation_id_to_name = {
    0: "{u} (Gene) interacts with {v} (Gene)",
    1: "{u} (Compound) resembles {v} (Compound)",
    2: "{u} (Disease) downregulates {v} (Gene)",
    3: "{u} (Disease) presents {v} (Symptom)",
    4: "{u} (Disease) localizes {v} (Anatomy)",
    5: "{u} (Compound) treats {v} (Disease)",
    6: "{u} (Compound) binds {v} (Gene)",
    7: "{u} (Compound) upregulates {v} (Gene)",
    8: "{u} (Disease) resembles {v} (Disease)",
    # 9: "{u} (Gene) associates with {v} (Disease)",
    9: "{u} (Diease) associated with {v} (Gene)",
    10: "{u} (Compound) palliates {v} (Disease)",
    11: "{u} (Anatomy) downregulates {v} (Gene)",
    12: "{u} (Anatomy) upregulates {v} (Gene)",
    13: "{u} (Gene) covaries with {v} (Gene)",
    14: "{u} (Gene) participates in {v} (Molecular Function)",
    # 15: "{u} (Compound) belongs to {v} (Pharmacologic Class)",
    15: "{u} (Pharmacologic Class) includes {v} (Compound)",
    16: "{u} (Gene) participates in {v} (Cellular Component)",
    17: "{u} (Gene) regulates {v} (Gene)",
    18: "{u} (Compound) downregulates {v} (Gene)",
    19: "{u} (Disease) upregulates {v} (Gene)",
    20: "{u} (Gene) participates in {v} (Pathway)",
    21: "{u} (Compound) causes {v} (Side Effect)",
    22: "{u} (Anatomy) expresses {v} (Gene)",
}

# Used chatgpt o1 for this mapping
# converts the non-symmetric relations into passive voice
reversed_common_relation_id_to_name = {
    0: "{u} (Gene) interacts with {v} (Gene)",  # symmetric
    1: "{u} (Compound) resembles {v} (Compound)",  # symmetric
    2: "{u} (Gene) is downregulated by {v} (Disease)",
    3: "{u} (Symptom) is presented by {v} (Disease)",
    4: "{u} (Anatomy) is localized by {v} (Disease)",
    5: "{u} (Disease) is treated by {v} (Compound)",
    6: "{u} (Gene) is bound by {v} (Compound)",
    7: "{u} (Gene) is upregulated by {v} (Compound)",
    8: "{u} (Disease) resembles {v} (Disease)",  # symmetric
    9: "{u} (Disease) associated with {v} (Gene)",  # symmetric
    10: "{u} (Disease) is palliated by {v} (Compound)",
    11: "{u} (Gene) is downregulated by {v} (Anatomy)",
    12: "{u} (Gene) is upregulated by {v} (Anatomy)",
    13: "{u} (Gene) covaries with {v} (Gene)",  # symmetric
    14: "{u} (Molecular Function) is participated in by {v} (Gene)",
    15: "{u} (Compound) is included by {v} (Pharmacologic Class)",
    16: "{u} (Cellular Component) is participated in by {v} (Gene)",
    17: "{u} (Gene) is regulated by {v} (Gene)",
    18: "{u} (Gene) is downregulated by {v} (Compound)",
    19: "{u} (Gene) is upregulated by {v} (Disease)",
    20: "{u} (Pathway) is participated in by {v} (Gene)",
    21: "{u} (Side Effect) is caused by {v} (Compound)",
    22: "{u} (Gene) is expressed by {v} (Anatomy)",
}

ddinter_relation_id_to_name = {
    23: "{u} may cause a minor interaction that can limit clinical effects when taken with {v}",
    24: "{u} may cause a moderate interaction that could exacerbate diseases when taken with {v}",
    25: "{u} may lead to a major life-threatening interaction when taken with {v}",
}

reverse_ddinter_relation_id_to_name = {
    k: v for k, v in ddinter_relation_id_to_name.items()
}

drugbank_relation_id_to_name = {
    23: "{u} may increase the photosensitizing activities of {v}",
    24: "{u} may increase the anticholinergic activities of {v}",
    25: "The bioavailability of {v} can be decreased when combined with {u}",
    26: "The metabolism of {v} can be increased when combined with {u}",
    27: "{u} may decrease the vasoconstricting activities of {v}",
    28: "{u} may increase the anticoagulant activities of {v}",
    29: "{u} may increase the ototoxic activities of {v}",
    30: "The therapeutic efficacy of {v} can be increased when used in combination with {u}",
    31: "{u} may increase the hypoglycemic activities of {v}",
    32: "{u} may increase the antihypertensive activities of {v}",
    33: "The serum concentration of the active metabolites of {v} can be reduced when {v} is used in combination with {u} resulting in a loss in efficacy",
    34: "{u} may decrease the anticoagulant activities of {v}",
    35: "The absorption of {v} can be decreased when combined with {u}",
    36: "{u} may decrease the bronchodilatory activities of {v}",
    37: "{u} may increase the cardiotoxic activities of {v}",
    38: "{u} may increase the central nervous system depressant (CNS depressant) activities of {v}",
    39: "{u} may decrease the neuromuscular blocking activities of {v}",
    40: "{u} can cause an increase in the absorption of {v} resulting in an increased serum concentration and potentially a worsening of adverse effects",
    41: "{u} may increase the vasoconstricting activities of {v}",
    42: "{u} may increase the QTc-prolonging activities of {v}",
    43: "{u} may increase the neuromuscular blocking activities of {v}",
    44: "{u} may increase the adverse neuromuscular activities of {v}",
    45: "{u} may increase the stimulatory activities of {v}",
    46: "{u} may increase the hypocalcemic activities of {v}",
    47: "{u} may increase the atrioventricular blocking (AV block) activities of {v}",
    48: "{u} may decrease the antiplatelet activities of {v}",
    49: "{u} may increase the neuroexcitatory activities of {v}",
    50: "{u} may increase the dermatologic adverse activities of {v}",
    51: "{u} may decrease the diuretic activities of {v}",
    52: "{u} may increase the orthostatic hypotensive activities of {v}",
    53: "The risk or severity of hypertension can be increased when {v} is combined with {u}",
    54: "{u} may increase the sedative activities of {v}",
    55: "The risk or severity of QTc prolongation can be increased when {u} is combined with {v}",
    56: "{u} may increase the immunosuppressive activities of {v}",
    57: "{u} may increase the neurotoxic activities of {v}",
    58: "{u} may increase the antipsychotic activities of {v}",
    59: "{u} may decrease the antihypertensive activities of {v}",
    60: "{u} may increase the vasodilatory activities of {v}",
    61: "{u} may increase the constipating activities of {v}",
    62: "{u} may increase the respiratory depressant activities of {v}",
    63: "{u} may increase the hypotensive and central nervous system depressant (CNS depressant) activities of {v}",
    64: "The risk or severity of hyperkalemia can be increased when {u} is combined with {v}",
    65: "The protein binding of {v} can be decreased when combined with {u}",
    66: "{u} may increase the central neurotoxic activities of {v}",
    67: "{u} may decrease effectiveness of {v} as a diagnostic agent",
    68: "{u} may increase the bronchoconstrictory activities of {v}",
    69: "The metabolism of {v} can be decreased when combined with {u}",
    70: "{u} may increase the myopathic rhabdomyolysis activities of {v}",
    71: "The risk or severity of adverse effects can be increased when {u} is combined with {v}",
    72: "The risk or severity of heart failure can be increased when {v} is combined with {u}",
    73: "{u} may increase the hypercalcemic activities of {v}",
    74: "{u} may decrease the analgesic activities of {v}",
    75: "{u} may increase the antiplatelet activities of {v}",
    76: "{u} may increase the bradycardic activities of {v}",
    77: "{u} may increase the hyponatremic activities of {v}",
    78: "The risk or severity of hypotension can be increased when {u} is combined with {v}",
    79: "{u} may increase the nephrotoxic activities of {v}",
    80: "{u} may decrease the cardiotoxic activities of {v}",
    81: "{u} may increase the ulcerogenic activities of {v}",
    82: "{u} may increase the hypotensive activities of {v}",
    83: "{u} may decrease the stimulatory activities of {v}",
    84: "The bioavailability of {v} can be increased when combined with {u}",
    85: "{u} may increase the myelosuppressive activities of {v}",
    86: "{u} may increase the serotonergic activities of {v}",
    87: "{u} may increase the excretion rate of {v} which could result in a lower serum level and potentially a reduction in efficacy",
    88: "The risk or severity of bleeding can be increased when {u} is combined with {v}",
    89: "{u} can cause a decrease in the absorption of {v} resulting in a reduced serum concentration and potentially a decrease in efficacy",
    90: "{u} may increase the hyperkalemic activities of {v}",
    91: "{u} may increase the analgesic activities of {v}",
    92: "The therapeutic efficacy of {v} can be decreased when used in combination with {u}",
    93: "{u} may increase the hypertensive activities of {v}",
    94: "{u} may decrease the excretion rate of {v} which could result in a higher serum level",
    95: "The serum concentration of {v} can be increased when it is combined with {u}",
    96: "{u} may increase the fluid retaining activities of {v}",
    97: "The serum concentration of {v} can be decreased when it is combined with {u}",
    98: "{u} may decrease the sedative activities of {v}",
    99: "The serum concentration of the active metabolites of {v} can be increased when {v} is used in combination with {u}",
    100: "{u} may increase the hyperglycemic activities of {v}",
    101: "{u} may increase the central nervous system depressant (CNS depressant) and hypertensive activities of {v}",
    102: "{u} may increase the hepatotoxic activities of {v}",
    103: "{u} may increase the thrombogenic activities of {v}",
    104: "{u} may increase the arrhythmogenic activities of {v}",
    105: "{u} may increase the hypokalemic activities of {v}",
    106: "{u} may increase the vasopressor activities of {v}",
    107: "{u} may increase the tachycardic activities of {v}",
    108: "The risk of a hypersensitivity reaction to {v} is increased when it is combined with {u}",
}

reverse_drugbank_relation_id_to_name = {
    k: v for k, v in drugbank_relation_id_to_name.items()
}

DATASET_REL_TO_NAME = {
    "drugbank": drugbank_relation_id_to_name,
    "ddinter": ddinter_relation_id_to_name,
}

REV_DATASET_REL_TO_NAME = {
    "drugbank": reverse_drugbank_relation_id_to_name,
    "ddinter": reverse_ddinter_relation_id_to_name,
}

PATHS = {
    "drugbank": {
        "inductive": "data/drugbank/All_KG_DB.txt",
        "transductive": "",
        "bkg_entity2id": "data/drugbank/BKG_entity2Id.json",
        "drug_info": "data/drugbank/DDI_dict_action_roberta.json",
        "node2id": "data/drugbank/node2id.json",
        "train_set": "data/drugbank/train_set_drugbank_all_shuffled_desc2.csv",
        "test_set": "data/drugbank/test_set_drugbank.json",
    },
    "ddinter": {
        "inductive": "data/ddinter/All_KG_DDinter.txt",
        "transductive": "",
        "bkg_entity2id": "data/ddinter/BKG_entity2Id.json",
        "drug_info": "data/ddinter/all_Ddinter_drugsX2.csv",
        "node2id": "data/ddinter/node2id.json",
        "train_set": "data/ddinter/DDinter_train.csv",
        "test_set": "data/ddinter/DDinter_test.json",
    },
}

HETIO_NET_PATH = (
    "data/external_KG/hetionet-v1.0.json"
)

def load_and_process_mappings(dataset_name):
    """
    Load and process mappings to create a dictionary that maps node IDs to their names.

    Args:
        dataset_name (str): Name of the dataset (e.g., "drugbank").

    Returns:
        dict: A dictionary mapping node IDs to names.
    """

    def preprocess_B_to_dict(B):
        B_dict = {}
        for item in B:
            kind = item["kind"]
            identifier = str(item["identifier"])
            B_dict[f"{kind}::{identifier}"] = item["name"]
        return B_dict

    def create_new_dict(A, B_dict):
        new_dict = {}
        for key, value in A.items():
            name = B_dict.get(value)  # O(1) lookup
            if name:
                new_dict[key] = name
        return new_dict

    # Load and preprocess entity2id JSON
    with open(PATHS[dataset_name]["bkg_entity2id"], "r") as file:
        data = json.load(file)
    KG_entities = {value: key for key, value in data.items()}

    # Load and preprocess Hetionet JSON
    with open(HETIO_NET_PATH, "r") as file:
        data = json.load(file)
    B_dict = preprocess_B_to_dict(data["nodes"])

    # Map entities to names
    entities_to_name_dict = create_new_dict(KG_entities, B_dict)

    # Load and preprocess DrugBank node-to-ID mappings
    with open(PATHS[dataset_name]["node2id"], "r") as file:
        drug_data = json.load(file)
    drug_bank_ids = {value: key for key, value in drug_data.items()}

    # Load and preprocess drug information

    if dataset_name == "drugbank":
        drug_info = pd.read_json(PATHS[dataset_name]["drug_info"]).T
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
    else:
        drug_info = pd.read_csv(PATHS[dataset_name]["drug_info"])
        drug_dict = pd.Series(
            drug_info.name.values, drug_info.drugbank_id.values
        ).to_dict()

    # Create mapping from DrugBank IDs to names
    drugs_to_name_dicts = {
        key: drug_dict.get(value)
        for key, value in drug_bank_ids.items()
        if value in drug_dict
    }

    # Merge dictionaries
    node_id_to_name = {**drugs_to_name_dicts, **entities_to_name_dict}

    return node_id_to_name


# Function to build a graph from a file
def build_graph_from_file(file_path, dataset, add_revese_edges=False, offset=0):

    # note: the label_idx is offsetted by the length of the common relations
    dataset_tuples = dataset.map(
        lambda x: {
            "tuple": (
                x["drug1_id"],
                (x["label_idx"] + len(common_relation_id_to_name)),
                x["drug2_id"],
            )
        },
        remove_columns=dataset.column_names,
    )
    dataset_tuples_set = set([tuple(x) for x in dataset_tuples["tuple"]])

    # existing tuples
    existing_tuples = []

    G = nx.DiGraph()
    with open(file_path, "r") as f:
        header = f.readline()
        if not header.strip().startswith("entitya"):
            f.seek(0)
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            entitya_id = int(parts[0])
            entityb_id = int(parts[1])
            relation_id = int(parts[2])
            if (entitya_id, relation_id, entityb_id) in dataset_tuples_set:
                existing_tuples.append((entitya_id, relation_id, entityb_id))
                continue

            G.add_node(entitya_id)
            G.add_node(entityb_id)
            G.add_edge(entitya_id, entityb_id, relation=relation_id)
            if add_revese_edges:
                reverse_relation_id = relation_id + offset
                G.add_edge(entityb_id, entitya_id, relation=reverse_relation_id)

    logger.info(f"Graph built with {len(G.nodes)} nodes and {len(G.edges)} edges")
    logger.info(f"Existing tuples in the graph: {len(existing_tuples)}")

    return G


def two_nodes_in_graph(G, drug_a, drug_b, max_length=3, max_paths=10):
    # no path between drug_a and drug_b return empty list
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


def one_node_in_graph(G, drug_id, max_paths=10):
    neighbors = list(G.neighbors(drug_id))

    # Collect information about neighbors and the relationships
    all_paths = []
    for neighbor in neighbors:
        edge_data = G.get_edge_data(drug_id, neighbor)
        relation_id = edge_data.get("relation", None)
        all_paths.append([(drug_id, relation_id, neighbor)])
        if len(all_paths) >= max_paths:
            break

    return all_paths


def find_paths(G, drug_a, drug_b, max_length=3, max_paths=10):
    # both nodes are not in the graph
    if not G.has_node(drug_a) and not G.has_node(drug_b):
        return []
    elif G.has_node(drug_a) and not G.has_node(drug_b):
        return one_node_in_graph(G, drug_a, max_paths)
    elif not G.has_node(drug_a) and G.has_node(drug_b):
        return one_node_in_graph(G, drug_b, max_paths)
    else:
        return two_nodes_in_graph(G, drug_a, drug_b, max_length, max_paths)


def load_and_process_dataset(dataset_name, split, debug=False):
    if split == "test":
        file_path = PATHS[dataset_name]["test_set"]
        dataset = Dataset.from_json(file_path)
    elif split == "train":
        file_path = PATHS[dataset_name]["train_set"]
        dataset = Dataset.from_csv(file_path)

    if debug:
        random.seed(0)
        sample_indices = random.sample(range(len(dataset)), 1000)
        dataset = dataset.select(sample_indices)

    # process the drugbank dataset
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
            "drug1_id": "drug1_db",
            "drug2_id": "drug2_db",
            "Y": "label_idx",
            "effect": "label",
        }
        columns_pairs = train_column_pairs if split == "train" else test_columns_pairs
        for k, v in columns_pairs.items():
            dataset = dataset.rename_column(k, v)

        with open(PATHS[dataset_name]["node2id"], "r") as file:
            drug_data = json.load(file)

        dataset = dataset.map(
            lambda example: {
                "drug1_id": drug_data.get(example["drug1_db"], -1),
                "drug2_id": drug_data.get(example["drug2_db"], -1),
            }
        )

        return dataset

    def process_ddinter_dataset(dataset):
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

        with open(PATHS[dataset_name]["node2id"], "r") as file:
            drug_data = json.load(file)

        dataset = dataset.map(
            lambda example: {
                "drug1_id": drug_data.get(example["drug1_db"], -1),
                "drug2_id": drug_data.get(example["drug2_db"], -1),
            }
        )

        return dataset

    if dataset_name == "drugbank":
        dataset = process_drugbank_dataset(dataset)
    elif dataset_name == "ddinter":
        dataset = process_ddinter_dataset(dataset)
    else:
        raise ValueError(f"Dataset {dataset_name} not recognized")

    # keep only the following columns
    columns = [
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
    dataset = dataset.remove_columns(
        [col for col in dataset.column_names if col not in columns]
    )
    # ordering for better readability
    dataset = dataset.select_columns(columns)

    logger.info("Dataset loaded and processed")
    logger.info(f"Columns: {dataset.column_names}")

    return dataset


def create_output_file_name(args):
    file_name = f"{args.dataset_name}_{args.split}"
    if args.add_reverse_edges:
        file_name += f"_add_reverse"

    if args.debug:
        file_name += "_debug"

    file_name += ".json"
    return file_name


def build_relations_dict(dataset_name, add_reverse_edges=False):

    dataset_rel_to_name = DATASET_REL_TO_NAME[dataset_name]

    relation_id_to_name = {**common_relation_id_to_name, **dataset_rel_to_name}
    offset = len(relation_id_to_name)
    if add_reverse_edges:
        rev_common = {k: v for k, v in reversed_common_relation_id_to_name.items()}
        rev_dataset_rel_to_name = REV_DATASET_REL_TO_NAME[dataset_name]
        rev_drugbank = {k: v for k, v in rev_dataset_rel_to_name.items()}
        combined = {**rev_common, **rev_drugbank}
        rev_with_offset = {k + offset: v for k, v in combined.items()}
        relation_id_to_name = {**relation_id_to_name, **rev_with_offset}
    return relation_id_to_name, offset


def main(args):
    logger.info("Processing nodes")
    node_id_to_name = load_and_process_mappings(args.dataset_name)

    # combine drugbank and common relation to get relation_id_to_name
    logger.info("Processing relations")
    relation_id_to_name, offset = build_relations_dict(
        args.dataset_name, args.add_reverse_edges
    )

    # Load the dataset
    logger.info(f"Loading the dataset; Split = {args.split}")
    dataset = load_and_process_dataset(args.dataset_name, args.split, args.debug)

    logger.info(f"Loading graph; Reverse Edges = {args.add_reverse_edges}")
    G = build_graph_from_file(
        PATHS[f"{args.dataset_name}"]["inductive"],
        dataset,
        args.add_reverse_edges,
        offset,
    )

    def wrapped_get_formatted_paths(example):
        all_paths = find_paths(G, example["drug1_id"], example["drug2_id"])
        all_paths_str_list = []
        for path in all_paths:
            tuple_str = []
            for u, r, v in path:
                u_name = node_id_to_name.get(u, f"Node-{u}")
                v_name = node_id_to_name.get(v, f"Node-{v}")
                relation_name = relation_id_to_name.get(r, "related to")
                tuple_str.append((u_name, relation_name, v_name))
            if tuple_str:
                all_paths_str_list.append(tuple_str)

        path_str = ""
        for _path in all_paths_str_list:
            for i, (u, r, v) in enumerate(_path):
                if i == 0:
                    path_str += r.format(u=u, v=v)
                else:
                    path_str += f" and {r.format(u=u, v=v)}"
            path_str += "\n"

        return {
            "all_paths": all_paths,
            "all_paths_str": all_paths_str_list,
            "path_str": path_str,
        }

    logger.info("Finding paths for all the entries in the dataset")
    dataset = dataset.map(
        wrapped_get_formatted_paths, num_proc=1, desc="Processing dataset rows"
    )

    file_name = create_output_file_name(args)
    os.makedirs(args.output_dir, exist_ok=True)
    file_path = os.path.join(args.output_dir, file_name)

    dataset.to_json(file_path)

    logger.info(f"Dataset saved at {file_path}")
    logger.info("Done!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="drugbank")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--add_reverse_edges", action="store_true")
    parser.add_argument("--output_dir", type=str, default="files/dataset_with_paths/")

    # debug option
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)