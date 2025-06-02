import json
import pandas as pd

def load_dataset_file(dataset, file_type, paths_dict, column_names=None, delimiter=r'\s+', convert_ids=True):
    """
    Loads a dataset file from JSON or TXT, and optionally converts ID-like keys or columns to integers.

    Parameters:
        dataset (str): Dataset name.
        file_type (str): File type key (e.g., 'node2id', 'bkg_file').
        paths_dict (dict): Paths to datasets.
        column_names (list, optional): Column names for TXT files.
        delimiter (str, optional): Delimiter for TXT files (default: whitespace).
        convert_ids (bool): Convert keys/columns with ID-like content to integers (default: True).

    Returns:
        dict or DataFrame
    """
    file_path = paths_dict.get(dataset, {}).get(file_type)
    if not file_path:
        raise ValueError(f"No path found for dataset '{dataset}' and file type '{file_type}'.")

    # Handle JSON files
    if file_path.endswith(".json"):
        with open(file_path, 'r') as file:
            data = json.load(file)

        # Convert dict keys to int if applicable
        if convert_ids and isinstance(data, dict):
            try:
                data = {int(k): v for k, v in data.items()}
            except ValueError:
                pass  # Only convert keys that are valid integers

        return data

    # Handle TXT files
    elif file_path.endswith(".txt"):
        df = pd.read_csv(
            file_path,
            sep=delimiter,
            names=column_names,
            engine="python"
        )

        if convert_ids and column_names:
            for col in column_names:
                if 'id' in col.lower() or 'node' in col.lower() or 'drug' in col.lower():
                    try:
                        df[col] = df[col].astype(int)
                    except ValueError:
                        pass

        return df

    else:
        raise ValueError(f"Unsupported file extension in: {file_path}")


import pandas as pd

def filter_pharmadb_kg(kg_df):
    """
    For pharmaDB: use either hetionet edges or PharmaDB edges for common relations to avoid duplicates of common relations (treats, palliates,23,24 <-> 5,10) 

    Applies PharmaDB-specific filtering rules to the knowledge graph:
    - Removes (node1, node2, relation=5) if the same (node1, node2) exists with relation=23
    - Removes (node1, node2, relation=10) if the same (node1, node2) exists with relation=24
    Parameters:
        kg_df (pd.DataFrame): The full knowledge graph DataFrame (with columns 'node1', 'node2', 'relation')

    Returns:
        pd.DataFrame: Filtered KG DataFrame
    """
    # Get (node1, node2) pairs for relations 23 and 24
    relation_23_pairs = kg_df[kg_df['relation'] == 23][['node1', 'node2']]
    relation_24_pairs = kg_df[kg_df['relation'] == 24][['node1', 'node2']]

    # Remove (node1, node2, relation=5) if same pair exists in relation 23
    kg_df = kg_df.merge(relation_23_pairs.assign(relation=5),
                        on=['node1', 'node2', 'relation'],
                        how='left', indicator=True)
    kg_df = kg_df[kg_df['_merge'] == 'left_only'].drop(columns=['_merge'])

    # Remove (node1, node2, relation=10) if same pair exists in relation 24
    kg_df = kg_df.merge(relation_24_pairs.assign(relation=10),
                        on=['node1', 'node2', 'relation'],
                        how='left', indicator=True)
    kg_df = kg_df[kg_df['_merge'] == 'left_only'].drop(columns=['_merge'])

    return kg_df

import pandas as pd

def merge_duplicate_relations(kg_df):
    """
    Merges multiple relations between the same (node1, node2) pair into a single composite relation.

    Parameters:
        kg_df (pd.DataFrame): The knowledge graph DataFrame with columns 'node1', 'node2', and 'relation'.

    Returns:
        pd.DataFrame: A deduplicated DataFrame where each (node1, node2) pair appears only once,
                      with combined relation strings like '5_10' if multiple existed.
    """
    # Group by (node1, node2) and collect relations
    grouped = kg_df.groupby(["node1", "node2"])["relation"].apply(list).reset_index()

    # Build new relation rows for duplicates
    new_rows = []
    for _, row in grouped.iterrows():
        if len(row["relation"]) > 1:
            sorted_relations = "_".join(map(str, sorted(row["relation"])))
            new_rows.append({
                "node1": row["node1"],
                "node2": row["node2"],
                "relation": sorted_relations
            })

    # DataFrame of new composite relations
    new_df = pd.DataFrame(new_rows)

    # Append to original KG
    combined_df = pd.concat([kg_df, new_df], ignore_index=True)

    # Flag synthetic relations
    combined_df["is_new_relation"] = combined_df["relation"].apply(lambda x: "_" in str(x))

    # Sort to prioritize composite relations
    combined_df = combined_df.sort_values(by="is_new_relation", ascending=True).drop(columns=["is_new_relation"])

    # Drop duplicates, keeping the composite relation (last one)
    deduplicated_df = combined_df.drop_duplicates(subset=["node1", "node2"], keep="last").reset_index(drop=True)

    return deduplicated_df

def generate_relation_mapping(df, relation_id_to_name):
   
    """This function generates a mapping of relation IDs to their names,
    including both existing relations and new composite relations formed by merging multiple relations.
    It handles both existing relations and new relations formed by combining multiple IDs."""

    unique_relations = df["relation"].astype(str).unique()

    # Get all old relations from relation_id_to_name
    old_relations = set(map(str, relation_id_to_name.keys()))  # Convert keys to strings

    # Separate old and new relations
    existing_relations = [rel for rel in unique_relations if rel in old_relations]
    new_relations = [rel for rel in unique_relations if "_" in rel and rel not in old_relations]

    # Map existing relations to their names
    old_relation_mapping = {int(rel): relation_id_to_name[int(rel)] for rel in existing_relations}

    # Determine the starting ID for new relations
    next_new_id = max(relation_id_to_name.keys()) + 1  # Start from last old ID + 1

    # Create mapping for new relations with incremental integer IDs
    new_relation_mapping = {}
    for rel in new_relations:
        components = map(int, rel.split("_"))  # Split and convert components to integers
        names = [relation_id_to_name[comp] for comp in components]  # Map components to names
        concatenated_name = " and ".join(names)  # Concatenate the names

        new_relation_mapping[next_new_id] = concatenated_name  # Assign new ID
        next_new_id += 1  # Increment for the next relation

    # Merge new relations into relation_id_to_name to ensure all keys are included
    final_mapping = {**relation_id_to_name, **new_relation_mapping}

    return final_mapping


def generate_sorted_df(df, relation_id_to_name):
    # Extract unique relations from the dataframe
    unique_relations = df["relation"].astype(str).unique()

    # Get all old relations from relation_id_to_name
    old_relations = set(map(str, relation_id_to_name.keys()))  # Convert keys to strings

    # Separate old and new relations
    existing_relations = [rel for rel in unique_relations if rel in old_relations]
    new_relations = [rel for rel in unique_relations if "_" in rel and rel not in old_relations]

    # Map existing relations to their names
    old_relation_mapping = {int(rel): relation_id_to_name[int(rel)] for rel in existing_relations}

    # Determine the starting ID for new relations
    next_new_id = max(relation_id_to_name.keys()) + 1  # Start from last old ID + 1

    # Create mapping for new relations with incremental integer IDs
    new_relation_mapping = {}
    for rel in new_relations:
        components = rel.split("_")  # Split into components (string format)
        
        try:
            # Convert components to integers and map them to names if they exist
            names = [relation_id_to_name[int(comp)] for comp in components if int(comp) in relation_id_to_name]
            concatenated_name = " and ".join(names)  # Concatenate names
            
            new_relation_mapping[rel] = next_new_id  # Use the original string key like the first version
            next_new_id += 1  # Increment for the next relation
            
        except ValueError:
            print(f"Warning: Skipping relation '{rel}' due to invalid components.")
            continue  # Skip invalid relations that can't be converted

    # Merge old and new relations into the final mapping
    final_mapping = {**relation_id_to_name, **new_relation_mapping}

    # Reverse mapping: relation -> relation ID
    relation_name_to_id = {k: v for v, k in final_mapping.items()}  # Flip the dictionary

    # Ensure `final_mapping` contains the correct relation IDs for `X_Y` relations
    relation_key_to_id = {**new_relation_mapping, **{str(k): k for k in relation_id_to_name.keys()}}  # Map both old and new


    # Function to map `df["relation"]` values to their corresponding integer IDs
    def map_relation_to_id(relation):
        relation = str(relation)  # Ensure it's a string
        return relation_key_to_id.get(relation, relation)  # Replace if found, else keep original

    # Apply the mapping to `df["relation"]`
    df["relation"] = df["relation"].astype(str).map(map_relation_to_id)
    df["relation"] = df["relation"].astype(int)  # Convert to integer type
    # Sort the DataFrame by the `relation` column
    df_sorted = df.sort_values(by="relation", ascending=True).reset_index(drop=True)
    return df_sorted



def create_augmentedKG(dataset, paths_dict, hetionet_relations, hetionet_reversed, handle_duplicate_edges=True):
    """
    Process a single dataset: load files, build augmented KG, handle duplicates, and build relation mappings.

    Returns:
        tuple: (All_KG_df, relation_mapping_dict)
    """
    import pandas as pd

    # Load data
    node_data = load_dataset_file(dataset, "node2id", paths_dict)
    train_data = load_dataset_file(dataset, "train_file", paths_dict, column_names=["Drug1_ID", "relation", "Drug2_ID"])
    KG = load_dataset_file(dataset, "bkg_file", paths_dict, column_names=["node1", "node2", "relation"])
    data_relations = load_dataset_file(dataset, "data_relations", paths_dict)
    data_relations = {k + len(hetionet_relations): v for k, v in data_relations.items()}

    # Map drug names to IDs
    default_value = '-1'
    train_data['node1'] = train_data['Drug1_ID'].map(node_data).fillna(default_value).astype(int)
    train_data['node2'] = train_data['Drug2_ID'].map(node_data).fillna(default_value).astype(int)

    # Offset relation IDs
    train_KG = train_data[['node1', 'node2', 'relation']].copy()
    train_KG['relation'] += len(hetionet_relations)

    # Filter valid rows
    train_KG = train_KG[(train_KG['node1'] != -1) & (train_KG['node2'] != -1)].reset_index(drop=True)

    # Augment KG
    All_KG = pd.concat([KG, train_KG], ignore_index=True).drop_duplicates(subset=['node1', 'node2', 'relation']).reset_index(drop=True)

    # Handle duplicates if requested
    if handle_duplicate_edges:
        if dataset.lower() == "pharmadb":
            print("Applying PharmaDB-specific filtering...")
            All_KG = filter_pharmadb_kg(All_KG)

        All_KG = merge_duplicate_relations(All_KG)
        combined_relations = {**hetionet_relations, **data_relations}
        combined_relations_reversed = {**hetionet_reversed, **data_relations}

        relation_mapping = {
            "common_relation_id_to_name": generate_relation_mapping(All_KG, combined_relations),
            "reversed_common_relation_id_to_name": generate_relation_mapping(All_KG, combined_relations_reversed),
            f"{dataset}_relation_id_to_name": data_relations
        }
    else:
        combined_relations = {**hetionet_relations, **data_relations}
        combined_relations_reversed = {**hetionet_reversed, **data_relations}

        relation_mapping = {
            "common_relation_id_to_name": combined_relations,
            "reversed_common_relation_id_to_name": combined_relations_reversed,
            f"{dataset}_relation_id_to_name": data_relations
        }


    All_KG = generate_sorted_df(All_KG, combined_relations)


    return All_KG, relation_mapping
