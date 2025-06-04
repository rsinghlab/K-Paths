# Function to load a dataset into a Pandas DataFram
import pandas as pd
import os
import json
import pandas as pd
import pandas as pd
import numpy as np



# Convert keys to integers if they are digits
def convert_keys_to_int(d):
    return {int(k) if k.isdigit() else k: v for k, v in d.items()}

def load_dataframe(dataset_name, split, paths, columns=None, debug=False):
    """
    Loads a dataset into a Pandas DataFrame, handling multiple file formats.

    Args:
        dataset_name (str): The dataset name (e.g., "drugbank").
        split (str): "train" or "test".
        paths (dict): Dictionary containing dataset paths.
        columns (list, optional): List of columns to keep in the DataFrame.
        debug (bool): Whether to load a small subset for debugging.

    Returns:
        pd.DataFrame: The processed DataFrame.
    """
    # Get the file path for the dataset
    file_path = paths[dataset_name].get(f"{split}")
    if not file_path:
        raise ValueError(f"File path not found for dataset '{dataset_name}' and split '{split}'")

    # Determine file type based on extension
    file_extension = os.path.splitext(file_path)[-1].lower()

    if file_extension == ".json":
        try:
            df = pd.read_json(file_path, lines=True)  # Try JSON Lines
        except ValueError:
            # If JSON Lines format fails, try loading as a standard JSON array
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):  # Ensure it's an array of objects
                    df = pd.DataFrame(data)
                else:
                    raise ValueError("JSON file must be in JSON Lines format or an array of objects.")

    elif file_extension == ".csv":
        df = pd.read_csv(file_path)
    elif file_extension == ".parquet":
        df = pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

    # Keep only selected columns
    if columns:
        df = df[columns]

    # Load only a small subset for debugging
    if debug:
        df = df.head(5)

    return df


def extract_triplets_from_paths(df):
    """
    Extracts triplets from nested paths in the 'all_paths' column and expands the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame with 'all_paths' containing nested triplets.

    Returns:
        pd.DataFrame: A new DataFrame where each row represents an extracted triplet.
    """
    def extract_triplets(paths):
        triplets = []
        for path in paths:
            if isinstance(path[0], list):  # Check if it's nested further
                triplets.extend(extract_triplets(path))  # Recursively flatten
            else:
                triplets.append(path)  # It's already a triplet
        return triplets

    # Process the DataFrame
    rows = []
    for _, row in df.iterrows():
        paths = row["all_paths"]
        triplets = extract_triplets(paths)  # Extract all triplets, even if nested
        for triplet in triplets:
            rows.append({
                "drug1_db": row["drug1_db"],
                "drug2_db": row["drug2_db"],
                "label_idx": row["label_idx"],
                "source": triplet[0],
                "relation": triplet[1],
                "target": triplet[2],
            })

    return pd.DataFrame(rows)  # Return the expanded DataFrame



def modify_relation_and_swap(row, offset):
    """
    Function to modify the relation and swap source and target based on the condition
    if relation id is greater than offset, swap the source and taget because it is a reversed relation

    """
    if row["relation"] >= offset:
        # Subtract relation from offset to get new relation id
        new_relation = row["relation"] - offset
        
        # Swap source and target if relation > offset
        swapped_source = row["target"]
        swapped_target = row["source"]
    else:
        # Keep the relation as is
        new_relation = row["relation"]
        
        # No swapping
        swapped_source = row["source"]
        swapped_target = row["target"]
    
    return swapped_source, swapped_target, new_relation,



def expand_relations(df, upper_bound, description_to_id):
    """
    Function to modify the relation and swap source and target based on the condition
    if relation id is greater than offset, swap the source and taget because it is a reversed relation
    This function will be used to modify the relation and swap source and target based on the condition.

    Expands rows where 'modified_relation' exceeds an upper bound by splitting 'relation_description'.

    Args:
        df (pd.DataFrame): Input DataFrame with 'relation_description' and 'modified_relation'.
        upper_bound (int): Threshold for splitting relations.
        description_to_id (dict): Mapping of descriptions to relation IDs.

    Returns:
        pd.DataFrame: Expanded DataFrame with split relations.
    """

    def split_relations(row):
        if row['modified_relation'] > upper_bound:
            # Split the relation_description by " and "
            descriptions = row['relation_description'].split(' and ')
            # Create new rows for each split description
            new_rows = []
            for desc in descriptions:
                new_row = row.copy()
                new_row['relation_description'] = desc.strip()
                # Map the new description back to its ID using the reverse dictionary
                new_row['modified_relation'] = description_to_id.get(desc.strip(), row['modified_relation'])
                # Store the original modified_relation in a new column
                new_row['original_modified_relation'] = row['modified_relation']
                new_rows.append(new_row)
            return new_rows
        else:
            # If not expanded, set original_modified_relation to NaN
            row['original_modified_relation'] = np.nan
            return [row]

    # Apply the function to each row and expand the DataFrame
    expanded_rows = []
    for _, row in df.iterrows():
        expanded_rows.extend(split_relations(row))

    return pd.DataFrame(expanded_rows)  # Return the expanded DataFrame

 
def correct_relations(df, upper_bound, corrected_relation_dict):
    """
     Function to correct the relation descriptions and update the modified_relation IDs
    This function will be used to correct the relation descriptions and update the modified_relation IDs.

    Corrects relation descriptions and updates modified_relation IDs for rows exceeding upper_bound.

    Args:
        df (pd.DataFrame): The input DataFrame.
        upper_bound (int): Threshold to identify incorrect rows.
        corrected_relation_dict (dict): Mapping of corrected descriptions to relation IDs.

    Returns:
        pd.DataFrame: Updated DataFrame with corrected descriptions and IDs.
    """

    # Step 1: Identify rows where modified_relation exceeds upper_bound
    incorrect_rows = df[df["modified_relation"] > upper_bound].copy()
    # print(incorrect_rows)
    # Step 2: Iterate over incorrect rows and apply corrections
    for index, row in incorrect_rows.iterrows():
        # Correct the spelling
        corrected_description = row['relation_description'].replace('Diease', 'Disease')
        
        # Update the relation_description in the DataFrame
        df.at[index, 'relation_description'] = corrected_description
        
        # Update the modified_relation ID if the corrected description is in the dictionary
        if corrected_description in corrected_relation_dict:
            df.at[index, 'modified_relation'] = corrected_relation_dict[corrected_description]

    return df  # Return the updated DataFrame

# Function to create the final subgraph by filtering and sorting relations
# This function will be used to create the final subgraph by filtering and sorting relations

def create_final_subgraph(expanded_df, dataset, data):
    """
    Creates the final subgraph by filtering and sorting relations.
    It will remove relations that are not in the Hetionet and sort the relations by modified_relation.
    So we are prunning the entire hetionet to keep only useful relations for the dataset.

    Args:
        expanded_df (pd.DataFrame): The processed DataFrame with relations.
        dataset (str): The dataset key used to extract relation IDs.
        data (dict): A dictionary containing dataset-related metadata.

    Returns:
        pd.DataFrame: The final filtered and sorted subgraph.
    """
    # Step 1: Identify relation IDs to remove
    ids_to_remove = set(data[dataset].get(f"{dataset}_relation_id_to_name", {}).keys())

    # Step 2: Filter out rows with non-Hetionet IDs
    filtered_df = expanded_df[~expanded_df["modified_relation"].isin(ids_to_remove)].reset_index(drop=True)

    # Step 3: Select relevant columns
    filtered_df = filtered_df[["swapped_source", "swapped_target", "modified_relation"]]

    # Step 4: Identify and drop duplicate relations
    filtered_df = filtered_df.drop_duplicates(subset=["swapped_source", "swapped_target", "modified_relation"]).reset_index(drop=True)
    

    # Step 5: Sort the DataFrame by modified_relation
    sorted_df = filtered_df.sort_values(by="modified_relation").reset_index(drop=True)

    return sorted_df


