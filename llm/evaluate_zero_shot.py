# this code parses the outputs
import argparse
import json
import os
import re
import unicodedata


import pandas as pd
import numpy as np
import logging
from sklearn.metrics import f1_score, cohen_kappa_score
from datasets import Dataset

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


TEST_SET_PATH = {
    "drugbank_transductive": "/dataset_with_paths/drugbank_test_add_reverse2.json",
    "drugbank_inductive": "/dataset_with_paths/drugbank_test_add_reverse3.json",
    "ddinter": "/dataset_with_paths/ddinter_test_add_reverse.json", 
    "pharmacotherapyDB": "/dataset_with_paths/pharmacotherapyDB_test_add_reverse.json",
}

DRUGBANK_LABELS = {
    0: "{u} may increase the photosensitizing activities of {v}",
    1: "{u} may increase the anticholinergic activities of {v}",
    2: "{u} can decrease the bioavailability of {v}",
    3: "{u} can increase the metabolism of {v}",
    4: "{u} may decrease the vasoconstricting activities of {v}",
    5: "{u} may increase the anticoagulant activities of {v}",
    6: "{u} may increase the ototoxic activities of {v}",
    7: "{u} can increase the therapeutic efficacy of {v}",
    8: "{u} may increase the hypoglycemic activities of {v}",
    9: "{u} may increase the antihypertensive activities of {v}",
    10: "{u} may reduce the serum concentration of the active metabolites of {v}",
    11: "{u} may decrease the anticoagulant activities of {v}",
    12: "{u} may decrease the absorption of {v}",
    13: "{u} may decrease the bronchodilatory activities of {v}",
    14: "{u} may increase the cardiotoxic activities of {v}",
    15: "{u} may increase the central nervous system depressant activities of {v}",
    16: "{u} may decrease the neuromuscular blocking activities of {v}",
    17: "{u} can increase the absorption and serum concentration of {v}",
    18: "{u} may increase the vasoconstricting activities of {v}",
    19: "{u} may increase the QTc prolonging activities of {v}",
    20: "{u} may increase the neuromuscular blocking activities of {v}",
    21: "{u} may increase the adverse neuromuscular activities of {v}",
    22: "{u} may increase the stimulatory activities of {v}",
    23: "{u} may increase the hypocalcemic activities of {v}",
    24: "{u} may increase the atrioventricular blocking activities of {v}",
    25: "{u} may decrease the antiplatelet activities of {v}",
    26: "{u} may increase the neuroexcitatory activities of {v}",
    27: "{u} may increase the dermatologic adverse activities of {v}",
    28: "{u} may decrease the diuretic activities of {v}",
    29: "{u} may increase the orthostatic hypotensive activities of {v}",
    30: "{u} may increase the hypertensive effects of {v}",
    31: "{u} may increase the sedative activities of {v}",
    32: "{u} may increase the severity of QTc prolonging effects when combined with {v}",
    33: "{u} may increase the immunosuppressive activities of {v}",
    34: "{u} may increase the neurotoxic activities of {v}",
    35: "{u} may increase the antipsychotic activities of {v}",
    36: "{u} may decrease the antihypertensive activities of {v}",
    37: "{u} may increase the vasodilatory activities of {v}",
    38: "{u} may increase the constipating activities of {v}",
    39: "{u} may increase the respiratory depressant activities of {v}",
    40: "{u} may increase the hypotensive and central nervous system depressant activities of {v}",
    41: "{u} may increase the severity of hyperkalemic effects when combined with {v}",
    42: "{u} may decrease the protein binding of {v}",
    43: "{u} may increase the central neurotoxic activities of {v}",
    44: "{u} may decrease the diagnostic effectiveness of {v} ",
    45: "{u} may increase the bronchoconstrictory activities of {v}",
    46: "{u} may decrease the metabolism of {v}",
    47: "{u} may increase the myopathic rhabdomyolysis activities of {v}",
    48: "{u} may increase the severity of adverse effects when combined with {v}",
    49: "{u} may increase heart failure risks when combined with {v}",
    50: "{u} may increase the hypercalcemic activities of {v}",
    51: "{u} may decrease the analgesic activities of {v}",
    52: "{u} may increase the antiplatelet activities of {v}",
    53: "{u} may increase the bradycardic activities of {v}",
    54: "{u} may increase the hyponatremic activities of {v}",
    55: "{u} may increase the hypotensive effects when combined with {v}",
    56: "{u} may increase the nephrotoxic activities of {v}",
    57: "{u} may decrease the cardiotoxic activities of {v}",
    58: "{u} may increase the ulcerogenic activities of {v}",
    59: "{u} may increase the hypotensive activities of {v}",
    60: "{u} may decrease the stimulatory activities of {v}",
    61: "{u} may increase the bioavailability of {v}",
    62: "{u} may increase the myelosuppressive activities of {v}",
    63: "{u} may increase the serotonergic activities of {v}",
    64: "{u} may increase the excretion rate {v}",
    65: "{u} may increase bleeding risks when combined with {v}",
    66: "{u} may decrease the absorption and serum concentration of {v}",
    67: "{u} may increase the hyperkalemic activities of {v}",
    68: "{u} may increase the analgesic activities of {v}",
    69: "{u} may decrease the therapeutic efficacy of {v}",
    70: "{u} may increase the hypertensive activities of {v}",
    71: "{u} may decrease the excretion rate  {v}",
    72: "{u} may increase the serum concentration of {v}",
    73: "{u} may increase the fluid retaining activities of {v}",
    74: "{u} may decrease the serum concentration of {v}",
    75: "{u} may decrease the sedative activities of {v}",
    76: "{u} may increase the serum concentration of the active metabolites of {v}",
    77: "{u} may increase the hyperglycemic activities of {v}",
    78: "{u} may increase the central nervous system depressant and hypertensive activities of {v}",
    79: "{u} may increase the hepatotoxic activities of {v}",
    80: "{u} may increase the thrombogenic activities of {v}",
    81: "{u} may increase the arrhythmogenic activities of {v}",
    82: "{u} may increase the hypokalemic activities of {v}",
    83: "{u} may increase the vasopressor activities of {v}",
    84: "{u} may increase the tachycardic activities of {v}",
    85: "{u} may increase the risk of hypersensitivity reaction to {v}",
}

CUSTOM_STOPWORDS = {
    "used",
    "a",
    "be",
    "is",
    "it",
    "may",
    "can",
    "the",
    "of",
    "and" "by",
    "when",
    "combined",
    "with",
    "risk",
    "to",
    "in",
    "drug1",
    "drug2",
    "risks",
}


DDINTER_LABELS = {
    "minor": 1,
    "moderate": 1,
    "major": 2,
}

ERROR_IDX = {
    "drugbank_transductive": 48,  # '{u} may increase the severity of adverse effects when combined with {v}',
    "drugbank_inductive": 48,  # '{u} may increase the severity of adverse effects when combined with {v}',
    "ddinter": 0,  # 'major'
    "pharmacotherapyDB": 0,  # "Disease-modifying": "The drug therapeutically changes the underlying or downstream biology of the disease"
}


def extract_answer(text):
    # Define the regex pattern to match '##Answer:<text>'
    pattern = r"##Answer:\s*(.+)"
    # Search for the pattern in the input text
    match = re.search(pattern, text)

    # If a match is found, return the string after '##Answer:'
    if match:
        return match.group(1).strip()
    else:
        return -1


def custom_reduce_derived_forms(word):
    """
    Reduce derived forms like 'effectiveness' to their root words.
    """
    reduction_map = {
        "effectiveness": "effect",
        "increased": "increase",
        "increases": "increase",
        "decreased": "decrease",
        "decreases": "decrease",
        "additive": "increase",
        "effects": "effect",
    }
    return reduction_map.get(word, word)  # Return reduced word if found


# Preprocess Input Text
def preprocess_text(text):
    """
    Convert input text into a preprocessed form by removing stopwords but keeping necessary terms.
    """
    if isinstance(text, str):
        tokens = re.findall(r"\b\w+\b", text.lower())  # Tokenize and lowercase
        cleaned_tokens = [
            custom_reduce_derived_forms(word)
            for word in tokens
            if word not in CUSTOM_STOPWORDS
        ]
        # cleaned_tokens = [custom_reduce_derived_forms(word) for word in tokens if word not in stopwords_set]
        cleaned_text = " ".join(cleaned_tokens)
        return cleaned_text
    return ""  # Return empty string if input is not a valid text


def find_closest_match_without_drug_names2(
    model_prediction: str, drug1: str, drug2: str
):
    """
    Finds the closest match by:
    - First checking exact matches with drug names.
    - Then checking exact matches without drug names.
    # - Then checking closest matches with drug names. (commented out)
    # - Finally, checking closest matches without drug names. (commented out)
    - Ensuring case-insensitive and stopword-filtered comparisons.
    """
    # Normalize and clean model prediction
    model_prediction_cleaned = (
        unicodedata.normalize("NFKC", model_prediction).strip().lower()
    )
    model_prediction_cleaned = re.sub(
        r"[^\w\s]", "", model_prediction_cleaned
    )  # Remove punctuation

    # Ensure drug1 and drug2 are lowercase
    drug1 = drug1.strip().lower() if drug1 else "drug_u"
    drug2 = drug2.strip().lower() if drug2 else "drug_v"

    # Preprocess model prediction (WITH drug names)
    model_prediction_processed = preprocess_text(model_prediction_cleaned)

    # Generate cleaned possible statements dynamically (WITH drug names)
    possible_statements = {
        key: unicodedata.normalize(
            "NFKC", value.replace("{u}", drug1).replace("{v}", drug2)
        )
        .strip()
        .lower()
        for key, value in DRUGBANK_LABELS.items()
    }
    possible_statements_processed = {
        key: preprocess_text(value) for key, value in possible_statements.items()
    }

    # **Step 1: Try exact matching WITH drug names**
    for key, value in possible_statements_processed.items():
        if value == model_prediction_processed:
            return key, "Exact Match (With Drug Names)"

    # **Step 2: Remove drug names and try exact matching**
    model_prediction_no_drugs = (
        model_prediction_cleaned.replace(drug1, "").replace(drug2, "").strip()
    )
    model_prediction_no_drugs = preprocess_text(model_prediction_no_drugs)

    possible_statements_no_drugs = {
        key: preprocess_text(value.replace(drug1, "").replace(drug2, "").strip())
        for key, value in possible_statements.items()
    }

    for key, value in possible_statements_no_drugs.items():
        if value == model_prediction_no_drugs:
            return key, "Exact Match (Without Drug Names)"

    return 48, "Error"  # return most common class


def parse_drugbank_outputs(predictions: list, drug_1_names: list, drug_2_names: list):
    error_label = ERROR_IDX[args.dataset]

    prediction_idx = []
    all_errors = []
    for i, pred in enumerate(predictions):
        pred_str = extract_answer(pred)
        if pred_str == -1:
            prediction_idx.append(error_label)
            all_errors.append(pred)
            continue
        drug1 = drug_1_names[i]
        drug2 = drug_2_names[i]
        matched_relation_id, match_type = find_closest_match_without_drug_names2(
            pred_str, drug1, drug2
        )
        prediction_idx.append(matched_relation_id)
        if matched_relation_id == error_label:
            all_errors.append(pred)

    # report the number of errors
    logger.info(
        f"Number of errors: {len(all_errors)}; Percentage of errors: {len(all_errors) / len(predictions)}"
    )

    return prediction_idx, all_errors


def load_test_dataset(path):
    return Dataset.from_json(path)


def load_dataset_predictions(prediction_path, dataset):
    if prediction_path.endswith(".csv"):
        logger.info("loading prediction csv file")
        return pd.read_csv(prediction_path)

    if os.path.isdir(prediction_path) and os.path.exists(
        os.path.join(prediction_path, "predictions.csv")
    ):
        logger.info("loading prediction csv file")
        return pd.read_csv(os.path.join(prediction_path, "predictions.csv"))

    step_size = 1000
    predictions = []
    for i in range(0, len(dataset), step_size):
        start = i
        end = min(i + step_size, len(dataset))
        logger.info(f"Loading predictions from {start} to {end}")
        _path = os.path.join(prediction_path, f"predictions_{start}_{end}.csv")
        if not os.path.exists(_path):
            logger.warning(f"File not found: {_path}")
            continue
        pred_df = pd.read_csv(_path, usecols=["prediction"], dtype={"prediction": str})
        # add to the predictions list
        predictions.append(pred_df)

    predictions_df = pd.concat(predictions)
    predictions_df = predictions_df.fillna("")
    return predictions_df


def parse_ddinter_outputs(predictions: list):
    error_label = ERROR_IDX[args.dataset]

    prediction_idx = []
    all_errors = []
    for pred in predictions:
        regex_pattern = "|".join(re.escape(v.strip()) for v in DDINTER_LABELS.keys())

        try:
            pred_str = re.findall(regex_pattern, pred.lower())
        except Exception as e:
            print(e)
            pred_str = ""
        if len(pred_str) == 0:
            prediction_idx.append(error_label)
            all_errors.append(pred)
        else:
            prediction_idx.append(DDINTER_LABELS[pred_str[0]])

    # report the number of errors
    logger.info(
        f"Number of errors: {len(all_errors)}; Percentage of errors: {len(all_errors) / len(predictions)}"
    )

    return prediction_idx, all_errors


def parse_pharmacotherapyDB_outputs(predictions: list):
    dd_dictsX = {
        "disease-modifying": 0,
        "dreating": 0,
        "treat": 0,
        "treatment": 0,
        "treating": 0,  # Adding variation for "Treating"
        "palliates": 1,
        "palliative": 1,  # Adding variation for "Palliates"
        "palliate": 1,
        "palliating": 1,  # Adding lowercase variation
        "palliation": 1,  # Adding another variation
        "non indications": 2,
        "not indicated": 2,
        "non indication": 2,  # Adding variation for "Non indications"
    }
    error_label = ERROR_IDX[args.dataset]
    all_errors = []
    prediction_idx = []
    for pred in predictions:
        pred_str = extract_answer(pred)
        if pred_str == -1:
            prediction_idx.append(error_label)
            all_errors.append(pred)
            continue
        match = re.search(
            r"\b(Non indications|Palliates|Palliative|palliate|palliating|palliation|Disease-modifying|treating|treat|Treatment|Not indicated|Non indication)\b[.,!?]?",
            pred_str,
            re.IGNORECASE,
        )
        if match:
            prediction_idx.append(dd_dictsX[match.group(1).lower()])
        else:
            prediction_idx.append(error_label)
            all_errors.append(pred)

    return prediction_idx, all_errors


def compute_metrics(prediction_path, dataset, use_options, option_style):
    dataset = load_test_dataset(TEST_SET_PATH[args.dataset])

    labels = dataset["label_idx"]
    predictions_df = load_dataset_predictions(prediction_path, dataset)
    predictions = predictions_df["prediction"].tolist()

    if "drugbank" in args.dataset:
        logger.info("Parsing DrugBank outputs")
        drug_1_names = dataset["drug1_name"]
        drug_2_names = dataset["drug2_name"]

        predictions, all_errors = parse_drugbank_outputs(
            predictions, drug_1_names, drug_2_names
        )
    elif args.dataset == "ddinter":
        logger.info("Parsing DDInter outputs")
        predictions, all_errors = parse_ddinter_outputs(predictions)
    elif args.dataset == "pharmacotherapyDB":
        logger.info("Parsing pharmacotherapyDB outputs")
        predictions, all_errors = parse_pharmacotherapyDB_outputs(predictions)
    else:
        NotImplementedError("Dataset not implemented")

    # compute accuracy and f1
    all_preds = np.array(predictions)
    all_labels = np.array(labels)

    # hot fix
    if len(all_preds) != len(all_labels):
        logger.warning("Length mismatch between predictions and labels")
        all_labels = all_labels[: len(all_preds)]
        logger.info(f"Number of unique labels: {len(set(all_labels))}")
        logger.info(f"Number of unique predictions: {len(set(all_preds))}")

    accuracy = np.mean(all_preds == all_labels)
    logger.info(f"Accuracy: {accuracy * 100:.2f}")

    f1 = f1_score(all_labels, all_preds, average=None).mean()
    logger.info(f"F1 Score: {f1 * 100:.2f}")

    kappa = cohen_kappa_score(all_labels, all_preds)
    logger.info(f"Cohen's Kappa: {kappa * 100:.2f}")

    # save the results
    outputs = {
        "accuracy": accuracy,
        "f1": f1,
        "kappa": kappa,
        "errors": all_errors,
        "error_rate": len(all_errors) / len(all_preds),
    }

    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse the outputs of the evaluation")
    parser.add_argument(
        "--prediction_path", type=str, help="Path to the predictions file or directory"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["drugbank_transductive", "drugbank_inductive", "ddinter", "pharmacotherapyDB"],
        help="Dataset to evaluate on",
    )

    # experiment parameters
    parser.add_argument("--use_options", action="store_true")
    parser.add_argument(
        "--option_style", type=str, default="numbered", choices=["numbered", "bulleted"]
    )
    parser.add_argument("--variant", type=str, default="v1")
    parser.add_argument("--output_dir", type=str, default="files/results/")

    args = parser.parse_args()
    outputs = compute_metrics(
        args.prediction_path, args.dataset, args.use_options, args.option_style
    )

    file_name = "regex_metrics.json"
    # check if the prediction_path is a directory
    if os.path.isdir(args.prediction_path):
        output_path = os.path.join(args.prediction_path, f"{file_name}")
    else:
        base_name = os.path.basename(args.prediction_path)
        name, ext = os.path.splitext(base_name)
        output_path = os.path.join(
            os.path.dirname(args.prediction_path), f"{name}_{file_name}"
        )

    with open(output_path, "w") as f:
        json.dump(outputs, f)