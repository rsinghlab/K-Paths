import argparse
import json
import os
import re
import unicodedata
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, cohen_kappa_score
from datasets import Dataset


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Dataset paths
TEST_SET_PATH = {
    "drugbank": "data/paths/drugbank_test_add_reverse.json",
    "ddinter": "data/dataset_with_paths/data/ddinter_test.json",
    "pharmaDB": "data/pharmaDB_test.json",
}

# Labels and stopwords
DRUGBANK_LABELS = { #original labels
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
    "minor": 0,
    "moderate": 1,
    "major": 2,
}

ERROR_IDX = {
    "drugbank_v1": 48,  # '{u} may increase the severity of adverse effects when combined with {v}',
    "drugbank_v2": 48,  # '{u} may increase the severity of adverse effects when combined with {v}',
    "drugbank": 48,  # '{u} may increase the severity of adverse effects when combined with {v}',
    "ddinter": 0,  # 'minor'
    "pharmaDB": 0,  # "Disease-modifying": "The drug therapeutically changes the underlying or downstream biology of the disease"
}

def extract_answer(text):
    # print("default style")
    pattern = r"##Answer:\s*(.+)"
    match = re.search(pattern, text)
    return match.group(1).strip() if match else -1


def extract_answer_tx_gemma(text):
    pattern = r"^\s*##\s*Answer:\s*(.*)$"
    match = re.search(pattern, text, flags=re.MULTILINE)
    return match.group(1).strip() if match else text

def extract_answer_finetuned(text):
    pattern = r"##Answer:\s*(?:assistant\s*)?(?:\n+Answer:\s*(?!assistant))?([\s\S]*?)(?=\n##Answer:|\Z)"
    matches = re.findall(pattern, text, flags=re.DOTALL)
    if not matches:
        return -1
    answer = matches[-1].strip()
    return re.sub(r'^Answer:\s*|^##Answer:\s*', '', answer)

def extract_answer_auto(text):
    if not isinstance(text, str):
        return -1
    if "## Answer:" in text or "##Answer:" in text:
        if "assistant" in text or re.search(r'\n+Answer:', text):
            return extract_answer_finetuned(text)
        elif re.search(r'^\s*##\s*Answer:', text, flags=re.MULTILINE):
            return extract_answer_tx_gemma(text)
        else:
            return extract_answer(text)
    return extract_answer(text)



def extract_last_answer(text):
    if not isinstance(text, str):
        return None

    # Split the text into lines
    lines = text.splitlines()

    # First pass: search for a line that starts with "##Answer:"
    for line in reversed(lines):
        if line.lstrip().startswith("##Answer:") or line.lstrip().startswith("## Answer:"):
            return line.strip()
    
    # Fallback: search for "Answer:<end_of_turn>" and return the first sentence after it
    fallback_pattern = r"Answer:<end_of_turn>(.*)"
    match = re.search(fallback_pattern, text, re.IGNORECASE | re.DOTALL)
    if match:
        after_answer = match.group(1).strip()
        # Extract the first sentence (naively split on ".")
        sentences = re.split(r"[.?!]", after_answer)
        for sentence in sentences:
            if sentence.strip():
                return sentence.strip()

    # If no suitable answer found
    return text


EXTRACTOR_MAP = {
    "default": extract_answer,
    "tx_gemma": extract_answer_tx_gemma,
    "finetuned": extract_answer_finetuned,
    "auto": extract_answer_auto,
}

LAST_EXTRACTOR_MAP = {
    "default": extract_last_answer,
    "tx_gemma": extract_last_answer,
    "finetuned": extract_last_answer,
    "auto": extract_last_answer,  # Use the same function for auto as well
}


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
        text = text.replace(".", "")  # Remove periods
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


    return 48, "No Match"  # return most common class


def parse_drugbank_outputs(predictions: list, drug_1_names: list, drug_2_names: list, extract_fn):
    error_label = 48  # default label for failed predictions

    prediction_idx = []
    parse_errors = []
    semantic_mismatches = []

    for i, pred in enumerate(predictions):
        pred_str = extract_fn(pred)

        if pred_str == -1:
            prediction_idx.append(error_label)
            parse_errors.append(pred)
            continue

        drug1 = drug_1_names[i]
        drug2 = drug_2_names[i]

        matched_relation_id, match_type = find_closest_match_without_drug_names2(
            pred_str, drug1, drug2
        )

        if match_type == 'No Match':
            prediction_idx.append(error_label)
            semantic_mismatches.append(pred)
        else:
            prediction_idx.append(matched_relation_id)

    total = len(predictions)
    logger.info(f"Total predictions: {total}")
    logger.info(f"Parse errors: {len(parse_errors)} ({len(parse_errors) / total:.2%})")
    logger.info(f"Semantic mismatches: {len(semantic_mismatches)} ({len(semantic_mismatches) / total:.2%})")
    logger.info(f"Total error-labeled predictions: {len(parse_errors) + len(semantic_mismatches)} ({(len(parse_errors) + len(semantic_mismatches)) / total:.2%})")

    return prediction_idx, parse_errors + semantic_mismatches

def parse_ddinter_outputs(predictions: list, last_extract_fn):
    # error_label = ERROR_IDX[args.dataset]
    error_label = 0

    prediction_idx = []
    all_errors = []
    for i, pred in enumerate(predictions):  # use enumerate here
        pred = last_extract_fn(pred)
        # print(i)
        if pred == None or not isinstance(pred, str):
            print(i)
        regex_pattern = "|".join(re.escape(v.strip()) for v in DDINTER_LABELS.keys())
        pred_str = re.findall(regex_pattern, pred.lower())
        if len(pred_str) == 0:
            prediction_idx.append(error_label)
            all_errors.append((i, pred))  # store index along with the error if needed
        else:
            prediction_idx.append(DDINTER_LABELS[pred_str[0]])
    
    logger.info(prediction_idx[0:10])
    logger.info(
        f"Number of errors: {len(all_errors)}; Percentage of errors: {len(all_errors) / len(predictions)}"
    )

    return prediction_idx, all_errors

def parse_pharmaDB_outputs(predictions: list,last_extract_fn):
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
    # error_label = ERROR_IDX[args.dataset]
    error_label = 0
    all_errors = []
    prediction_idx = []
    for pred in predictions:
        # pred_str = extract_answer(pred)
        pred_str = last_extract_fn(pred)
        # print(pred_str)
        # if pred_str == -1:
        if pred_str == -1 or not isinstance(pred_str, str):
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

def load_test_dataset(path):
    # print(path)
    return pd.read_json(path,lines=True)
    
    # return Dataset.from_json(path)

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


def compute_metrics(prediction_path, dataset_string, use_options, option_style, model_style):
    dataset = load_test_dataset(TEST_SET_PATH[dataset_string])
    labels = dataset["label_idx"]
    predictions_df = load_dataset_predictions(prediction_path, dataset)
    predictions = predictions_df["prediction"].tolist()

    extract_fn = EXTRACTOR_MAP[model_style]
    last_extract_fn = LAST_EXTRACTOR_MAP[model_style if model_style != "finetuned" else "auto"]

    if "drugbank" in dataset_string:
        logger.info("Parsing DrugBank outputs")
        drug_1_names = dataset["drug1_name"]
        drug_2_names = dataset["drug2_name"]
        predictions, all_errors = parse_drugbank_outputs(predictions, drug_1_names, drug_2_names, extract_fn)

    elif dataset_string == "ddinter":
        logger.info("Parsing DDInter outputs")
        predictions, all_errors = parse_ddinter_outputs(predictions, last_extract_fn)

    elif dataset_string == "pharmaDB":
        logger.info("Parsing pharmaDB outputs")
        predictions, all_errors = parse_pharmaDB_outputs(predictions, last_extract_fn)
        print(predictions[:10])

    else:
        raise NotImplementedError("Dataset not implemented")

    all_preds = np.array(predictions)
    all_labels = np.array(labels)

    if len(all_preds) != len(all_labels):
        logger.warning("Length mismatch between predictions and labels")
        all_labels = all_labels[: len(all_preds)]
        logger.info(f"Number of unique labels: {len(set(all_labels))}")
        logger.info(f"Number of unique predictions: {len(set(all_preds))}")

    accuracy = np.mean(all_preds == all_labels)
    f1 = f1_score(all_labels, all_preds, average=None).mean()
    kappa = cohen_kappa_score(all_labels, all_preds)

    logger.info(f"Accuracy: {accuracy * 100:.2f}")
    logger.info(f"F1 Score: {f1 * 100:.2f}")
    logger.info(f"Cohen's Kappa: {kappa * 100:.2f}")
    logger.info(
        f"Number of errors: {len(all_errors)}; Percentage of errors: {len(all_errors) / len(predictions)}"
    )

    outputs = {
        "accuracy": accuracy,
        "f1": f1,
        "kappa": kappa,
        "errors": all_errors,
        "error_rate": len(all_errors) / len(all_preds),
    }

    return outputs, all_preds


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Parse the outputs of the evaluation")

    parser.add_argument(
        "--prediction_path", type=str, required=True,
        help="Path to the predictions file or directory"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["drugbank_transductive", "drugbank_inductive", "ddinter", "pharmacotherapyDB"],
        help="Dataset to evaluate on",
    )
    parser.add_argument("--use_options", action="store_true")
    parser.add_argument(
        "--option_style", type=str, default="numbered", choices=["numbered", "bulleted"]
    )
    parser.add_argument("--variant", type=str, default="v1")
    parser.add_argument("--output_dir", type=str, default="outputs")

    #Allow user to specify model output format
    parser.add_argument(
        "--model_style",
        type=str,
        default="default",
        choices=["default", "tx_gemma", "finetuned", "auto"],
        help="Model output format style for answer extraction",
    )

    args = parser.parse_args()

    # Dataset name normalization
    dataset_map = {
        "drugbank_transductive": "drugbank",
        "drugbank_inductive": "drugbank",
        "pharmaDB": "pharmaDB",
        "ddinter": "ddinter",
    }
    dataset_string = dataset_map[args.dataset]

    outputs, _ = compute_metrics(
        prediction_path=args.prediction_path,
        dataset_string=dataset_string,
        use_options=args.use_options,
        option_style=args.option_style,
        model_style=args.model_style,
    )

    # print(f"Outputs: {outputs}")

    # # Construct output path
    # file_name = "regex_metrics.json"
    # if os.path.isdir(args.prediction_path):
    #     output_path = os.path.join(args.prediction_path, file_name)
    # else:
    #     base_name = os.path.basename(args.prediction_path)
    #     name, _ = os.path.splitext(base_name)
    #     output_path = os.path.join(os.path.dirname(args.prediction_path), f"{name}_{file_name}")

    # # Save to JSON
    # with open(output_path, "w") as f:
    #     json.dump(outputs, f, indent=2)

# run script: python evaluate_llm_regex.py \
#   --prediction_path outputs/google-txgemma-27b-chat-files-dataset_with_paths-drugbank_test_add_reverse-json-predictions-text.csv \
#   --dataset pharmacotherapyDB \
#   --model_style tx_gemma