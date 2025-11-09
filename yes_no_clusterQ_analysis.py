import os
import re
import argparse
import pandas as pd
from collections import Counter
from itertools import chain

# Define your keyword groups
negative_keywords = ["did not", "will not", "won't", "didn't", "has not", "not happen", "not started", "not finished"]
past_keywords     = ["happened before", "did not happen before", "did happen before",
                    "happened after", "did not happen after", "did happen after",
                    "already happened", "already finished", "started before", "finished before"]
present_keywords  = ["happening now", "happened during", "started during",
                    "has started", "is happening", "are happening", "has begun"]
future_keywords   = ["will start", "will finish", "will begin", "will end",
                    "will be done", "will be finished", "will be started", "will be ended",
                    "might happen", "may happen", "will happen", "will have happened",
                    "going to happen"]
possible_hypo     = ["could happen", "could have happened", "could be happening", "could have been happening",
                    "might happen", "may happen", "possibly happened", "possibly happen",
                    "is it possible that"]
event_keywords    = ["event has started", "event has not started", "event has already finished",
                    "event has not finished", "event has begun", "event has not begun",
                    "event has ended", "event has not ended", "event has happened",
                    "does the event", "when does the event"]
causal_keywords   = ["caused", "lead to", "resulted in", "triggered", "enabled",
                    "due to", "because of"]
temporal_conflict_keywords = ["before and after", "at the same time", "overlap", "simultaneous", "during", 
                              "while"]
existential_keywords = ["did happen", "did occur", "was there", "did exist", 
                        "does exist"]
counterfactual_keywords = ["if it had", "even if", "had it been", "would have", "should have"]

positive_keywords = ["did happen", "has happened", "will happen", "is happening",
                    "does exist", "did occur", "was there", "is true"]

# Answer Extraction ===
def extract_answer_from_text(text):
    patterns = [
        r'final answer (?:is|:)\s*(yes|no)\b',
        r'\b(yes|no)\b'
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return m.group(1).lower()
    return None

# Classification function
def classify_single_question_multi(q):
    q = q.lower().strip()
    matched = []

    # Occurrence-style: "Did X happen?" without temporal anchors
    if re.match(r"^did .* happen[?]?$", q) and "before" not in q and "after" not in q:
        matched.append("occurrence")
    elif re.match(r"^has .* happened[?]?$", q) and "before" not in q and "after" not in q:
        matched.append("occurrence")

    # Hypothetical: "Could X have happened?"
    if q.startswith("could") and "have happened" in q:
        matched.append("possible")
    if any(k in q for k in possible_hypo):
        matched.append("possible")

    # Past temporal: before/after
    if q.startswith("did") and ("before" in q or "after" in q):
        matched.append("past")
    if any(k in q for k in past_keywords):
        matched.append("past")
        
    # Future/present/past
    if any(k in q for k in future_keywords):
        matched.append("future")
    if any(k in q for k in present_keywords):
        matched.append("present")

    # Causal reasoning
    if any(k in q for k in causal_keywords):
        matched.append("causal")

    # Temporal conflict: "during", "overlap", etc.
    if any(k in q for k in temporal_conflict_keywords):
        matched.append("temporal_conflict")

    # Negation
    if any(k in q for k in negative_keywords):
        matched.append("negative")

    # Event-oriented
    if any(k in q for k in event_keywords):
        matched.append("event")

    # Existential
    if any(k in q for k in existential_keywords):
        matched.append("existential")

    # Counterfactual
    if any(k in q for k in counterfactual_keywords):
        matched.append("counterfactual")

    # Positive (only if it isn’t already negated)
    if any(k in q for k in positive_keywords) and not any(k in q for k in negative_keywords):
        matched.append("positive")

    return list(set(matched)) if matched else ["unknown"]


def classify_prompt_multilabel(prompt):
    qa_blocks = re.findall(r"### Question ###\s*(.*?)\s*### Answer ###", prompt, flags=re.DOTALL)
    all_types = []
    for q in qa_blocks:
        all_types.extend(classify_single_question_multi(q))
    return list(set(all_types)) if all_types else ["unknown"]


def analyze_file(file_path):
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith(".jsonl"):
        df = pd.read_json(file_path, lines=True)
    else:
        raise ValueError("Unsupported file format: use .csv or .jsonl")

    if "prompt" not in df.columns or "predicted_answer" not in df.columns or "target_answer" not in df.columns or "accuracy" not in df.columns:
        raise ValueError("File must contain 'prompt', 'predicted_answer', 'target_answer' and 'accuracy' columns.")

    # Extract predicted labels and compute accuracy
    df["predicted_label"] = df["predicted_answer"].apply(extract_answer_from_text)
    df["target_label"] = df["target_answer"].str.lower().str.strip()
    df["accuracy"] = (df["predicted_label"] == df["target_label"]).astype(int)
    
    df["question_clusters"] = df["prompt"].apply(classify_prompt_multilabel)
    df_exploded = df.explode("question_clusters")

    cluster_summary = (
        df_exploded
        .groupby("question_clusters")["accuracy"]
        .agg(num_questions="count", avg_accuracy="mean")
        .reset_index()
        .rename(columns={"question_clusters": "cluster"})
    )
    return cluster_summary
        

if __name__ == "__main__":

    folder     = "data/yes_no_prompts/full_run_1/"
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="T5", required=False, help="Model name to use T5/LLama/QwQ/gpt")
    parser.add_argument("--prompt", type=str, default="zero", required=True, help="Prompt type to use zero/few/cot")
    parser.add_argument("--modality", type=str, default="text", required=True, help="Modality to use text or graph or text_graph")
    args = parser.parse_args()
    
    MODEL_NAME  = args.model
    PROMPT_TYPE = args.prompt
    MODALITY    = args.modality
    
    if PROMPT_TYPE == "zero":
        prompt_file_name = "zero_shot"
    elif PROMPT_TYPE == "few":
        prompt_file_name = "few_shot"
    elif PROMPT_TYPE == "cot":
        prompt_file_name = "cot"
    
    if MODALITY == "text":
        TEXT_OR_GRAPH =  "text_"
    elif MODALITY == "graph":
        TEXT_OR_GRAPH =  "graph_"
    elif MODALITY == "text_graph":
        TEXT_OR_GRAPH =  "text_graph_"
    
    matched_file = folder + MODEL_NAME + "_" + MODALITY + "_" + prompt_file_name + "_scored.csv"
    print(f"Analyzing file: {matched_file}")
    
    result_df = analyze_file(matched_file)
    print(result_df)
    
    # Save to CSV
    out_file = folder + "cluster/" + f"{MODEL_NAME}_{MODALITY}_{PROMPT_TYPE}_cluster_accuracy.csv"
    result_df.to_csv(out_file, index=False)
    print(f"Results saved to: {out_file}")
    print("\n\n")
    print(result_df.iloc[:, -1].to_string(index=False))