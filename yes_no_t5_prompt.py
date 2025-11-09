# all imports
import argparse
import re
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm
import json
import jsonlines    
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, accuracy_score)
import evaluate
rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")
import pandas as pd
import matplotlib.pyplot as plt
MODEL_NAME = "google/flan-t5-xxl"
BATCH_SIZE = 64


def extract_answer_from_text(text):
    patterns = [
        r'final answer (?:is|:)\s*(yes|no)\b',
        r'\b(yes|no)\b'
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            label = m.group(1).lower()
            return label
    return None

def compute_cot_accuracy(generated_text, targets):
    if isinstance(generated_text, str):
        predictions = [generated_text]
    else:
        predictions = generated_text
    if isinstance(targets, str):
        targets = targets
    else:
        targets = targets
    
    pred_labels = [extract_answer_from_text(p) for p in predictions]
    true_labels = [extract_answer_from_text(t) for t in targets]
    valid = [(p, t) for p, t in zip(pred_labels, true_labels)
             if p is not None and t is not None]
    if not valid:
        return 0.0
    correct = sum(p == t for p, t in valid)
    return correct / len(valid)
    
def calculate_metrics_for_prompt(generated_text, target, prompt_type, reasoning=None):
    generated_text = generated_text.strip().lower()
    target = target.strip().lower()
    if prompt_type == "cot":
        reasoning = reasoning.strip().lower()
        # exact‐match accuracy
        accuracy = compute_cot_accuracy(generated_text, target)
        # BLEU
        try:
            bleu_score = bleu.compute(
                predictions=[generated_text],
                references=[reasoning],
                max_order=4
            )["bleu"]
        except ZeroDivisionError:
            bleu_score = 0.0

        # ROUGE-1
        rouge_score = rouge.compute(
            predictions=[generated_text],
            references=[reasoning],
            rouge_types=["rouge1"]
        )["rouge1"]
        # token‐overlap precision
        ref_tokens  = reasoning.split()
        cand_tokens = generated_text.split()
        true_set    = set(ref_tokens)
        pred_set    = set(cand_tokens)
        common      = true_set & pred_set
        precision   = len(common) / len(pred_set) if pred_set else 0.0
    else:
        bleu_score = rouge_score = precision = 0.0
        accuracy = int(generated_text.strip().lower() == target.strip().lower())

    return {
        "bleu_score":      bleu_score,
        "rouge_score":     rouge_score,
        "precision_score": precision,
        "accuracy":        accuracy
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="T5", required=False, help="Model name to use")
    parser.add_argument("--prompt_type", type=str, default="zero", required=True, help="Prompt type to use")
    parser.add_argument("--text_or_graph", type=str, default="text", required=True, help="Text or graph")
    args = parser.parse_args()
    
    T5_or_LLAMA = args.model_name
    PROMPT_TYPE = args.prompt_type
    MAX_TOKEN   = 200 if PROMPT_TYPE == "cot" else 8
    
    print(f"Using model: {T5_or_LLAMA}")
    print(f"Using prompt type: {PROMPT_TYPE}")
    print(f"Using text or graph: {args.text_or_graph}\n")
    
    # Load data
    data_file_dir = "data/yes_no_prompts/"
    
    if PROMPT_TYPE == "zero":
        prompt_file_name = "zero_shot"
    elif PROMPT_TYPE == "few":
        prompt_file_name = "few_shot"
    elif PROMPT_TYPE == "cot":
        prompt_file_name = "cot"
    
    if args.text_or_graph == "text":
        TEXT_OR_GRAPH =  "text_"
    elif args.text_or_graph == "graph":
        TEXT_OR_GRAPH =  "graph_"
    elif args.text_or_graph == "text_graph":
        TEXT_OR_GRAPH =  "text_graph_"
    
    input_file        = data_file_dir + "prompts/" + f"{args.text_or_graph}/" + TEXT_OR_GRAPH + prompt_file_name + ".jsonl"    # change this for other prompts
    
    jsonl_output_file = data_file_dir + "full_run_1/" + T5_or_LLAMA + "_" + TEXT_OR_GRAPH + prompt_file_name + "_scored.jsonl"   # write scored JSONL here
    csv_output_file   = data_file_dir + "full_run_1/" + T5_or_LLAMA + "_" + TEXT_OR_GRAPH + prompt_file_name + "_scored.csv"
    
    print(f"\nLoading data from {input_file}")
    print(f"\nWriting scored data to {jsonl_output_file}")
    print(f"\nWriting CSV data to {csv_output_file}\n")
    
    # Load the JSONL file
    all_data = []
    with jsonlines.open(input_file) as reader:
        for i, obj in enumerate(reader):
            # if i>=1024:
            #     break
            all_data.append(obj)
        
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16,
                                                  tp_plan="auto"
                                                ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    all_pred_labels = []
    all_true_labels = []
    
    for batch_num, i in enumerate(tqdm(range(0, len(all_data), BATCH_SIZE), desc="Processing Batches"), start=1):
        batch_lines = all_data[i:i + BATCH_SIZE]
        batch_prompts = [item['prompt'] for item in batch_lines]
        batch_targets = [item['target_answer'] for item in batch_lines]
        if PROMPT_TYPE == "cot":
            batch_reason = [item['target_reason'] for item in batch_lines]
        else:
            batch_reason = [None] * len(batch_lines)
        # move inputs to device
        batch_inputs = tokenizer(batch_prompts, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            batch_output_ids = model.generate(
                **batch_inputs,
                max_new_tokens=MAX_TOKEN,
                temperature=0.0
            )
        batch_generated_texts = tokenizer.batch_decode(batch_output_ids, skip_special_tokens=True)
        
        # Calculate metrics for each prompt in the batch
        for item, gen, tgt, rsn in zip(batch_lines, batch_generated_texts, batch_targets, batch_reason):
            metrics = calculate_metrics_for_prompt(gen, tgt, PROMPT_TYPE, rsn)
            # Update the item with the calculated scores
            item['predicted_answer'] = gen
            item['score'] = metrics['bleu_score']
            item['bleu_score'] = metrics['bleu_score']
            item['rouge_score'] = metrics['rouge_score']
            item['precision_score'] = metrics['precision_score']
            item['accuracy']        = metrics['accuracy']
            
            pred_label = extract_answer_from_text(gen)
            true_label = extract_answer_from_text(tgt)
            if pred_label is not None and true_label is not None:
                all_pred_labels.append(pred_label)
                all_true_labels.append(true_label)
            
    # After processing each batch, save the updated data
    with jsonlines.open(jsonl_output_file, mode="w") as writer:
        for record in all_data:
            writer.write(record)

    # Turn into a DataFrame and export
    df = pd.DataFrame(all_data)
    df.to_csv(csv_output_file, index=False)
    print(f"Results saved to {csv_output_file}")
    
    # Final confusion matrix + metrics
    labels = ["yes", "no"]
    cm = confusion_matrix(all_true_labels, all_pred_labels, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix: Yes/No")
    plt.show()
    
    print("\nConfusion Matrix (rows=true, cols=predicted):")
    print(pd.DataFrame(cm, index=[f"True: {l}" for l in labels], columns=[f"Pred: {l}" for l in labels]))

    acc = accuracy_score(all_true_labels, all_pred_labels)
    prec = precision_score(all_true_labels, all_pred_labels, pos_label="yes", zero_division=0)
    rec = recall_score(all_true_labels, all_pred_labels, pos_label="yes", zero_division=0)
    f1 = f1_score(all_true_labels, all_pred_labels, pos_label="yes", zero_division=0)

    print("\nClassification Metrics:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")