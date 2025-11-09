# all imports
import argparse
import re
import torch
torch.cuda.empty_cache()
from tqdm import tqdm
import jsonlines
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay)
import evaluate
rouge = evaluate.load("rouge")
bleu  = evaluate.load("bleu")
import pandas as pd
import matplotlib.pyplot as plt

from openai import OpenAI
client = OpenAI(api_key="your-api-key-here")  # Replace with your OpenAI API key
BATCH_SIZE = 16

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

def extract_reason_from_text(text):
    pattern = re.compile(
        r'(?:Reasoning:\s*)?'          # optional leading “Reasoning:”
        r'(?P<reason>.*?)'             # non-greedy capture of reasoning
        r'(?:final answer (?:is|:)\s*' # the marker
        r'(?:yes|no)\b)', # final answer token
        re.IGNORECASE | re.DOTALL
    )
    m = pattern.search(text)
    if m:
        reason = m.group("reason").strip()
        return reason
    # Pull everything before the first yes/no/unanswerable
    answer_label = extract_answer_from_text(text)
    if answer_label:
        idx = re.search(rf'\b{answer_label}\b', text, flags=re.IGNORECASE)
        if idx:
            reason = text[:idx.start()].strip()
            return reason
    return None

def compute_cot_accuracy(generated_text, targets):
    if isinstance(generated_text, str):
        predictions = [generated_text]
    else:
        predictions = generated_text
    if isinstance(generated_text, str):
        predictions = [generated_text]
    if isinstance(targets, str):
        targets = [targets]
    
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
    parser.add_argument("--model_name", type=str, default="GPT", required=False, help="Model name to use")
    parser.add_argument("--prompt_type", type=str, default="zero", required=True, help="Prompt type to use")
    parser.add_argument("--text_or_graph", type=str, default="graph", required=True, help="Text or graph")
    args = parser.parse_args()
    
    PROMPT_TYPE = args.prompt_type
    MAX_TOKEN   = 200 if PROMPT_TYPE == "cot" else 8
    
    if PROMPT_TYPE == "cot":
        OPENAI_MODEL = "gpt-4o"
    else:
        OPENAI_MODEL = "gpt-3.5-turbo"
    
    print(f"Using model: {OPENAI_MODEL}")
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
    
    jsonl_output_file = data_file_dir + "full_run_1/" + OPENAI_MODEL + "_" + TEXT_OR_GRAPH + prompt_file_name + "_scored.jsonl"   # write scored JSONL here
    csv_output_file   = data_file_dir + "full_run_1/" + OPENAI_MODEL + "_" + TEXT_OR_GRAPH + prompt_file_name + "_scored.csv"
    
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
    def call_openai_chat(prompt):
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=MAX_TOKEN
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print("Error:", e)
            return ""
    
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
        
        batch_generated_answer = []
        batch_generated_label  = []
        batch_generated_reason = []
        
        for prompt in batch_prompts:
            raw = call_openai_chat(prompt)
            answer_block = raw.split("### Instruction")[0].strip()
            label = extract_answer_from_text(answer_block) or ""
            reason = extract_reason_from_text(answer_block) or ""
            full_answer = reason + " " + label
            batch_generated_label.append(label)
            batch_generated_reason.append(reason)
            batch_generated_answer.append(full_answer)
        
        # Calculate metrics for each prompt in the batch
        for item, gen_ans, gen_lbl, tgt, rsn in zip(batch_lines, batch_generated_answer, batch_generated_label, batch_targets, batch_reason):
            metrics = calculate_metrics_for_prompt(gen_ans, tgt, PROMPT_TYPE, rsn)
            item['predicted_answer'] = gen_ans
            item['predicted_label'] = gen_lbl
            item['score'] = metrics['bleu_score']
            item['bleu_score'] = metrics['bleu_score']
            item['rouge_score'] = metrics['rouge_score']
            item['precision_score'] = metrics['precision_score']
            item['accuracy'] = metrics['accuracy']

            pred_label = extract_answer_from_text(gen_lbl)
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