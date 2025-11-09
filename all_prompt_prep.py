import pandas as pd
import json
import jsonlines
import re
from tqdm import tqdm

# Instruction Templates
ZERO_SHOT_TEXT = (
    "### Instruction ###\n"
    "You are provided with a piece of text. Analyze the text carefully and answer “yes” or “no” only.\n"
)
FEW_SHOT_TEXT = (
    "### Instruction ###\n"
    "You are provided with a piece of text and examples showing how to answer. Analyze the text carefully and answer “yes” or “no” only.\n"
)
COT_TEXT = (
    "### Instruction ###\n"
    "You are provided with a piece of text. Analyze it step-by-step to answer the question. "
    "Follow the reasoning format below, then provide only “yes” or “no”.\n"
)

ZERO_SHOT_GRAPH = (
    "### Instruction ###\n"
    "You are provided with a causal graph. Analyze it carefully and answer “yes” or “no” only.\n"
)
FEW_SHOT_GRAPH = (
    "### Instruction ###\n"
    "You are provided with a causal graph and examples showing how to answer. Analyze it carefully and answer “yes” or “no” only.\n"
)
COT_GRAPH = (
    "### Instruction ###\n"
    "You are provided with a causal graph. Analyze it step-by-step to answer the question. "
    "Follow the reasoning format below, then provide only “yes” or “no”.\n"
)

ZERO_SHOT_TEXT_GRAPH = (
    "### Instruction ###\n"
    "You are provided with text and a causal graph. Integrate both and answer “yes” or “no” only.\n"
)
FEW_SHOT_TEXT_GRAPH = (
    "### Instruction ###\n"
    "You are provided with text, a causal graph, and examples showing how to answer. Integrate both and answer “yes” or “no” only.\n"
)
COT_TEXT_GRAPH = (
    "### Instruction ###\n"
    "You are provided with text and a causal graph. Analyze both step-by-step to answer the question. "
    "Follow the reasoning format below, then provide only “yes” or “no”.\n"
)

# Question‐to‐Template Rules
def template_question(orig_q, candidate):
    rules = [
    (r"^What .+ has already (finished|happened)\?$",      r"Has \"event_x\" already \1?"),
    (r"^What .+ has begun but (?:has not |not )?finished\?$",       r"Has \"event_x\" begun but not finished?"),
    (r"^What will happen in the future\?$",               r"Will \"event_x\" happen?"),
    (r"^What happened after (.+)\?$",                     r"Did \"event_x\" happen after \"\1\"?"),
    (r"^What happened before (.+)\?$",                    r"Did \"event_x\" happen before \"\1\"?"),
    (r"^What did not happen (before|after) (.+)\?$",      r"Did \"event_x\" not happen \1 \"\2\"?"),
    (r"^What is happening now\?$",                        r"Is \"event_x\" currently happening?"),
    (r"^What could have happened to (.+)\?$",             r"Could \"event_x\" have happened to \"\1\"?"),
    (r"^What does (.+) refer to\?$",                      r"Does \"\1\" refer to \"event_x\"?"),
    ]

    for pat, repl in rules:
        if re.match(pat, orig_q, re.IGNORECASE):
            tmpl = re.sub(pat, repl, orig_q, flags=re.IGNORECASE)
            return tmpl[0].upper() + tmpl[1:].replace("event_x", candidate)
    # Fallback: guard split
    parts = orig_q.rstrip('?').split(' ', 1)
    core = parts[1] if len(parts) > 1 else parts[0]
    return f"Is \"{candidate}\" {core.lower()}?"


# Graph Verbalization
def get_verbal_graph(causal_graph):
    """
    Convert graph entries into a textual description.
    causal_graph: list of {head, tail, rel}
    """
    desc = []
    for e in causal_graph:
        if e['rel'] == 'ENABLES':
            desc.append(f'The event "{e["head"]}" enables the event "{e["tail"]}".')
        elif e['rel'] == 'BLOCKS':
            desc.append(f'The event "{e["head"]}" blocks the event "{e["tail"]}".')
    return ' '.join(desc)


# Data Reading Helpers
def get_QnA(item):
    questions = item['questions']
    answers   = item['answers']
    all_as = sorted({a for sub in answers for a in sub})
    return questions, answers, all_as


# Prompt Builder
def build_prompts_for_group(q_group, a_group, text, causal_graph, all_events):
    """
    Returns dict with keys 'text','graph','textgraph', each mapping to {'zs','fs','cot'} prompts.
    """
    graph_text = get_verbal_graph(causal_graph)
    main_q = q_group[3]
    main_a = a_group[3]

    # Zero-Shot
    txt_zs = f"{ZERO_SHOT_TEXT}\n### Text ###\n{text}\n\n### Question ###\n{main_q}\n### Answer ###\n"
    gph_zs = f"{ZERO_SHOT_GRAPH}\n### Graph ###\n{graph_text}\n\n### Question ###\n{main_q}\n### Answer ###\n"
    tg_zs  = f"{ZERO_SHOT_TEXT_GRAPH}\n### Text ###\n{text}\n### Graph ###\n{graph_text}\n\n### Question ###\n{main_q}\n### Answer ###\n"

    # Few-Shot Examples
    exs = [f"Question: {q}\nAnswer: {a}" for q,a in zip(q_group[:3], a_group[:3])]
    ex_block = '\n'.join(exs)
    txt_fs = f"{FEW_SHOT_TEXT}\n### Text ###\n{text}\n\n### Examples ###\n{ex_block}\n\n### Question ###\n{main_q}\n### Answer ###\n"
    gph_fs = f"{FEW_SHOT_GRAPH}\n### Graph ###\n{graph_text}\n\n### Examples ###\n{ex_block}\n\n### Question ###\n{main_q}\n### Answer ###\n"
    tg_fs  = f"{FEW_SHOT_TEXT_GRAPH}\n### Text ###\n{text}\n### Graph ###\n{graph_text}\n\n### Examples ###\n{ex_block}\n\n### Question ###\n{main_q}\n### Answer ###\n"

    # Chain-of-Thought Examples
    cot_exs = []
    for q,a in zip(q_group[:3], a_group[:3]):
        _, reason = matching(q, all_events, causal_graph, graph_text)
        cot_exs.append(f"Question: {q}\nAnswer:\nReasoning: {reason}\nTherefore, the final answer is: {a}")
    cot_block = '\n\n'.join(cot_exs)
    
    _, main_reason = matching(main_q, all_events, causal_graph, graph_text)
    cot_target_reason = f"Answer: Reasoning: {main_reason}\n"
    
    txt_cot = f"{COT_TEXT}\n### Text ###\n{text}\n\n### Examples ###\n{cot_block}\n\n### Question ###\n{main_q}\n### Answer ###\n"
    gph_cot = f"{COT_GRAPH}\n### Graph ###\n{graph_text}\n\n### Examples ###\n{cot_block}\n\n### Question ###\n{main_q}\n### Answer ###\n"
    tg_cot  = f"{COT_TEXT_GRAPH}\n### Text ###\n{text}\n### Graph ###\n{graph_text}\n\n### Examples ###\n{cot_block}\n\n### Question ###\n{main_q}\n### Answer ###\n"

    return {
        'text':      {'zs': txt_zs,  'fs': txt_fs,  'cot': (txt_cot, cot_target_reason)},
        'graph':     {'zs': gph_zs,  'fs': gph_fs,  'cot': (gph_cot, cot_target_reason)},
        'textgraph': {'zs': tg_zs,   'fs': tg_fs,   'cot': ( tg_cot, cot_target_reason)},
    }


# Matching for COT Reasoning
def matching(question, all_events, causal_graph, verbal_graph):
    text_graphs = []
    for e in causal_graph:
        if e['head'] in question or e['tail'] in question:
            relation = 'enables' if e['rel']=='ENABLES' else 'blocks'
            text_graphs.append(f'The event "{e["head"]}" {relation} the event "{e["tail"]}".')
    return [], (verbal_graph if not text_graphs else ' '.join(sorted(set(text_graphs))))


if __name__ == "__main__":
    input_path = "data/main_train_data_only.jsonl"
    out_dir    = "data/yes_no_prompts/prompts/"
    files = {
        'text_zs':    out_dir + "text/"  + "text_zero_shot.jsonl",
        'text_fs':    out_dir + "text/"  + "text_few_shot.jsonl",
        'text_cot':   out_dir + "text/"  + "text_cot.jsonl",
        'graph_zs':   out_dir + "graph/" + "graph_zero_shot.jsonl",
        'graph_fs':   out_dir + "graph/" + "graph_few_shot.jsonl",
        'graph_cot':  out_dir + "graph/" + "graph_cot.jsonl",
        'textgraph_zs':      out_dir + "text_graph/" + "text_graph_zero_shot.jsonl",
        'textgraph_fs':      out_dir + "text_graph/" + "text_graph_few_shot.jsonl",
        'textgraph_cot':     out_dir + "text_graph/" + "text_graph_cot.jsonl",
    }

    # open all output files
    handles = {k: open(v, 'w', encoding='utf-8') for k,v in files.items()}
    with jsonlines.open(input_path) as reader:
        for item in tqdm(reader):
            text = item['text']
            questions, answers, all_as = get_QnA(item)
            causal_graph = item['causal_graph']
            verbal_graph = get_verbal_graph(causal_graph)
            
            # generate yes/no pairs
            yesno = []
            for q,a in zip(questions, answers):
                for cand in all_as:
                    yesno.append({
                        'question': template_question(q, cand),
                        'answer':   'yes' if cand in a else 'no'
                    })
            Q = [d['question'] for d in yesno]
            A = [d['answer']   for d in yesno]

            # sliding-window grouping
            questions_groups = [Q[i:i+4] for i in range(len(Q)-3)]
            answers_groups   = [A[i:i+4] for i in range(len(A)-3)]
            for q_group, a_group in zip(questions_groups, answers_groups):
                outputs = build_prompts_for_group(q_group, a_group, text, causal_graph, all_as)
                target_answer  = a_group[3]
                for modality in ['text','graph','textgraph']:
                    for style in ['zs','fs','cot']:
                        if style == 'cot':
                            prompt, target_reason = outputs[modality][style]
                            handles[f"{modality}_{style}"].write(
                            json.dumps({'prompt': prompt, 'target_answer': target_answer, 'target_reason': target_reason}) + "\n"
                        )
                        else:
                            handles[f"{modality}_{style}"].write(
                                json.dumps({'prompt': outputs[modality][style], 'target_answer': target_answer}) + "\n"
                            )
    # close all files
    for f in handles.values(): f.close()