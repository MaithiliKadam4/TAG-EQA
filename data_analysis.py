import os
import glob
import jsonlines
from collections import defaultdict

PROMPT_TYPES  = ["zero_shot", "few_shot", "cot"]
MODALITIES    = ["text", "graph", "text_graph"]


def avg(lst):
    return sum(lst)/len(lst) if lst else 0

def analyze_prompt_stats(base_dir="data/yes_no_prompts/prompts/"):
    overall = {
                'total_questions': 0,
                'empty_or_unanswerable': 0,
                'yes': 0,
                'no': 0
    }
    per_mod_style = defaultdict(lambda: 
                    {
                        'count': 0,
                        'empty_or_unanswerable': 0,
                        'yes': 0,
                        'no': 0,
                        'prompt_lens': [],
                        'answer_lens': [],
                        'reason_lens': []   # only used for cot
                    }
    )

    for modality in MODALITIES:
        for style in PROMPT_TYPES:
            filename = f"{modality}_{style}.jsonl"
            path = os.path.join(base_dir, modality, filename)
            if not os.path.exists(path):
                print(f"Warning: {path} not found, skipping.")
                continue

            stats = per_mod_style[(modality, style)]
            with jsonlines.open(path) as reader:
                for obj in reader:
                    stats['count'] += 1
                    overall['total_questions'] += 1

                    prompt_text = obj.get('prompt', "")
                    answer_text = obj.get('target_answer', "").strip().lower()

                    stats['prompt_lens'].append(len(prompt_text.split()))
                    stats['answer_lens'].append(len(answer_text.split()))

                    if answer_text == "" or answer_text == "unanswerable":
                        stats['empty_or_unanswerable'] += 1
                        overall['empty_or_unanswerable'] += 1
                    elif answer_text == "yes":
                        stats['yes'] += 1
                        overall['yes'] += 1
                    elif answer_text == "no":
                        stats['no'] += 1
                        overall['no'] += 1

                    if style == "cot":
                        reason_text = obj.get('target_reason', "").strip()
                        stats['reason_lens'].append(len(reason_text.split()))

    # Return both raw and summarized statistics
    return overall, per_mod_style

def print_prompt_stats(overall, per_mod_style):
    print("\n=== OVERALL SUMMARY ===")
    print(f"Total questions:             {overall['total_questions']}")
    print(f"Empty / unanswerable:        {overall['empty_or_unanswerable']}")
    print(f"  yes:                       {overall['yes']}")
    print(f"  no:                        {overall['no']}")

    for (mod, ptype), s in sorted(per_mod_style.items()):
        print(f"\n• {mod} [{ptype}]")
        print(f"  Count:                     {s['count']}")
        print(f"  Empty / unanswerable:      {s['empty_or_unanswerable']}")
        print(f"  yes:                       {s['yes']}")
        print(f"  no:                        {s['no']}")
        print(f"  Avg prompt length:         {avg(s['prompt_lens']):.1f}")
        print(f"  Avg answer length:         {avg(s['answer_lens']):.1f}")
        if ptype == 'cot':
            print(f"  Avg reason length:         {avg(s['reason_lens']):.1f}")

def calculate_missing_events(input_path):
    count_missing = []
    with jsonlines.open(input_path) as reader:
        for item in reader:
            # print("\nkeys:", item.keys())  # dict_keys(['torque_id', 'text', 'questions', 'answers', 'event_types', 'causal_graph'])
            # print("\n\ncausal_graph:", item['causal_graph'])
            # print("\n\nevent_types:", item['event_types'])
            
            causal_graph = item['causal_graph']
            head_tail_set = set((edge['head'], edge['tail']) for edge in causal_graph)
            
            event_types = item['event_types']
            event_keys_set = set(event_types.keys())
            
            missing_events = event_keys_set - head_tail_set
            count_missing.append(len(missing_events))
    return count_missing
            

if __name__ == "__main__":
    overall_stats, detailed_stats = analyze_prompt_stats()
    print_prompt_stats(overall_stats, detailed_stats)
    
    input_path = "data/main_train_data_only.jsonl"
    count_missing = calculate_missing_events(input_path)
    print("Average number of missing events:", avg(count_missing))
    


# 	                     ZERO SHOT			             FEW SHOT			      CHAIN OF THOUGHT		
# METRIC	   TEXT	   GRAPH	TEXT + GRAPH	TEXT	GRAPH	TEXT + GRAPH	TEXT	GRAPH	TEXT + GRAPH
# Empty / 
# unanswerable	  0	       0	   0	          0	        0	    0	           0	    0	    0
# Yes	        14081	14081	  14081	        14081	14081	    14081	    14081	14081	    14081
# No	        38980	38980	  38980	        38980	38980	    38980	    38980	38980	    38980
# Count	        53061	53061	  53061	        53061	53061	    53061	    53061	53061	    53061
# Avg prompt 
# length	    95.2	80.6	    138	        136.3	121.7	    178	        242.8	229.2	    287.6
# Avg answer 
# length	      1	       1	    1	           1	    1	    1	            1	    1	    1
# Avg reason 
# length							                                            30.7	30.7	    30.7


# yes	                            126729
# no	                            350820
# Total questions	                477549
# Average number of missing events	5.271