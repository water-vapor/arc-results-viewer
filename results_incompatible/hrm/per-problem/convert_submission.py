import json
import os
import sys

num_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 50

script_dir = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(script_dir, 'submission.json')
suffix = f'_top{num_seeds*2}' if num_seeds < 50 else ''
output_file = os.path.join(script_dir, f'submission_converted{suffix}.json')

with open(input_file, 'r') as f:
    data = json.load(f)

result = {}
for problem_hash, test_cases in data.items():
    result[problem_hash] = []
    for seed in range(num_seeds):
        guess1_list = []
        guess2_list = []
        for test_case_dict in test_cases:
            attempt_idx1 = seed * 2 + 1
            attempt_idx2 = seed * 2 + 2
            guess1_list.append(test_case_dict[f"attempt_{attempt_idx1}"])
            guess2_list.append(test_case_dict[f"attempt_{attempt_idx2}"])

        result[problem_hash].append([seed, False, [guess1_list, guess2_list]])

with open(output_file, 'w') as f:
    json.dump(result, f)