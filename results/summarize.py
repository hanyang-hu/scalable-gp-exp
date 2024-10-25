"""
Example: 
[
    {
        "dataset": "elevators",
        "split": 0,
        "fraction": 0.1,
        "method": "FPS_keops",
        "seed": 42,
        "rmse": 0.4147663116455078,
        "nll": 0.5713725686073303,
        "ast": false
    },
    {
        "dataset": "elevators",
        "split": 1,
        "fraction": 0.1,
        "method": "FPS_keops",
        "seed": 42,
        "rmse": 0.4167042374610901,
        "nll": 0.5770530104637146,
        "ast": false
    },
    {
        "dataset": "elevators",
        "split": 0,
        "fraction": 0.1,
        "method": "random_sampling",
        "seed": 42,
        "rmse": 0.43556275963783264,
        "nll": 0.6046236157417297,
        "ast": false
    },
    {
        "dataset": "elevators",
        "split": 1,
        "fraction": 0.1,
        "method": "random_sampling",
        "seed": 42,
        "rmse": 0.4040585160255432,
        "nll": 0.5327125787734985,
        "ast": false
    }
]
"""

import json
import os
import sys


def summarize_statistics(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    summary = {}
    
    for entry in data:
        dataset = entry['dataset']
        method = entry['method']
        
        if dataset not in summary:
            summary[dataset] = {}
        
        if method not in summary[dataset]:
            summary[dataset][method] = {'rmse': [], 'nll': []}
        
        summary[dataset][method]['rmse'].append(entry['rmse'])
        summary[dataset][method]['nll'].append(entry['nll'])
    
    for dataset in summary:
        for method in summary[dataset]:
            rmse_values = summary[dataset][method]['rmse']
            nll_values = summary[dataset][method]['nll']
            
            summary[dataset][method] = {
                'rmse_mean': sum(rmse_values) / len(rmse_values),
                'rmse_std': (sum((x - sum(rmse_values) / len(rmse_values)) ** 2 for x in rmse_values) / len(rmse_values)) ** 0.5,
                'nll_mean': sum(nll_values) / len(nll_values),
                'nll_std': (sum((x - sum(nll_values) / len(nll_values)) ** 2 for x in nll_values) / len(nll_values)) ** 0.5
            }
    
    return summary

if __name__ == "__main__":
    file_path = os.path.join(os.path.dirname(__file__), 'offline_subsampling_exp_done.json')
    summary = summarize_statistics(file_path)
    print(json.dumps(summary, indent=4))