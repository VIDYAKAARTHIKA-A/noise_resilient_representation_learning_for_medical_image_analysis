import json
try:
    with open('results.json', 'r') as f:
        res = json.load(f)
    for r in res:
        print(f"{r['stage']} ({r['noise_level']*100}%): {r['macro_f1']:.4f}")
except Exception as e:
    print(e)
