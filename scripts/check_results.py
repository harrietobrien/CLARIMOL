import json, glob, os

base = os.path.expanduser("~/storage/CLARIMOL/output")

# Causal ablation
p = os.path.join(base, "ablation_heads/ablation_results.json")
if os.path.exists(p):
    d = json.load(open(p))
    print("=== CAUSAL ABLATION ===")
    for t in ["functional_group","ring_counting","chain_length","canonicalization","fragment_assembly"]:
        i = d["no_ablation"][t]["accuracy"]
        r = d["ring_circuit"][t]["accuracy"]
        c = d["random_controls"][t]["accuracy_mean"]
        print(f"  {t}: intact={i:.4f} ring_abl={r:.4f} delta={r-i:+.4f} rnd={c:.4f}")
else:
    print("No ablation results found")

# Mistral robustness
p2 = os.path.join(base, "robustness_mistral/robustness_results.json")
if os.path.exists(p2):
    d2 = json.load(open(p2))
    print("=== MISTRAL ROBUSTNESS ===")
    for t in d2:
        r = d2[t]
        print(f"  {t}: canon={r['canonical_accuracy']:.4f} rand={r['randomized_accuracy']:.4f} drop={r['drop_pp']:+.1f}pp cons={r['consistency']:.1%}")
else:
    print("No Mistral robustness results found")

# Downstream Mistral
for cond in ["downstream_mistral_pretrained", "downstream_mistral_direct"]:
    p3 = os.path.join(base, cond)
    if os.path.isdir(p3):
        print(f"=== {cond.upper()} ===")
        for task in ["retrosynthesis","reagent_prediction","forward_reaction_prediction"]:
            rp = os.path.join(p3, task, "results.json")
            if os.path.exists(rp):
                d3 = json.load(open(rp))
                print(f"  {task}: {d3}")
            else:
                print(f"  {task}: no results")
