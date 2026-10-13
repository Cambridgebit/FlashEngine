import json
import random


def load_prompts(path, category, num, seed):
    prompts = []
    with open(path, encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            item = json.loads(line)
            item_category = item.get("category")
            if category == "all" or item_category is None or item_category == category:
                prompts.append(item["turns"][0])
    if not prompts:
        raise SystemExit(f"No prompts found in {path} (category={category}).")
    if seed >= 0:
        random.Random(seed).shuffle(prompts)
    return prompts[:num] if num > 0 else prompts


def percentile(values, percent):
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def metric_delta(before, after, names):
    return {key: after[name] - before.get(name, after[name])
            for key, name in names.items() if name in after}
