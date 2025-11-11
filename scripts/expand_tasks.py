#!/usr/bin/env python3
import os, sys, yaml, copy, json, re
from datetime import datetime

def load_seed(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def interpolate(text, env):
    def repl(m):
        key = m.group(1)
        return str(env.get(key, m.group(0)))
    return re.sub(r"\{([a-zA-Z0-9_:\-]+)\}", repl, text)

def labels_with(vars):
    return [interpolate(l, vars) for l in vars.get('_labels', [])]

def expand(seed):
    out = {
        "version": seed["version"],
        "repo": seed["repo"],
        "project": seed["project"],
        "labels": seed["labels"],
        "milestones": seed["milestones"],
        "epics": seed["epics"],
        "issues": []
    }
    vars_global = seed.get("vars", {})

    # include singletons
    for s in seed.get("singleton_issues", []):
        issue = copy.deepcopy(s)
        issue["labels"] = s.get("labels", [])
        out["issues"].append(issue)

    # expand generators
    for gen in seed.get("generators", []):
        bp = gen["blueprint"]
        loops = gen.get("for_each", {})
        # build cartesian product of loops (1 or many axes)
        axes = []
        keys = list(loops.keys())
        if keys:
            # resolve ${var} into list
            def resolve(val):
                if isinstance(val, str) and val.startswith("${") and val.endswith("}"):
                    name = val[2:-1]
                    return vars_global.get(name, [])
                return val
            lists = [resolve(loops[k]) for k in keys]
            def rec(i, cur):
                if i == len(keys):
                    yield cur
                else:
                    for v in lists[i]:
                        nxt = cur.copy()
                        nxt[keys[i]] = v
                        yield from rec(i+1, nxt)
            combos = list(rec(0, {}))
        else:
            combos = [{}]

        for env_vars in combos:
            issue = {
                "milestone": bp["milestone"],
                "title": interpolate(bp["title"], env_vars),
                "labels": [interpolate(l, env_vars) for l in bp.get("labels", [])],
                "parent": bp.get("parent"),
                "estimate_h": bp.get("estimate_h"),
                "body": interpolate(bp.get("body",""), env_vars),
                "acceptance": bp.get("acceptance", []),
                "tests": bp.get("tests", [])
            }
            out["issues"].append(issue)
    return out

def main():
    in_path = sys.argv[1] if len(sys.argv)>1 else "TASKS.seed.yaml"
    out_path = sys.argv[2] if len(sys.argv)>2 else "TASKS.yaml"
    seed = load_seed(in_path)
    tasks = expand(seed)
    with open(out_path, "w") as f:
        yaml.safe_dump(tasks, f, sort_keys=False)
    print(f"Wrote {out_path} with {len(tasks['issues'])} issues")

if __name__ == "__main__":
    main()
