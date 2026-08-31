#!/usr/bin/env python
"""What students already know by the start of week N.

The contract handed to a week's builder so that weeks can be written in parallel without
one of them using something a later week is supposed to introduce. Generated from
modules.yml, so it reflects the plan rather than any one author's memory of it.

    python tools/prior_knowledge.py 5
"""
import sys, pathlib, yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
course = yaml.safe_load((ROOT / "course.yml").read_text())
mods = yaml.safe_load((ROOT / "modules.yml").read_text())
by_id = {m["id"]: m for m in mods["modules"]}


def main(week):
    earlier = [s for s in course["schedule"] if s["n"] < week]
    this = next(s for s in course["schedule"] if s["n"] == week)
    ids = [m for s in earlier for m in s["modules"]]

    out = [f"# What students already know, entering week {week}", ""]
    if not ids:
        out += ["**Nothing.** This is the first session; assume no programming experience at all.",
                ""]
    else:
        out += ["Everything below has been taught. Use it freely, and do NOT re-teach it.", "",
                "## Concepts", ""]
        for s in earlier:
            for mid in s["modules"]:
                m = by_id[mid]
                out.append(f"- **week {s['n']} ({mid}, {m['title']})** — "
                           + ", ".join(m.get("teaches", [])))
        fns = [f for mid in ids for f in by_id[mid].get("functions", [])]
        if fns:
            out += ["", "## Functions they can already use", ""]
            out += [f"- `{f['name']}` — {f['does']}" for f in fns]
        ideas = [d for d in mods.get("plain_words", []) if d["module"] in ids]
        if ideas:
            out += ["", "## Ideas already named, and the wording that was used", "",
                    "**Reuse these sentences verbatim if you refer back to them.**", ""]
            out += [f"- **{d['idea']}** — {d['words']}" for d in ideas]

    out += ["", f"# What week {week} may NOT assume", "",
            "Anything not listed above. In particular, these arrive LATER and must not appear "
            "except in a setup cell with an explicit 'Coming later' note:", ""]
    later = [(s["n"], m) for s in course["schedule"] if s["n"] > week for m in s["modules"]]
    for n, mid in later[:8]:
        out.append(f"- week {n} ({mid}) — " + ", ".join(by_id[mid].get("teaches", [])[:6]))

    out += ["", f"# Week {week} itself", "",
            f"- **Modules:** {', '.join(this['modules'])}",
            f"- **Question:** {this['question']}",
            f"- **Teaches:** " + "; ".join(", ".join(by_id[m].get("teaches", []))
                                           for m in this["modules"]), ""]
    for t in this.get("takeaways", []):
        out.append(f"- **Takeaway:** {t}")
    return "\n".join(out)


if __name__ == "__main__":
    print(main(int(sys.argv[1])))
