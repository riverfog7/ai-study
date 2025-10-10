#!/usr/bin/env python3
import json
from pathlib import Path
from collections import defaultdict

RESP_PATH = Path("responses.json")
DATASET_ROOT = Path("dataset")
MIXED_ROOT = DATASET_ROOT / "mixed"
CATEGORIES_ORDER = ["mixed", "memes", "advertising", "poems", "recipie"]
OUT_MD = Path("report.md")

def load_responses(p: Path):
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def build_category_index(data, category):
    # Returns: { img_path: { question: answer } }
    per_img = defaultdict(dict)
    for question, catmap in data.items():
        for img_path, resp in catmap.get(category, []):
            per_img[img_path][question] = resp
    return per_img

def get_mixed_subcat(img_path: str) -> str:
    p = Path(img_path)
    try:
        rel = p.relative_to(MIXED_ROOT)
        if len(rel.parts) > 1:
            return rel.parts[0]
    except Exception:
        pass
    parts = p.parts
    if "mixed" in parts:
        idx = parts.index("mixed")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return "unknown"

def rel_path_for_md(img_path: str) -> str:
    p = Path(img_path)
    try:
        rp = p.relative_to(Path().resolve())
    except Exception:
        rp = p
    return rp.as_posix()

def add_heading(md_list, text, level):
    md_list.append("")  # ensure blank line before every heading
    md_list.append(f"{'#' * level} {text}")

def make_report(data):
    md = []
    questions_order = list(data.keys())

    # Raised one more level relative to the previous script:
    # - Category: # (was ##)
    # - Mixed subcategory: ## (was ###)
    # - Mixed image: ### (was ####)
    # - Mixed question: #### (was #####)
    # - Non-mixed image: ## (was ###)
    # - Non-mixed question: ### (was ####)
    for category in CATEGORIES_ORDER:
        add_heading(md, category, 1)  # # Category

        if category == "mixed":
            per_img = build_category_index(data, category)
            subcat_groups = defaultdict(list)
            for img_path in sorted(per_img.keys()):
                subcat = get_mixed_subcat(img_path)
                subcat_groups[subcat].append(img_path)

            for subcat in sorted(subcat_groups.keys()):
                add_heading(md, subcat, 2)  # ## mixed subcategory
                for img_path in sorted(subcat_groups[subcat]):
                    img_name = Path(img_path).name
                    add_heading(md, img_name, 3)  # ### image name
                    md.append(f'<img src="{rel_path_for_md(img_path)}" alt="{img_name}" width="500" />')
                    md.append("")  # extra blank line after image
                    qas = per_img.get(img_path, {})
                    for q in questions_order:
                        if q in qas:
                            a = qas[q] if qas[q] is not None else ""
                            add_heading(md, q, 4)  # #### question
                            md.append(f"{a}")
                    md.append("")  # blank after each image block

        else:
            per_img = build_category_index(data, category)
            for img_path in sorted(per_img.keys()):
                img_name = Path(img_path).name
                add_heading(md, img_name, 2)  # ## image name
                md.append(f'<img src="{rel_path_for_md(img_path)}" alt="{img_name}" width="500" />')
                md.append("")  # extra blank line after image
                qas = per_img.get(img_path, {})
                for q in questions_order:
                    if q in qas:
                        a = qas[q] if qas[q] is not None else ""
                        add_heading(md, q, 3)  # ### question
                        md.append(f"{a}")
                md.append("")  # blank after each image block

        md.append("")  # blank after each category

    return "\n".join(md).strip() + "\n"

def main():
    if not RESP_PATH.exists():
        raise FileNotFoundError(f"Missing {RESP_PATH}")
    data = load_responses(RESP_PATH)
    report = make_report(data)
    OUT_MD.write_text(report, encoding="utf-8")
    print(f"Wrote {OUT_MD.resolve()}")

if __name__ == "__main__":
    main()
