"""Create reproducible ground-truth artifacts from the owner's final review.

The labels in this script come from the review table supplied by the dataset
owner.  They are *user-reviewed labels*, not an independently adjudicated
benchmark.  Keeping this distinction in the filenames prevents accidental
overclaiming in the paper.
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "groundtruth" / "generated"

PARTLY_SUPPORTED = {
    "P001_G2", "P001_G3", "P002_G1", "P002_G3", "P004_G1", "P007_G3",
    "P010_G3", "P011_G3", "P016_G3", "P020_G1", "P021_G1", "P022_G3",
    "P028_G3", "P029_G3", "P030_G2", "P031_G3", "P035_G1", "P036_G1",
    "P042_G2", "P043_G2", "P044_G1", "P045_G1", "P046_G3", "P047_G3",
    "P048_G2", "P052_G1", "P057_G1", "P060_G1", "P061_G1",
}
EXCLUDE = {
    "P009_G3", "P010_G1", "P027_G1", "P033_G1", "P041_G1", "P041_G2",
    "P053_G1", "P053_G2", "P053_G3",
}


def main():
    queries = list(csv.DictReader((GEN / "queries_DRAFT.csv").open(encoding="utf-8-sig")))
    ids = {row["query_id"] for row in queries}
    unknown = (PARTLY_SUPPORTED | EXCLUDE) - ids
    overlap = PARTLY_SUPPORTED & EXCLUDE
    if unknown or overlap or len(queries) != 183:
        raise ValueError(f"Invalid review partition: unknown={unknown}, overlap={overlap}, n={len(queries)}")

    rows = []
    for query in queries:
        query_id = query["query_id"]
        if query_id in EXCLUDE:
            decision, support = "exclude", "excluded_by_user_review"
        elif query_id in PARTLY_SUPPORTED:
            decision, support = "keep", "partly_supported"
        else:
            decision, support = "keep", "supported"
        rows.append({**query, "decision": decision, "review_status": "user_reviewed",
                     "content_support": support})

    included = [row for row in rows if row["decision"] == "keep"]
    if len(included) != 174:
        raise ValueError(f"Expected 174 included queries, got {len(included)}")

    with (GEN / "qrels_target_USER_REVIEWED.tsv").open("w", encoding="utf-8") as handle:
        for row in included:
            handle.write(f"{row['query_id']}\t0\t{row['candidate_movie_id']}\t1\n")
    with (GEN / "groundtruth_audit_USER_REVIEWED.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "status": "user_reviewed_target_qrels",
        "input_queries": len(rows),
        "included_queries": len(included),
        "supported": sum(row["content_support"] == "supported" for row in included),
        "partly_supported": sum(row["content_support"] == "partly_supported" for row in included),
        "excluded": sorted(EXCLUDE),
        "label_source": "dataset owner's final review table",
        "reporting_note": "Describe these as user-reviewed target labels. Independent double adjudication is still required for a claim of independently adjudicated ground truth.",
    }
    (GEN / "groundtruth_summary_USER_REVIEWED.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
