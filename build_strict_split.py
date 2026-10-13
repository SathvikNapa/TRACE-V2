"""
build_strict_split.py
-----------------------
Builds the "strict_v2" split called for in docs/TRACE_v2_proposal.md Section 2:
a speaker/conversation-identity-disjoint train/val/test split, as the Week-1
completion gate ("Reproduced scores or documented discrepancies; zero
unaccounted leakage").

Why this is needed: the existing split (legacy_v1, used by sweep_trace_v2.sh
and all prior runs) reports one identity crossing splits in 12 rows (per the
paper's own leakage audit). This script builds a split with zero identity
crossing, auditing it the same way the previous (unrelated) SPEAR/NIPS project
did -- that project found a dominating connected component covering 88% of
rows, making a clean split infeasible there. DyadEE is structured very
differently: 500 connected components across 2125 conversations, largest only
129 conversations (6.1%), median size 2 -- a real speaker-disjoint split is
achievable here at a reasonable size.

Identity recovery (schema quirks worth documenting):
  - row_id is the join key shared across all 4 augmentation_type variants of
    the same underlying conversation (original, emotivoice_tts, original_vc,
    emotivoice_tts_vc) -- confirmed via master_dataset_{train,test}_df.csv.
  - source_row_p1/source_row_p2 are always empty (0/5131 non-null) -- dead
    columns, not usable.
  - participant*_relpath_abs only reliably encodes identity (a P#### token)
    on "original" rows; original_vc rows have no identity in their path at
    all. Fix: recover every row_id's "original" row's relpaths first, then
    parse P####, then join to dyad_lookup_enriched.csv on the wav basename to
    get (vendor_id, participant_id) -- the real globally-unique identity,
    since raw P#### numbering is not guaranteed unique across vendors.

Usage: python build_strict_split.py [--val-frac 0.1] [--test-frac 0.392]
Writes strict_v2/{train,val,test}.csv + strict_v2/audit.json under
golden_datasets/cumulative_dataset_v6/.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

TRAIN_CSV = "golden_datasets/cumulative_dataset_v6/master_dataset_train_df.csv"
TEST_CSV = "golden_datasets/cumulative_dataset_v6/master_dataset_test_df.csv"
DYAD_LOOKUP = "dyad_lookup_enriched.csv"
OUT_DIR = "golden_datasets/cumulative_dataset_v6/strict_v2"
SEED = 13


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def recover_identity(combined: pd.DataFrame) -> pd.DataFrame:
    """Returns one row per row_id with (p1_vendor_id, p1_participant_id,
    p2_vendor_id, p2_participant_id) recovered via the "original" row's path,
    joined against dyad_lookup_enriched.csv on wav basename."""
    orig = combined[combined.augmentation_type == "original"][
        ["row_id", "participant1_relpath_abs", "participant2_relpath_abs"]
    ].copy()
    orig["bn1"] = orig.participant1_relpath_abs.apply(os.path.basename)
    orig["bn2"] = orig.participant2_relpath_abs.apply(os.path.basename)

    dl = pd.read_csv(DYAD_LOOKUP, low_memory=False)
    dl["bn1"] = dl.participant1_relpath.apply(lambda p: os.path.basename(str(p)))
    dl["bn2"] = dl.participant2_relpath.apply(lambda p: os.path.basename(str(p)))
    dl_small = dl[
        ["bn1", "bn2", "p1_vendor_id", "p1_participant_id", "p2_vendor_id", "p2_participant_id"]
    ].drop_duplicates()

    merged = orig.merge(dl_small, on=["bn1", "bn2"], how="left")
    n_missing = merged.p1_vendor_id.isna().sum()
    if n_missing:
        raise RuntimeError(
            f"{n_missing} row_ids failed to match dyad_lookup_enriched.csv -- "
            "identity recovery is incomplete, do not proceed with a split."
        )
    return merged


def build_components(identity: pd.DataFrame) -> pd.Series:
    uf = UnionFind()
    id1 = list(zip(identity.p1_vendor_id, identity.p1_participant_id))
    id2 = list(zip(identity.p2_vendor_id, identity.p2_participant_id))
    for a, b in zip(id1, id2):
        uf.union(a, b)
    return pd.Series([uf.find(a) for a in id1], index=identity.row_id, name="component")


def assign_split(component_of_row_id: pd.Series, val_frac: float, test_frac: float, seed: int) -> pd.Series:
    """Greedy size-balanced assignment of CONNECTED COMPONENTS (not individual
    conversations) to train/val/test, so no identity crosses splits."""
    sizes = component_of_row_id.value_counts().to_dict()  # {component: n_row_ids}
    comp_ids = list(sizes.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(comp_ids)

    total = sum(sizes.values())
    target_val, target_test = val_frac * total, test_frac * total

    val_set, test_set = set(), set()
    running_val, running_test = 0, 0
    for cid in comp_ids:
        if running_val < target_val:
            val_set.add(cid)
            running_val += sizes[cid]
        elif running_test < target_test:
            test_set.add(cid)
            running_test += sizes[cid]
        # else: falls through to train

    def split_of(cid):
        if cid in val_set:
            return "val"
        if cid in test_set:
            return "test"
        return "train"

    return component_of_row_id.map(split_of)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--test-frac", type=float, default=0.392)  # ~ legacy_v1's row-level test fraction
    args = ap.parse_args()

    train = pd.read_csv(TRAIN_CSV, low_memory=False)
    test = pd.read_csv(TEST_CSV, low_memory=False)
    combined = pd.concat([train, test], ignore_index=True)

    identity = recover_identity(combined)
    component_of_row_id = build_components(identity)

    sizes = component_of_row_id.value_counts()
    print(f"components: {len(sizes)} | conversations: {len(component_of_row_id)} | "
          f"largest component: {sizes.iloc[0]} ({sizes.iloc[0] / len(component_of_row_id):.1%})")

    row_id_split = assign_split(component_of_row_id, args.val_frac, args.test_frac, SEED)
    combined["strict_v2_split"] = combined.row_id.map(row_id_split)
    assert combined.strict_v2_split.isna().sum() == 0, "every row_id must map to a component"

    # Audit: zero identity overlap across splits (by (vendor,participant) id, not just component).
    id_by_split = {}
    for split in ["train", "val", "test"]:
        rids = row_id_split[row_id_split == split].index
        sub = identity[identity.row_id.isin(rids)]
        ids = set(zip(sub.p1_vendor_id, sub.p1_participant_id)) | set(zip(sub.p2_vendor_id, sub.p2_participant_id))
        id_by_split[split] = ids

    overlaps = {
        f"{a}_{b}": len(id_by_split[a] & id_by_split[b])
        for i, a in enumerate(["train", "val", "test"])
        for b in ["train", "val", "test"][i + 1:]
    }
    assert all(v == 0 for v in overlaps.values()), f"identity leakage across splits: {overlaps}"

    os.makedirs(OUT_DIR, exist_ok=True)
    counts = {}
    for split in ["train", "val", "test"]:
        df = combined[combined.strict_v2_split == split].drop(columns=["strict_v2_split"])
        df.to_csv(os.path.join(OUT_DIR, f"{split}.csv"), index=False)
        counts[split] = {
            "rows": len(df),
            "conversations": df.row_id.nunique(),
            "unique_identities": len(id_by_split[split]),
        }

    audit = {
        "n_components": int(len(sizes)),
        "largest_component_conversations": int(sizes.iloc[0]),
        "largest_component_frac": float(sizes.iloc[0] / len(component_of_row_id)),
        "val_frac_requested": args.val_frac,
        "test_frac_requested": args.test_frac,
        "split_counts": counts,
        "cross_split_identity_overlap": overlaps,
        "seed": SEED,
    }
    with open(os.path.join(OUT_DIR, "audit.json"), "w") as f:
        json.dump(audit, f, indent=2)

    print(json.dumps(audit, indent=2))
    print(f"\nWrote strict_v2 split to {OUT_DIR}/ (zero cross-split identity overlap, verified above)")


if __name__ == "__main__":
    main()
