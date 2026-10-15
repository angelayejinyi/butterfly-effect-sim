import argparse
import csv
import json
import math
import random
from pathlib import Path

import networkx as nx
import numpy as np


# ============================================================
# Load personas
# ============================================================

def load_agents(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# Basic helpers
# ============================================================

def side_label(stance):
    if stance <= -1:
        return "anti"
    elif stance >= 1:
        return "pro"
    return "neutral"


def add_node_attributes(G, agents):
    for a in agents:
        G.add_node(
            a["agent_id"],
            bio=a["bio"],
            follower_count=a["follower_count"],
            stance=a["ai_regulation_stance"],
            side=side_label(a["ai_regulation_stance"]),
        )


def graph_stats(G):
    stats = {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "density": nx.density(G),
        "avg_degree": float(np.mean([d for _, d in G.degree()])),
        "max_degree": max((d for _, d in G.degree()), default=0),
    }
    try:
        stats["stance_assortativity"] = float(nx.numeric_assortativity_coefficient(G, "stance"))
    except Exception:
        stats["stance_assortativity"] = None
    try:
        stats["side_assortativity"] = float(nx.attribute_assortativity_coefficient(G, "side"))
    except Exception:
        stats["side_assortativity"] = None

    cross = 0
    m = G.number_of_edges()
    if m > 0:
        for u, v in G.edges():
            if G.nodes[u]["side"] != G.nodes[v]["side"]:
                cross += 1
        stats["cross_group_edge_ratio"] = cross / m
    else:
        stats["cross_group_edge_ratio"] = 0.0
    return stats


def save_graph_edgelist(G, path):
    nx.write_edgelist(G, path, data=False)


def save_summary_csv(rows, path):
    fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# Fixed power-law degree sequence
# ============================================================

def make_powerlaw_degree_sequence(n, target_mean_degree=8, exponent=2.4, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(1000):
        raw = rng.zipf(exponent, size=n)
        raw = np.clip(raw, 1, n - 1)

        current_mean = raw.mean()
        scaled = np.maximum(1, np.round(raw * (target_mean_degree / current_mean))).astype(int)
        scaled = np.clip(scaled, 1, n - 1)

        if scaled.sum() % 2 == 1:
            idx = int(np.argmax(scaled))
            scaled[idx] = max(1, scaled[idx] - 1)

        seq = sorted(scaled.tolist(), reverse=True)
        if nx.is_graphical(seq, method="eg"):
            return seq
    raise RuntimeError("Failed to generate a graphical power-law-like degree sequence.")


def assign_degree_sequence_to_agents(agents, degree_sequence, mode="mixed_hubs", seed=0):
    """
    Assign the same fixed degree sequence to agents in different ways.

    modes:
      - random
      - mixed_hubs
      - pro_hubs
      - anti_hubs
      - follower_hubs
    """
    rng = random.Random(seed)
    agents_copy = agents[:]

    if mode == "random":
        rng.shuffle(agents_copy)

    elif mode == "follower_hubs":
        agents_copy.sort(key=lambda a: math.log1p(a["follower_count"]), reverse=True)

    elif mode == "pro_hubs":
        agents_copy.sort(
            key=lambda a: (
                a["ai_regulation_stance"] >= 1,
                a["ai_regulation_stance"],
                math.log1p(a["follower_count"]),
                rng.random(),
            ),
            reverse=True,
        )

    elif mode == "anti_hubs":
        agents_copy.sort(
            key=lambda a: (
                a["ai_regulation_stance"] <= -1,
                -a["ai_regulation_stance"],
                math.log1p(a["follower_count"]),
                rng.random(),
            ),
            reverse=True,
        )

    elif mode == "mixed_hubs":
        neg = [a for a in agents_copy if a["ai_regulation_stance"] <= -1]
        neu = [a for a in agents_copy if a["ai_regulation_stance"] == 0]
        pos = [a for a in agents_copy if a["ai_regulation_stance"] >= 1]

        for group in (neg, neu, pos):
            group.sort(key=lambda a: math.log1p(a["follower_count"]), reverse=True)

        interleaved = []
        while neg or neu or pos:
            if neg:
                interleaved.append(neg.pop(0))
            if neu:
                interleaved.append(neu.pop(0))
            if pos:
                interleaved.append(pos.pop(0))
        agents_copy = interleaved

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return {a["agent_id"]: d for a, d in zip(agents_copy, degree_sequence)}


def configuration_model_from_assignment(agents, degree_assignment, seed=0):
    """
    Create a simple graph with exactly the assigned degree sequence.

    Use Havel-Hakimi instead of configuration_model + simplification.
    This guarantees an exact simple realization whenever the degree sequence
    is graphical, which we already enforce upstream.
    """
    ids = [a["agent_id"] for a in agents]
    deg_seq = [degree_assignment[aid] for aid in ids]

    if not nx.is_graphical(deg_seq, method="eg"):
        raise RuntimeError("Degree sequence is not graphical.")

    H = nx.havel_hakimi_graph(deg_seq)
    mapping = {i: ids[i] for i in range(len(ids))}
    G = nx.relabel_nodes(H, mapping)
    G = nx.Graph(G)
    add_node_attributes(G, agents)

    if not all(G.degree(aid) == degree_assignment[aid] for aid in ids):
        raise RuntimeError("Failed to realize exact simple graph from fixed degree assignment.")

    return G


# ============================================================
# Degree-preserving rewiring to change assortativity
# ============================================================

def edge_similarity(G, u, v):
    su = G.nodes[u]["stance"]
    sv = G.nodes[v]["stance"]
    return 1.0 - abs(su - sv) / 4.0


def current_assortativity(G):
    try:
        return float(nx.numeric_assortativity_coefficient(G, "stance"))
    except Exception:
        return float("nan")


def degree_preserving_rewire_to_band(
    G,
    target_min,
    target_max,
    seed=0,
    max_steps=100000,
):
    """
    Rewire edges using double-edge swaps while keeping every node's degree fixed.
    The algorithm heuristically pushes the graph toward a desired assortativity band.
    """
    rng = random.Random(seed)
    H = G.copy()

    assort = current_assortativity(H)
    if assort == assort and target_min <= assort <= target_max:
        return H, assort, 0

    target_mid = 0.5 * (target_min + target_max)

    for step in range(max_steps):
        edges = list(H.edges())
        if len(edges) < 2:
            break

        (a, b) = rng.choice(edges)
        (c, d) = rng.choice(edges)

        if len({a, b, c, d}) < 4:
            continue

        candidate_pairings = [((a, c), (b, d)), ((a, d), (b, c))]
        rng.shuffle(candidate_pairings)

        moved = False
        for (u1, v1), (u2, v2) in candidate_pairings:
            if u1 == v1 or u2 == v2:
                continue
            if H.has_edge(u1, v1) or H.has_edge(u2, v2):
                continue

            old_score = edge_similarity(H, a, b) + edge_similarity(H, c, d)
            new_score = edge_similarity(H, u1, v1) + edge_similarity(H, u2, v2)

            assort_now = current_assortativity(H)
            if not (assort_now == assort_now):
                assort_now = -999.0

            want_higher = assort_now < target_mid
            improve = new_score > old_score if want_higher else new_score < old_score

            if improve:
                H.remove_edge(a, b)
                H.remove_edge(c, d)
                H.add_edge(u1, v1)
                H.add_edge(u2, v2)
                moved = True
                break

        if moved and step % 250 == 0:
            assort = current_assortativity(H)
            if assort == assort and target_min <= assort <= target_max:
                return H, assort, step + 1

    assort = current_assortativity(H)
    return H, assort, max_steps


# ============================================================
# Homophily experiment: same power-law degree sequence, fixed hub assignment,
# vary assortativity by rewiring
# ============================================================

def generate_homophily(
    agents,
    out_dir,
    n_per_condition=5,
    target_mean_degree=8,
    exponent=2.4,
    hub_mode="mixed_hubs",
    seed=42,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    degree_sequence = make_powerlaw_degree_sequence(
        n=len(agents),
        target_mean_degree=target_mean_degree,
        exponent=exponent,
        seed=seed,
    )

    degree_assignment = assign_degree_sequence_to_agents(
        agents,
        degree_sequence,
        mode=hub_mode,
        seed=seed,
    )

    base_graph = configuration_model_from_assignment(
        agents,
        degree_assignment,
        seed=seed,
    )

    conditions = {
        "low_assortativity": (0.02, 0.10),
        "medium_assortativity": (0.12, 0.20),
        "paper_high_assortativity": (0.22, 0.30),
    }

    rows = []

    with open(out_dir / "degree_sequence.json", "w", encoding="utf-8") as f:
        json.dump(degree_sequence, f, indent=2)
    with open(out_dir / "degree_assignment.json", "w", encoding="utf-8") as f:
        json.dump(dict(sorted(degree_assignment.items(), key=lambda x: x[1], reverse=True)), f, indent=2)

    for cond_name, (amin, amax) in conditions.items():
        cond_dir = out_dir / cond_name
        cond_dir.mkdir(parents=True, exist_ok=True)

        for k in range(1, n_per_condition + 1):
            local_seed = seed + 1000 * k + 10000 * list(conditions.keys()).index(cond_name)
            G, assort, n_steps = degree_preserving_rewire_to_band(
                base_graph,
                target_min=amin,
                target_max=amax,
                seed=local_seed,
                max_steps=50000,
            )

            graph_name = f"{cond_name}_{k:02d}"
            save_graph_edgelist(G, cond_dir / f"{graph_name}.edgelist")

            stats = graph_stats(G)
            row = {
                "study": "homophily",
                "hub_mode": hub_mode,
                "condition": cond_name,
                "graph_name": graph_name,
                "seed": local_seed,
                "rewire_steps": n_steps,
                **stats,
            }
            rows.append(row)

    save_summary_csv(rows, out_dir / "graph_summary.csv")
    print(f"Saved Homophily experiment power-law assortativity graphs to: {out_dir.resolve()}")


# ============================================================
# Hub-assignment experiment: same power-law degree sequence, same assortativity band,
# vary hub assignment
# ============================================================

def generate_hub_assignment(
    agents,
    out_dir,
    n_per_condition=5,
    target_mean_degree=8,
    exponent=2.4,
    assort_band=(0.12, 0.20),
    seed=42,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    degree_sequence = make_powerlaw_degree_sequence(
        n=len(agents),
        target_mean_degree=target_mean_degree,
        exponent=exponent,
        seed=seed,
    )

    hub_modes = ["random", "mixed_hubs", "pro_hubs", "anti_hubs", "follower_hubs"]
    rows = []

    with open(out_dir / "degree_sequence.json", "w", encoding="utf-8") as f:
        json.dump(degree_sequence, f, indent=2)

    for mode in hub_modes:
        mode_dir = out_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)

        for k in range(1, n_per_condition + 1):
            local_seed = seed + 1000 * k + 10000 * hub_modes.index(mode)

            degree_assignment = assign_degree_sequence_to_agents(
                agents,
                degree_sequence,
                mode=mode,
                seed=local_seed,
            )

            with open(mode_dir / f"{mode}_{k:02d}_degree_assignment.json", "w", encoding="utf-8") as f:
                json.dump(dict(sorted(degree_assignment.items(), key=lambda x: x[1], reverse=True)), f, indent=2)

            G0 = configuration_model_from_assignment(
                agents,
                degree_assignment,
                seed=local_seed,
            )

            G, assort, n_steps = degree_preserving_rewire_to_band(
                G0,
                target_min=assort_band[0],
                target_max=assort_band[1],
                seed=local_seed,
                max_steps=50000,
            )

            graph_name = f"{mode}_{k:02d}"
            save_graph_edgelist(G, mode_dir / f"{graph_name}.edgelist")

            stats = graph_stats(G)
            row = {
                "study": "hub_assignment",
                "condition": mode,
                "graph_name": graph_name,
                "seed": local_seed,
                "rewire_steps": n_steps,
                **stats,
            }
            rows.append(row)

    save_summary_csv(rows, out_dir / "graph_summary.csv")
    print(f"Saved Hub-assignment experiment power-law hub graphs to: {out_dir.resolve()}")


# ============================================================
# Demo: rewire one Hub-assignment experiment graph into different assortativity variants
# ============================================================

def rewire_one_hub_graph_to_multiple_assortativity_levels(
    agents,
    out_dir,
    hub_mode="mixed_hubs",
    target_mean_degree=8,
    exponent=2.4,
    seed=42,
):
    """
    Create one fixed hub-assignment graph, then rewire it into low / medium /
    slightly higher assortativity variants while keeping the same degree sequence
    and same hub assignment.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    degree_sequence = make_powerlaw_degree_sequence(
        n=len(agents),
        target_mean_degree=target_mean_degree,
        exponent=exponent,
        seed=seed,
    )
    degree_assignment = assign_degree_sequence_to_agents(
        agents,
        degree_sequence,
        mode=hub_mode,
        seed=seed,
    )
    base_graph = configuration_model_from_assignment(
        agents,
        degree_assignment,
        seed=seed,
    )

    with open(out_dir / "degree_sequence.json", "w", encoding="utf-8") as f:
        json.dump(degree_sequence, f, indent=2)
    with open(out_dir / "degree_assignment.json", "w", encoding="utf-8") as f:
        json.dump(dict(sorted(degree_assignment.items(), key=lambda x: x[1], reverse=True)), f, indent=2)

    bands = {
        "low_assortativity": (0.02, 0.10),
        "medium_assortativity": (0.12, 0.20),
        "paper_high_assortativity": (0.22, 0.30),
    }

    rows = []
    for idx, (name, (amin, amax)) in enumerate(bands.items(), start=1):
        G, assort, n_steps = degree_preserving_rewire_to_band(
            base_graph,
            target_min=amin,
            target_max=amax,
            seed=seed + 1000 * idx,
            max_steps=60000,
        )
        save_graph_edgelist(G, out_dir / f"{name}.edgelist")
        stats = graph_stats(G)
        rows.append({
            "study": "demo_rewire_one_hub_graph",
            "hub_mode": hub_mode,
            "condition": name,
            "seed": seed + 1000 * idx,
            "rewire_steps": n_steps,
            **stats,
        })

    save_summary_csv(rows, out_dir / "graph_summary.csv")
    print(f"Saved rewired variants of one hub graph to: {out_dir.resolve()}")


# ============================================================
# Main
# ============================================================

def main():
    ec_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Generate the synthetic networks used by the echo-chamber study.")
    parser.add_argument(
        "--personas",
        type=Path,
        default=Path(__file__).resolve().parent / "inputs" / "ai_regulation_minimal_personas_100.json",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ec_root.parents[1] / "data/generated/networks",
    )
    args = parser.parse_args()
    agents = load_agents(args.personas)

    # Homophily experiment: same power-law family, fixed hub assignment, vary assortativity
    generate_homophily(
        agents=agents,
        out_dir=args.out_root / "homophily",
        n_per_condition=5,
        target_mean_degree=8,
        exponent=2.4,
        hub_mode="mixed_hubs",
        seed=42,
    )

    # Hub-assignment experiment: same power-law family, same assortativity band, vary hub assignment
    generate_hub_assignment(
        agents=agents,
        out_dir=args.out_root / "hub_assignment",
        n_per_condition=5,
        target_mean_degree=8,
        exponent=2.4,
        assort_band=(0.12, 0.20),
        seed=42,
    )

    # Optional demo: take one hub graph and rewire it to different assortativity levels
    rewire_one_hub_graph_to_multiple_assortativity_levels(
        agents=agents,
        out_dir=args.out_root / "rewired_one_hub_graph_variants",
        hub_mode="mixed_hubs",
        target_mean_degree=8,
        exponent=2.4,
        seed=42,
    )


if __name__ == "__main__":
    main()
