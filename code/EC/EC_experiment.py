from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import networkx as nx
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import BadRequestError, InternalServerError, OpenAI
from openai import RateLimitError, APIConnectionError, APITimeoutError


# -----------------------------
# API helpers
# -----------------------------

def load_env(model: str = "gpt-5.2") -> tuple[str, str, str]:
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=str(env_path), override=True)
        print(f"Loaded env from: {env_path}")
    else:
        print("Using current environment variables.")

    required = ["OPENAI_ENDPOINT", "OPENAI_API_KEY"]
    missing = [v for v in required if not os.getenv(v)]
    if missing:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing)}")

    endpoint = os.getenv("OPENAI_ENDPOINT", "")
    api_key = os.getenv("OPENAI_API_KEY", "")
    return endpoint, api_key, model


def make_client(endpoint: str, api_key: str) -> OpenAI:
    return OpenAI(base_url=endpoint, api_key=api_key)


def call_model(client: OpenAI, model: str, prompt: str, temperature: float, top_p: float) -> str:
    max_attempts = 5
    base_delay_seconds = 1.0
    for attempt in range(1, max_attempts + 1):
        try:
            request: dict[str, Any] = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": temperature,
            }
            if not model.lower().startswith("claude-"):
                request["top_p"] = top_p
            resp = client.chat.completions.create(**request)
            return resp.choices[0].message.content or ""
        except (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError) as e:
            if attempt == max_attempts:
                raise
            delay_seconds = base_delay_seconds * (2 ** (attempt - 1)) + random.uniform(0.0, 0.5)
            print(f"Transient API error on attempt {attempt}/{max_attempts}: {e}. Retrying in {delay_seconds:.1f}s...")
            time.sleep(delay_seconds)
        except BadRequestError:
            raise

# -----------------------------
# Data structures
# -----------------------------

ACTIONS = ["POST", "REPOST", "REPLY", "SILENT"]


@dataclass
class Persona:
    agent_id: str
    bio: str
    follower_count: int
    ai_regulation_stance: int


@dataclass
class Post:
    post_id: str
    round_idx: int
    author_id: str
    author_stance: int
    action_type: str
    text: str
    target_post_id: Optional[str] = None


# -----------------------------
# Loading / preprocessing
# -----------------------------

def load_personas(path: str) -> list[Persona]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    personas = []
    for row in raw:
        personas.append(
            Persona(
                agent_id=row["agent_id"],
                bio=row["bio"],
                follower_count=int(row["follower_count"]),
                ai_regulation_stance=int(row["ai_regulation_stance"]),
            )
        )
    return personas


def side_label(stance: int) -> str:
    if stance <= -1:
        return "anti"
    if stance >= 1:
        return "pro"
    return "neutral"


def load_graph(path: str, personas: list[Persona]) -> nx.Graph:
    G = nx.read_edgelist(path)
    persona_ids = {p.agent_id for p in personas}
    G = nx.Graph(G)

    for pid in persona_ids:
        if pid not in G:
            G.add_node(pid)

    extra_nodes = [n for n in G.nodes if n not in persona_ids]
    if extra_nodes:
        raise ValueError(f"Edgelist contains unknown agent ids: {extra_nodes[:10]}")

    persona_map = {p.agent_id: p for p in personas}
    for pid in G.nodes:
        p = persona_map[pid]
        G.nodes[pid]["bio"] = p.bio
        G.nodes[pid]["follower_count"] = p.follower_count
        G.nodes[pid]["initial_stance"] = p.ai_regulation_stance
        G.nodes[pid]["current_stance"] = p.ai_regulation_stance
        G.nodes[pid]["side"] = side_label(p.ai_regulation_stance)
    return G


# -----------------------------
# Metrics
# -----------------------------

def exposure_diversity(exposures: list[dict[str, Any]], own_stance: int) -> float:
    if not exposures:
        return 0.0
    own_side = side_label(own_stance)
    different = sum(1 for x in exposures if side_label(int(x["author_stance"])) != own_side)
    return different / len(exposures)


def stance_polarization(values: list[int]) -> float:
    if len(values) < 2:
        return 0.0
    arr = np.array(values, dtype=float)
    diffs = np.abs(arr.reshape(-1, 1) - arr.reshape(1, -1))
    return float(diffs[np.triu_indices(len(arr), k=1)].mean())


def cross_group_interaction_ratio(interactions: list[dict[str, Any]], current_stances: dict[str, int]) -> float:
    relevant = [x for x in interactions if x["action"] in {"REPOST", "REPLY"} and x.get("target_author_id")]
    if not relevant:
        return 0.0
    cross = 0
    for x in relevant:
        a = side_label(current_stances[x["agent_id"]])
        b = side_label(current_stances[x["target_author_id"]])
        if a != b:
            cross += 1
    return cross / len(relevant)


def graph_side_assortativity(G: nx.Graph, current_stances: dict[str, int]) -> float:
    H = G.copy()
    for n in H.nodes:
        H.nodes[n]["side"] = side_label(current_stances[n])
    try:
        return float(nx.attribute_assortativity_coefficient(H, "side"))
    except Exception:
        return float("nan")


def top_hub_ids(G: nx.Graph, top_k: int = 10) -> list[str]:
    return [n for n, _ in sorted(G.degree, key=lambda x: x[1], reverse=True)[:top_k]]


def hub_metrics(
    G: nx.Graph,
    interactions_this_round: list[dict[str, Any]],
    recent_posts: list[Post],
    hub_ids: set[str],
) -> dict[str, float]:
    if not interactions_this_round:
        return {
            "hub_post_share": 0.0,
            "hub_interaction_target_share": 0.0,
            "hub_visible_post_share": 0.0,
        }

    non_silent = [r for r in interactions_this_round if r["action"] != "SILENT"]
    hub_posts = sum(1 for r in non_silent if r["agent_id"] in hub_ids)
    relevant_targets = [r for r in interactions_this_round if r["action"] in {"REPOST", "REPLY"} and r.get("target_author_id")]
    hub_target_hits = sum(1 for r in relevant_targets if r["target_author_id"] in hub_ids)
    visible_hub_posts = sum(1 for p in recent_posts if p.author_id in hub_ids)

    return {
        "hub_post_share": hub_posts / len(non_silent) if non_silent else 0.0,
        "hub_interaction_target_share": hub_target_hits / len(relevant_targets) if relevant_targets else 0.0,
        "hub_visible_post_share": visible_hub_posts / len(recent_posts) if recent_posts else 0.0,
    }


# -----------------------------
# Prompting
# -----------------------------

def build_feed_summary(feed_posts: list[dict[str, Any]]) -> str:
    if not feed_posts:
        return "No recent posts from your neighbors."

    lines = []
    for i, post in enumerate(feed_posts, start=1):
        target_note = f" | target_post_id={post['target_post_id']}" if post.get("target_post_id") else ""
        lines.append(
            f"[{i}] post_id={post['post_id']} | author={post['author_id']} | author_stance={post['author_stance']} | "
            f"action_type={post['action_type']}{target_note} | text={post['text']}"
        )
    return "\n".join(lines)


PERSONA_FORMATS = ("BULLET_ORIGINAL", "NARRATIVE_PROSE")

STANCE_CLAUSE = {
    -2: "strongly oppose stronger regulation",
    -1: "somewhat oppose stronger regulation",
    0: "hold a mixed or undecided view on regulation",
    1: "somewhat support stronger regulation",
    2: "strongly support stronger regulation",
}


def build_persona_block(persona_format: str, persona: Persona, current_stance: int, neighbor_ids: list[str]) -> str:
    if persona_format == "BULLET_ORIGINAL":
        return (
            "Your profile:\n"
            f"- agent_id: {persona.agent_id}\n"
            f"- bio: {persona.bio}\n"
            f"- follower_count: {persona.follower_count}\n"
            f"- current stance on AI regulation: {current_stance}  (where -2 strongly oppose, -1 somewhat oppose, "
            "0 mixed/undecided, 1 somewhat support, 2 strongly support)\n"
            f"- neighbor count in the network: {len(neighbor_ids)}\n"
        )
    elif persona_format == "NARRATIVE_PROSE":
        bio_clause = persona.bio[0].lower() + persona.bio[1:] if persona.bio else "a social media user"
        stance_clause = STANCE_CLAUSE.get(current_stance, "hold a mixed or undecided view on regulation")
        return (
            f"You are {persona.agent_id}, {bio_clause}. You have {persona.follower_count} followers, and "
            f"{len(neighbor_ids)} people in the network whose posts you can see. On the question of whether "
            f"governments should impose stronger regulation on advanced AI systems, you currently {stance_clause}.\n"
        )
    else:
        raise ValueError(f"Unknown persona_format: {persona_format}")


def build_prompt(
    persona: Persona,
    current_stance: int,
    neighbor_ids: list[str],
    feed_posts: list[dict[str, Any]],
    round_idx: int,
    total_rounds: int,
    persona_format: str = "BULLET_ORIGINAL",
) -> str:
    persona_block = build_persona_block(persona_format, persona, current_stance, neighbor_ids)
    return (
        "You are a social media user discussing one issue: whether governments should impose stronger regulation on advanced AI systems.\n\n"
        f"{persona_block}\n"
        "Simulation rules:\n"
        "- This is a fixed network for this run.\n"
        "- You only see recent posts from your neighbors.\n"
        "- Keep behavior simple and realistic.\n"
        "- Do not invent facts. Use short social-media-style language.\n"
        "- You may choose exactly one action this round: POST, REPOST, REPLY, or SILENT.\n"
        "- REPOST or REPLY should only target one visible post from the feed below.\n"
        "- Your stance is fixed for the entire run and must not change.\n"
        "- Set updated_stance equal to your current stance.\n\n"
        f"Round: {round_idx} / {total_rounds}\n\n"
        "Visible feed from your neighbors:\n"
        f"{build_feed_summary(feed_posts)}\n\n"
        "Return valid JSON with exactly these keys:\n"
        "{\n"
        '  "action": "POST|REPOST|REPLY|SILENT",\n'
        '  "target_index": integer or null,\n'
        '  "message": "short text, or empty string if SILENT",\n'
        '  "updated_stance": -2|-1|0|1|2,\n'
        '  "reason": "brief reason"\n'
        "}\n"
        "If target_index is used, it must refer to one visible feed item by its 1-based index."
    )

def clamp_stance(old: int, new: int) -> int:
    # frozen stance
    return old


def safe_parse_response(raw_text: str) -> tuple[dict[str, Any], bool, str]:
    default = {
        "action": "SILENT",
        "target_index": None,
        "message": "",
        "updated_stance": 0,
        "reason": "fallback",
    }

    try:
        obj = json.loads(raw_text)
        if isinstance(obj, dict):
            return obj, True, ""
    except Exception:
        pass

    m = re.search(r"\{[\s\S]*\}", raw_text)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj, True, ""
        except Exception:
            pass

    return default, False, "parse_failed"


def fallback_decision(persona: Persona, current_stance: int, feed_posts: list[dict[str, Any]], rng: random.Random) -> dict[str, Any]:
    if not feed_posts:
        return {
            "action": "POST",
            "target_index": None,
            "message": f"I think AI regulation should be approached carefully. stance={current_stance}",
            "updated_stance": current_stance,
            "reason": "No neighbor posts visible, so posted a short stance-consistent message.",
        }

    nearest_idx = min(
        range(len(feed_posts)),
        key=lambda i: abs(int(feed_posts[i]["author_stance"]) - current_stance),
    )
    action = "REPOST" if rng.random() < 0.5 else "REPLY"
    msg = (
        "Signal boosting this perspective on AI regulation."
        if action == "REPOST"
        else "I see the point here, though AI regulation needs balance."
    )
    return {
        "action": action,
        "target_index": nearest_idx + 1,
        "message": msg,
        "updated_stance": current_stance,
        "reason": "Fallback to nearest visible stance.",
    }


def choose_action_for_agent(
    persona: Persona,
    current_stance: int,
    neighbor_ids: list[str],
    feed_posts: list[dict[str, Any]],
    round_idx: int,
    total_rounds: int,
    endpoint: str,
    api_key: str,
    model: str,
    temperature: float,
    top_p: float,
    seed: int,
    persona_format: str = "BULLET_ORIGINAL",
) -> dict[str, Any]:
    rng = random.Random(seed)
    client = make_client(endpoint, api_key)
    prompt = build_prompt(persona, current_stance, neighbor_ids, feed_posts, round_idx, total_rounds, persona_format)
    raw = call_model(client, model, prompt, temperature, top_p)
    parsed, valid, note = safe_parse_response(raw)

    if not valid:
        parsed = fallback_decision(persona, current_stance, feed_posts, rng)

    action = str(parsed.get("action", "SILENT")).upper()
    if action not in ACTIONS:
        action = "SILENT"

    target_index = parsed.get("target_index", None)
    if target_index is not None:
        try:
            target_index = int(target_index)
        except Exception:
            target_index = None
    if target_index is not None and not (1 <= target_index <= len(feed_posts)):
        target_index = None

    message = str(parsed.get("message", ""))[:280]
    updated_stance = clamp_stance(current_stance, parsed.get("updated_stance", current_stance))
    reason = str(parsed.get("reason", ""))

    if action in {"REPOST", "REPLY"} and target_index is None:
        action = "POST" if feed_posts else "SILENT"
    if action == "SILENT":
        message = ""
        target_index = None

    return {
        "prompt": prompt,
        "raw_output": raw,
        "valid_parse": valid,
        "parse_note": note,
        "action": action,
        "target_index": target_index,
        "message": message,
        "updated_stance": updated_stance,
        "reason": reason,
    }


# -----------------------------
# Simulation
# -----------------------------

def collect_feed(
    G: nx.Graph,
    agent_id: str,
    recent_posts: list[Post],
    current_stances: dict[str, int],
    feed_size: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    neighbors = set(G.neighbors(agent_id))
    if not neighbors:
        return []

    candidates = [p for p in recent_posts if p.author_id in neighbors]
    if not candidates:
        return []

    candidates = sorted(candidates, key=lambda p: (p.round_idx, p.post_id), reverse=True)
    if len(candidates) > feed_size:
        sampled = candidates[: feed_size * 2]
        candidates = rng.sample(sampled, k=feed_size)
        candidates = sorted(candidates, key=lambda p: (p.round_idx, p.post_id), reverse=True)
    else:
        candidates = candidates[:feed_size]

    return [
        {
            "post_id": p.post_id,
            "round_idx": p.round_idx,
            "author_id": p.author_id,
            "author_stance": current_stances[p.author_id],
            "action_type": p.action_type,
            "text": p.text,
            "target_post_id": p.target_post_id,
        }
        for p in candidates
    ]


def infer_graph_condition(edgelist_path: str) -> dict[str, str]:
    p = str(edgelist_path).replace("\\", "/").lower()
    out = {
        "network_family": "unknown",
        "assortativity_condition": "unknown",
        "hub_condition": "unknown",
    }
    if "homophily" in p or "powerlaw" in p:
        out["network_family"] = "powerlaw_fixed_degree"
    if "low_assortativity" in p:
        out["assortativity_condition"] = "low"
    elif "medium_assortativity" in p:
        out["assortativity_condition"] = "medium"
    elif "high_assortativity" in p:
        out["assortativity_condition"] = "high"
    elif "slightly_higher_assortativity" in p:
        out["assortativity_condition"] = "slightly_higher"

    for key in ["random", "mixed_hubs", "pro_hubs", "anti_hubs", "follower_hubs"]:
        if key in p:
            out["hub_condition"] = key
            break
    return out


def run_simulation(
    personas_path: str,
    edgelist_path: str,
    rounds: int,
    activation_prob: float,
    feed_size: int,
    memory_rounds: int,
    workers: int,
    seed: int,
    temperature: float,
    top_p: float,
    out_dir: str,
    persona_format: str = "BULLET_ORIGINAL",
    model: str = "gpt-5.2",
) -> None:
    endpoint, api_key, model = load_env(model)
    personas = load_personas(personas_path)
    G = load_graph(edgelist_path, personas)
    persona_map = {p.agent_id: p for p in personas}

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    current_stances = {p.agent_id: p.ai_regulation_stance for p in personas}
    posts: list[Post] = []
    round_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    post_counter = 0

    hubs_top10 = top_hub_ids(G, top_k=max(1, math.ceil(0.10 * G.number_of_nodes())))
    hub_set = set(hubs_top10)
    graph_meta = infer_graph_condition(edgelist_path)

    print(f"Using model={model}")
    print(f"Nodes={G.number_of_nodes()} edges={G.number_of_edges()}")

    for round_idx in range(1, rounds + 1):
        active_agents = [p.agent_id for p in personas if rng.random() < activation_prob]
        if not active_agents:
            active_agents = [rng.choice(personas).agent_id]

        recent_posts = [p for p in posts if p.round_idx >= max(1, round_idx - memory_rounds)]

        future_map = {}
        decisions: dict[str, dict[str, Any]] = {}

        with ThreadPoolExecutor(max_workers=workers) as executor:
            for agent_id in active_agents:
                local_seed = seed * 100000 + round_idx * 1000 + int(agent_id[1:])
                feed = collect_feed(
                    G=G,
                    agent_id=agent_id,
                    recent_posts=recent_posts,
                    current_stances=current_stances,
                    feed_size=feed_size,
                    rng=random.Random(local_seed),
                )
                neighbor_ids = list(G.neighbors(agent_id))
                fut = executor.submit(
                    choose_action_for_agent,
                    persona_map[agent_id],
                    current_stances[agent_id],
                    neighbor_ids,
                    feed,
                    round_idx,
                    rounds,
                    endpoint,
                    api_key,
                    model,
                    temperature,
                    top_p,
                    local_seed,
                    persona_format,
                )
                future_map[fut] = (agent_id, feed)

            for fut in as_completed(future_map):
                agent_id, feed = future_map[fut]
                try:
                    decisions[agent_id] = fut.result() | {"feed": feed}
                except Exception as e:
                    print(f"Decision failed for {agent_id} in round {round_idx}: {e}")
                    print(traceback.format_exc())
                    decisions[agent_id] = fallback_decision(
                        persona_map[agent_id],
                        current_stances[agent_id],
                        feed,
                        random.Random(seed + round_idx),
                    ) | {
                        "prompt": "",
                        "raw_output": "",
                        "valid_parse": False,
                        "parse_note": f"exception:{e}",
                        "feed": feed,
                    }

        interactions_this_round: list[dict[str, Any]] = []
        for agent_id in active_agents:
            d = decisions[agent_id]
            target_post_id = None
            target_author_id = None
            if d.get("target_index") is not None:
                target = d["feed"][d["target_index"] - 1]
                target_post_id = target["post_id"]
                target_author_id = target["author_id"]

            if d["action"] != "SILENT":
                post_counter += 1
                new_post = Post(
                    post_id=f"P{post_counter:06d}",
                    round_idx=round_idx,
                    author_id=agent_id,
                    author_stance=current_stances[agent_id],
                    action_type=d["action"],
                    text=d["message"],
                    target_post_id=target_post_id,
                )
                posts.append(new_post)

            exposure_score = exposure_diversity(d["feed"], current_stances[agent_id])

            row = {
                "round": round_idx,
                "agent_id": agent_id,
                "bio": persona_map[agent_id].bio,
                "follower_count": persona_map[agent_id].follower_count,
                "degree": G.degree(agent_id),
                "is_top_hub": agent_id in hub_set,
                "old_stance": current_stances[agent_id],
                "new_stance": current_stances[agent_id],
                "action": d["action"],
                "target_post_id": target_post_id,
                "target_author_id": target_author_id,
                "target_is_top_hub": target_author_id in hub_set if target_author_id else False,
                "message": d["message"],
                "reason": d["reason"],
                "n_visible_posts": len(d["feed"]),
                "exposure_diversity": exposure_score,
                "valid_parse": d["valid_parse"],
                "parse_note": d["parse_note"],
                "agent_prompt_input": d["prompt"],
                "agent_raw_output": d["raw_output"],
                "persona_format": persona_format,
                "seed": seed,
            }
            round_rows.append(row)
            interactions_this_round.append(row)

        polarization = stance_polarization(list(current_stances.values()))
        assortativity = graph_side_assortativity(G, current_stances)
        cross_ratio = cross_group_interaction_ratio(interactions_this_round, current_stances)
        avg_exposure_div = float(np.mean([r["exposure_diversity"] for r in interactions_this_round])) if interactions_this_round else 0.0
        active_rate = float(np.mean([r["action"] != "SILENT" for r in interactions_this_round])) if interactions_this_round else 0.0
        hm = hub_metrics(G, interactions_this_round, recent_posts, hub_set)

        summary_rows.append(
            {
                "round": round_idx,
                "n_active_agents": len(active_agents),
                "n_posts_created": sum(1 for r in interactions_this_round if r["action"] != "SILENT"),
                "cross_group_interaction_ratio": cross_ratio,
                "avg_exposure_diversity": avg_exposure_div,
                "graph_side_assortativity": assortativity,
                "stance_polarization": polarization,
                "active_rate": active_rate,
                "hub_post_share": hm["hub_post_share"],
                "hub_interaction_target_share": hm["hub_interaction_target_share"],
                "hub_visible_post_share": hm["hub_visible_post_share"],
            }
        )
        print(
            f"Round {round_idx:02d}: active={len(active_agents):3d} posts={summary_rows[-1]['n_posts_created']:3d} "
            f"cross_interact={cross_ratio:.3f} exposure_div={avg_exposure_div:.3f} "
            f"assort={assortativity:.3f} polarization={polarization:.3f} "
            f"hub_post_share={hm['hub_post_share']:.3f} hub_target_share={hm['hub_interaction_target_share']:.3f}"
        )

    pd.DataFrame(round_rows).to_csv(out_path / "agent_rounds.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(out_path / "round_summary.csv", index=False)

    pd.DataFrame([{
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "mean_degree": float(np.mean([d for _, d in G.degree()])),
        "top_hub_ids": ",".join(hubs_top10),
        "top_hub_threshold_count": len(hubs_top10),
        "final_graph_side_assortativity": graph_side_assortativity(G, current_stances),
        "final_stance_polarization": stance_polarization(list(current_stances.values())),
        "rounds": rounds,
        "activation_prob": activation_prob,
        "feed_size": feed_size,
        "memory_rounds": memory_rounds,
        "seed": seed,
        "model": model,
        "persona_format": persona_format,
        "temperature": temperature,
        "top_p": None if model.lower().startswith("claude-") else top_p,
        "workers": workers,
        "edgelist": str(edgelist_path),
        "personas": str(personas_path),
        "stance_frozen": True,
        **graph_meta,
    }]).to_csv(out_path / "run_metadata.csv", index=False)

    final_state = []
    for p in personas:
        final_state.append({
            "agent_id": p.agent_id,
            "bio": p.bio,
            "follower_count": p.follower_count,
            "initial_stance": p.ai_regulation_stance,
            "final_stance": current_stances[p.agent_id],
            "degree": G.degree(p.agent_id),
            "is_top_hub": p.agent_id in hub_set,
        })
    pd.DataFrame(final_state).to_csv(out_path / "final_agent_state.csv", index=False)
    print(f"Saved results to {out_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Echo-chamber simulation on a fixed network with frozen stance")
    parser.add_argument("--personas", type=str, default=str(Path(__file__).resolve().parent / "inputs/ai_regulation_minimal_personas_100.json"), help="Path to persona JSON (default: bundled EC input)")
    parser.add_argument("--edgelist", type=str, default=str(Path(__file__).resolve().parent / "inputs/networks/homophily/medium_assortativity/medium_assortativity_01.edgelist"), help="Path to one edgelist (default: bundled medium graph)")
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--activation-prob", type=float, default=0.3)
    parser.add_argument("--feed-size", type=int, default=5)
    parser.add_argument("--memory-rounds", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--out-dir", type=str, default=str(Path(__file__).resolve().parents[2] / "data/generated/EC/single_run"))
    parser.add_argument("--model", default="gpt-5.2", help="Provider model/deployment ID")
    parser.add_argument("--persona-format", choices=PERSONA_FORMATS, default="BULLET_ORIGINAL")
    parser.add_argument("--dry-run", action="store_true", help="Print a round-one prompt without API calls")
    args = parser.parse_args()
    if not 0 <= args.activation_prob <= 1:
        parser.error("--activation-prob must be between 0 and 1")
    if min(args.rounds, args.feed_size, args.memory_rounds, args.workers) < 1:
        parser.error("rounds, feed size, memory rounds and workers must be positive")
    if args.dry_run:
        personas = load_personas(args.personas)
        graph = load_graph(args.edgelist, personas)
        persona = personas[0]
        print(build_prompt(persona, persona.ai_regulation_stance, list(graph.neighbors(persona.agent_id)),
                           [], 1, args.rounds, args.persona_format))
        return

    run_simulation(
        personas_path=args.personas,
        edgelist_path=args.edgelist,
        rounds=args.rounds,
        activation_prob=args.activation_prob,
        feed_size=args.feed_size,
        memory_rounds=args.memory_rounds,
        workers=args.workers,
        seed=args.seed,
        temperature=args.temperature,
        top_p=args.top_p,
        out_dir=args.out_dir,
        persona_format=args.persona_format,
        model=args.model,
    )


if __name__ == "__main__":
    main()
