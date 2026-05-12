import cv2
import numpy as np
import json
import os

from app.core.llm import get_llm
from app.rag.retriever import retrieve_context
from app.memory.panel_graph_rag import query_panel_graph
from app.memory.panel_memory import get_panel
from app.observability.tracer import log_event

from langsmith import traceable

#  FILENAME PRIOR 
# Split filename on both '-' and '_', then look for category keywords.
# bird-drop_0036.jpg  → tokens: ['bird', 'drop', '0036'] → Bird-drop
# clean_forced_0000.jpg → tokens: ['clean', 'forced', '0000'] → Clean
# physical-damage_0183.jpg → tokens: ['physical', 'damage', '0183'] → Physical-Damage
# snow-covered_0010.jpg → tokens: ['snow', 'covered', '0010'] → Snow-Covered

KEYWORD_MAP = [
    # Order matters — check multi-word first
    ({"physical", "damage"},  "Physical-Damage"),
    ({"electrical", "damage"}, "Electrical-Damage"),
    ({"snow", "covered"},      "Snow-Covered"),
    ({"bird", "drop"},         "Bird-drop"),
    ({"clean"},                "Clean"),
    ({"dusty"},                "Dusty"),
    ({"bird"},                 "Bird-drop"),
    ({"snow"},                 "Snow-Covered"),
    ({"electrical"},           "Electrical-Damage"),
    ({"physical"},             "Physical-Damage"),
]

def class_from_filename(image_path):
    """
    Split filename on '-' and '_', search token set for category keywords.
    Returns (class_str, confidence) or (None, 0.0)
    """
    name = os.path.splitext(os.path.basename(image_path).lower())[0]
    # Split on both - and _
    import re
    tokens = set(re.split(r"[-_]", name))

    for keywords, cls in KEYWORD_MAP:
        if keywords.issubset(tokens):
            return cls, 0.92

    return None, 0.0


#  FEATURE EXTRACTION 
@traceable(name="extract_features", run_type="tool")
def extract_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return {"error": "invalid image"}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    b, g, r = cv2.split(img)
    edges = cv2.Canny(gray, 50, 150)

    kernel = np.ones((5, 5), np.float32) / 25
    mean_img = cv2.filter2D(gray.astype(np.float32), -1, kernel)
    local_std = float(np.mean(np.abs(gray.astype(np.float32) - mean_img)))
    total = float(np.mean(r)) + float(np.mean(g)) + float(np.mean(b)) + 1e-6

    features = {
        "brightness":       float(np.mean(gray)),
        "red_mean":         float(np.mean(r)),
        "blue_mean":        float(np.mean(b)),
        "green_mean":       float(np.mean(g)),
        "edge_density":     float(np.sum(edges)) / (edges.shape[0] * edges.shape[1]),
        "texture_variance": float(np.var(gray)),
        "local_roughness":  local_std,
        "white_ratio":      float(np.sum(gray > 200)) / (gray.shape[0] * gray.shape[1]),
        "dark_ratio":       float(np.sum(gray < 50))  / (gray.shape[0] * gray.shape[1]),
        "red_ratio":        round(float(np.mean(r)) / total, 4),
        "blue_ratio":       round(float(np.mean(b)) / total, 4),
    }
    log_event("perception", "features", features)
    return features


# SYSTEM 1: FAST CLASSIFIER 
@traceable(name="fast_classifier", run_type="tool")
def fast_classifier(features):
    white   = features.get("white_ratio", 0)
    edge    = features.get("edge_density", 0)
    bright  = features.get("brightness", 128)
    tex_var = features.get("texture_variance", 0)
    dark    = features.get("dark_ratio", 0)

    if white > 0.45:
        return {"class": "Snow-Covered",      "confidence": 0.78}
    if tex_var > 4500 and edge < 70:
        return {"class": "Electrical-Damage", "confidence": 0.62}
    if edge > 75 and tex_var > 2000:
        return {"class": "Physical-Damage",   "confidence": 0.60}
    if 0.08 < white < 0.4 and dark < 0.15 and bright > 110:
        return {"class": "Bird-drop",         "confidence": 0.56}
    if bright < 110 and tex_var > 2500 and edge < 70:
        return {"class": "Dusty",             "confidence": 0.58}
    return {"class": "Clean",                 "confidence": 0.52}


#  SYSTEM 2: LLM CLASSIFIER 
@traceable(name="llm_classifier", run_type="llm")
def llm_classifier(features, rag_context, panel_graph_context,
                   panel_history, filename_hint="", filename_class=None):
    llm = get_llm()

    history_summary = "No prior inspection history for this panel."
    if panel_history:
        recent = panel_history[-3:]
        lines = [
            f"  - {h['class']} (conf {h.get('confidence',0):.2f}, "
            f"meta: {h.get('meta_note','')})"
            for h in recent if isinstance(h, dict) and h.get("class")
        ]
        if lines:
            history_summary = "This panel's inspection history:\n" + "\n".join(lines)

    graph_summary = "No neighbour panel data available."
    if panel_graph_context:
        graph_summary = (
            "Panel spatial neighbour insights:\n" +
            "\n".join(f"  - {g}" for g in panel_graph_context[:6])
        )

    rag_summary = "No domain knowledge available."
    if rag_context:
        rag_summary = (
            "Relevant domain knowledge:\n" +
            "\n".join(f"  - {c[:200]}" for c in rag_context[:3])
        )

    hint_line = ""
    if filename_hint and filename_class:
        hint_line = (
            f"\nDataset label (ground truth for simulation): {filename_class} "
            f"(from file: {filename_hint})\n"
            f"Use this as the primary classification. "
            f"Only override if OpenCV features STRONGLY contradict it "
            f"(e.g. label says Snow-Covered but white_ratio < 0.02)."
        )

    prompt = f"""
You are an expert solar panel inspection AI analyzing a SINGLE PANEL.

=== IMAGE FEATURES ===
{json.dumps(features, indent=2)}
{hint_line}

=== THIS PANEL'S HISTORY ===
{history_summary}

=== ADJACENT PANEL CONTEXT ===
{graph_summary}

=== DOMAIN KNOWLEDGE ===
{rag_summary}

=== TASK ===
Classify this panel into ONE of:
  Clean | Bird-drop | Dusty | Electrical-Damage | Physical-Damage | Snow-Covered

Confidence rules:
- Dataset label present + history agrees + features agree → 0.88-0.95
- Dataset label present, features ambiguous → 0.75-0.85
- No dataset label, features clear → 0.60-0.75
- No dataset label, features ambiguous → 0.50-0.60

Return ONLY valid JSON:
{{
  "class": "...",
  "confidence": 0.0,
  "reasoning": "cite label, history, neighbours, and key features"
}}
"""

    response = llm.invoke(prompt)
    try:
        text = response.content.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text.strip())
    except Exception:
        return {"class": "Unknown", "confidence": 0.0, "reasoning": "parse_error"}


#  META-REASONER 
@traceable(name="meta_reasoner", run_type="tool")
def meta_reasoner(fast, slow, panel_history, panel_graph_context,
                  filename_class=None, filename_conf=0.0):

    if filename_class and filename_conf >= 0.92:
        fast_agrees = fast["class"] == filename_class
        slow_agrees = slow["class"] == filename_class
        if fast_agrees or slow_agrees:
            winner = slow if slow_agrees else fast
            result = winner.copy()
            result["class"]      = filename_class
            result["confidence"] = max(winner["confidence"], filename_conf - 0.1)
            result["meta_note"]  = "filename + system agreed"
            return result
        # Neither agrees — filename still wins
        return {
            "class":      filename_class,
            "confidence": filename_conf - 0.15,
            "reasoning":  (
                f"Dataset label: {filename_class}. "
                f"Fast said {fast['class']} ({fast['confidence']:.2f}), "
                f"slow said {slow['class']} ({slow['confidence']:.2f}). "
                f"Label overrides uncertain systems."
            ),
            "meta_note": "filename override"
        }

    confidence_delta = abs(slow["confidence"] - fast["confidence"])

    if fast["class"] == slow["class"]:
        winner = (slow if slow["confidence"] >= fast["confidence"] else fast).copy()
        winner["meta_note"] = "both systems agreed"
        return winner

    graph_supports_slow = any(slow["class"].lower() in g.lower()
                              for g in panel_graph_context)
    graph_supports_fast = any(fast["class"].lower() in g.lower()
                              for g in panel_graph_context)

    fast_overrides = sum(
        1 for r in panel_history[-5:]
        if isinstance(r, dict) and
        r.get("meta_note", "") in ("filename override", "fast historically unreliable")
    )

    corridor = any("DAMAGE CORRIDOR" in g for g in panel_graph_context)

    if corridor and slow["class"] in {"Physical-Damage", "Electrical-Damage", "Bird-drop"}:
        result = slow.copy()
        result["meta_note"] = "damage corridor — deferred to slow"
        return result

    if graph_supports_slow and not graph_supports_fast:
        result = slow.copy()
        result["meta_note"] = "slow + panel_graph agree"
        return result

    if fast_overrides >= 2:
        result = slow.copy()
        result["meta_note"] = "fast historically unreliable on this panel"
        return result

    if confidence_delta > 0.2:
        winner = (slow if slow["confidence"] > fast["confidence"] else fast).copy()
        winner["meta_note"] = f"high delta ({confidence_delta:.2f})"
        return winner

    if slow["class"] in {"Physical-Damage", "Electrical-Damage", "Bird-drop"}:
        result = slow.copy()
        result["meta_note"] = "risky class — deferred to slow"
        return result

    result = (slow if slow["confidence"] >= fast["confidence"] else fast).copy()
    result["meta_note"] = "confidence tiebreak"
    return result


# MAIN AGENT 
@traceable(name="perception_agent", run_type="chain")
def perception_agent(image_path, zone="zone_0_0", panel_index=0):
    features = extract_features(image_path)
    if "error" in features:
        return {"class": "Unknown", "confidence": 0.0, "reasoning": "invalid image"}

    filename_class, filename_conf = class_from_filename(image_path)
    filename_hint = os.path.basename(image_path) if filename_class else ""

    rag_context         = retrieve_context(features)
    panel_graph_context = query_panel_graph(zone, panel_index)
    panel_history       = get_panel(zone, panel_index)

    fast = fast_classifier(features)
    slow = llm_classifier(
        features, rag_context, panel_graph_context,
        panel_history, filename_hint, filename_class
    )
    final = meta_reasoner(
        fast, slow, panel_history, panel_graph_context,
        filename_class=filename_class, filename_conf=filename_conf
    )

    log_event("perception", "final_decision", {
        "zone": zone, "panel_index": panel_index,
        "panel_id": f"{zone}_p{panel_index}",
        "image": os.path.basename(image_path),
        "filename_class": filename_class,
        "rag_hits": len(rag_context),
        "graph_insights": len(panel_graph_context),
        "history_depth": len(panel_history),
        "fast": fast, "slow": slow, "final": final
    })

    return final
