from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from llm_client import LLMClient

CLUSTER_BUCKET_KEYS = (
    "reinforcement_clusters",
    "manual_review_clusters",
    "risk_control_clusters",
)

def count_reinforcement_clusters(reinforcement_plan: Dict) -> int:
    """Count current compression cluster buckets, with legacy fallback."""
    bucket_count = sum(
        len(reinforcement_plan.get(bucket_key, []))
        for bucket_key in CLUSTER_BUCKET_KEYS
    )
    if bucket_count:
        return bucket_count
    return len(reinforcement_plan.get("compressed_reinforcement_plan", []))

def run_compression(
    validated_calibration: Dict,
    compact_dict: Dict,
    compression_prompt: str,
    llm_client: "LLMClient"
) -> Dict:
    """Run the compression step to create non-duplicative reinforcement plan."""
    payload = {
        "WS_THEME_DICTIONARY_JSON": compact_dict,
        "VALIDATED_CALIBRATION_JSON": validated_calibration,
    }
    response = llm_client.complete_json(compression_prompt, payload)
    return response
