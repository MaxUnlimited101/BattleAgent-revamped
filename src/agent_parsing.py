"""Commander LLM-response parsing and semantic validation.

Extracted from ``Detachment_Agent`` (Phase 6 decomposition). These are pure functions —
text in, JSON/error-list out — with no agent state. ``Detachment_Agent`` keeps
``parse_llm_output``/``validate_parsed_output`` as thin facade methods that delegate here,
so existing callers and tests are unaffected.
"""

import logging
from typing import Any, Union

import json5 as json

logger = logging.getLogger(__name__)


def parse_llm_output(llm_response: str) -> Union[dict, str]:
    """Extract the JSON object from a raw LLM response.

    Assumes the JSON payload starts at the first ``{`` and ends at the last ``}``. Returns the
    parsed ``dict`` on success, or an ``"Error ..."`` string when no valid JSON can be extracted.
    """

    def extract_json_from_text(text: str) -> Union[dict, str]:
        try:
            # Finding the indices of the first '{' and the last '}'
            start_index = text.index('{')
            end_index = text.rindex('}') + 1

            # Extracting the JSON substring
            json_part = text[start_index:end_index]

            # Attempting to parse the JSON part
            parsed_json = json.loads(json_part)
            return parsed_json
        except ValueError as e:
            # Error handling if JSON parsing fails
            return f"Error extracting JSON: {e}"

    extracted_json = extract_json_from_text(llm_response)

    if not isinstance(extracted_json, dict):
        logger.warning("Failed to extract JSON from LLM response: %s", llm_response)

    return extracted_json


def validate_parsed_output(json_data: dict) -> list[str]:
    """Semantic (legacy) validation of a parsed commander decision.

    Returns a list of human-readable error messages; an empty list means the payload is valid.
    """
    invalid_messages: list[str] = []

    # Validate root object fields
    if not isinstance(json_data.get("agentNextActionType"), str):
        invalid_messages.append("agentNextActionType must be a String")
    if not isinstance(json_data.get("remarks"), str):
        invalid_messages.append("remarks must be a String")
    if not isinstance(json_data.get("SubAgentsRecall"), list) or not all(isinstance(item, str) for item in json_data.get("SubAgentsRecall", [])):
        invalid_messages.append("SubAgentsRecall must be a list of Strings")
    if json_data.get("agentMoral") not in ["High", "Medium", "Low"]:
        invalid_messages.append("agentMoral must be 'High', 'Medium', or 'Low'")
    if not isinstance(json_data.get("speed"), int):
        invalid_messages.append("speed must be an Integer")
    if not isinstance(json_data.get("agentNextPosition"), list) or len(json_data.get("agentNextPosition", [])) != 2 or not all(isinstance(item, int) for item in json_data.get("agentNextPosition", [])):
        invalid_messages.append("agentNextPosition must be an Array of 2 Integers")
    if not isinstance(json_data.get("deploySubUnit"), bool):
        invalid_messages.append("deploySubUnit must be a Boolean")
    if not isinstance(json_data.get("targetedAgentId", ""), str):
        invalid_messages.append("targetedAgentId must be a String")

    # Validate actions array and its objects
    actions: Any = json_data.get("actions", [])
    if not isinstance(actions, list):
        invalid_messages.append("actions must be an Array of Objects")
    else:
        for action in actions:
            if not isinstance(action.get("subAgent_NextActionType"), str):
                invalid_messages.append("Each action's subAgent_NextActionType must be a String")
            if not isinstance(action.get("troopType"), str):
                invalid_messages.append("Each action's troopType must be a String")
            if not isinstance(action.get("speed"), int):
                invalid_messages.append("Each action's speed must be an Integer")
            if not isinstance(action.get("deployedNum"), int):
                invalid_messages.append("Each action's deployedNum must be an Integer")
            if not isinstance(action.get("ownPotentialLostNum"), int):
                invalid_messages.append("Each action's ownPotentialLostNum must be an Integer")
            if not isinstance(action.get("enemyPotentialLostNum"), int):
                invalid_messages.append("Each action's enemyPotentialLostNum must be an Integer")
            if not isinstance(action.get("position"), list) or len(action.get("position", [])) != 2 or not all(isinstance(item, int) for item in action.get("position", [])):
                invalid_messages.append("Each action's position must be an Array of 2 Integers")
            if not isinstance(action.get("agent_id"), str):
                invalid_messages.append("Each action's agent_id must be a String")
            if not isinstance(action.get("remarks"), str):
                invalid_messages.append("Each action's remarks must be a String")

    return invalid_messages
