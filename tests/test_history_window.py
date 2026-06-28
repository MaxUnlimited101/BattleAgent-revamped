"""Phase 4: commander prompt history truncation. history_window > 0 keeps only the last K
parsed JSON decisions; <= 0 falls back to the full raw-response dump."""


def _seed_history(agent):
    # Five rounds of parsed decisions + raw responses.
    for nb in range(1, 6):
        agent.extracted_json_history[nb] = {"round": nb, "agentNextActionType": f"Act{nb}"}
        agent.LLM_response_history[nb] = f"RAW-RESPONSE-{nb}"


def test_history_window_keeps_last_k(make_agent):
    agent = make_agent()
    agent.history_window = 3
    _seed_history(agent)

    text = agent._history_text()

    # Last 3 decisions present, older ones absent.
    assert "Act3" in text and "Act4" in text and "Act5" in text
    assert "Act1" not in text and "Act2" not in text
    # Compact JSON form, not the raw dump.
    assert "RAW-RESPONSE" not in text


def test_history_window_window_of_one(make_agent):
    agent = make_agent()
    agent.history_window = 1
    _seed_history(agent)

    text = agent._history_text()
    assert "Act5" in text
    assert "Act4" not in text


def test_history_window_zero_uses_full_raw_history(make_agent):
    agent = make_agent()
    agent.history_window = 0
    _seed_history(agent)

    text = agent._history_text()
    # Legacy behavior: the full raw LLM_response_history dict is dumped.
    assert "RAW-RESPONSE-1" in text
    assert "RAW-RESPONSE-5" in text


def test_history_window_default_is_three(make_agent):
    agent = make_agent()
    assert agent.history_window == 3
