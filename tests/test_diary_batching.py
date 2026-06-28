"""Phase 4: diary journal batching. Many independent soldier journals are generated in a single
batched fan-out instead of one sequential call each."""
from types import SimpleNamespace


_SAMPLE = {
    "Name": "Test Soldier",
    "Age": 30,
    "Family": "None",
    "Occupation": "Pikeman",
    "Personality": "Stoic",
    "Social Status": "Commoner",
    "Potential Illness": "None",
    "Body Condition": "Healthy",
    "Hobbies and Interests": "Whittling",
    "Style of Talking": "Terse",
    "Unique Quirks": "Hums",
    "Secrets or Scandals": "None",
}


def _make_profile(model_type="fake"):
    from group_experience.individual_profile import Soldier_Profile

    return Soldier_Profile(dict(_SAMPLE), model_type=model_type)


def test_generate_journal_batch_returns_one_per_prompt_over_fake():
    p = _make_profile("fake")
    profiles = [p, p, p]
    journals = p.generate_journal_batch(profiles, ["p1", "p2", "p3"])
    assert len(journals) == 3
    assert all(isinstance(j, str) and j for j in journals)


def test_generate_journal_batch_empty_is_noop():
    p = _make_profile("fake")
    assert p.generate_journal_batch([], []) == []


def test_generate_journal_batch_accounts_diary_tokens(monkeypatch):
    from utils import LLM_api
    from utils.token_accounting import TokenAccumulator

    class _Resp:
        def __init__(self, i):
            self.content = f"journal{i}"
            self.usage_metadata = {"input_tokens": 4, "output_tokens": 6}

    class _Model:
        def batch(self, messages_list, config=None):
            return [_Resp(i) for i, _ in enumerate(messages_list)]

    monkeypatch.setattr(LLM_api, "_get_model", lambda mt: _Model())

    p = _make_profile("fake")
    acc = TokenAccumulator()
    journals = p.generate_journal_batch([p, p], ["a", "b"], accumulator=acc)

    assert journals == ["journal0", "journal1"]
    assert acc.diary.calls == 2
    assert acc.diary.input_tokens == 8
    assert acc.diary.output_tokens == 12


def test_soldier_agent_prepare_collect_split():
    """prepare() builds a prompt (and runs structure logic); collect() stores the journal under
    the given time key — together equivalent to the old single execute()."""
    from group_experience.individual_profile import Soldier_Agent

    profile = _make_profile("fake")
    # Stub hierarchy: no structural change so prepare just builds the prompt.
    hierarchy = SimpleNamespace(
        monitor_structure_change=lambda executed_troop: [],
        current_troop=None,
    )
    soldier = Soldier_Agent(profile, hierarchy)

    executed_troop = SimpleNamespace()
    prompt = soldier.prepare(executed_troop, command="advance", surrounding="enemy ahead")
    assert isinstance(prompt, str) and "advance" in prompt

    soldier.collect("t0", "my journal entry")
    assert profile.journal["t0"] == "my journal entry"
