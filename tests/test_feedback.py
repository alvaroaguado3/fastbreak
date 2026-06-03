from fastbreak.feedback.store import FeedbackStore
from fastbreak.generation.templates import TemplateGenerator
from fastbreak.selection.entropy import SurpriseSelector
from fastbreak.models import GameContext, StatEvent, StatFlag


def _selected():
    ctx = GameContext("g", "NYK", "SAS", game_type="finals", venue="MSG")
    ev = StatEvent("jb", "Brunson", "player", "points", 41.0, ctx)
    f = StatFlag(ev, 0.99, "finals", 200, 7.0)
    return SurpriseSelector(threshold=0.0).select(TemplateGenerator().generate([f], n=4))


def test_log_and_label_roundtrip(tmp_path):
    fb = FeedbackStore(tmp_path / "fb.jsonl")
    hid = fb.log_published(_selected())
    fb.record_signal(hid, reused=True, source="test")
    labeled = list(fb.iter_labeled())
    assert len(labeled) == 1
    rec, reused = labeled[0]
    assert reused is True and rec["headline_id"] == hid
