from fastbreak.selection.entropy import SurpriseSelector, NoveltyTracker
from fastbreak.generation.templates import TemplateGenerator
from fastbreak.models import GameContext, StatEvent, StatFlag


def _cands(stat="blocks", val=9, bits=8.0):
    ctx = GameContext("g", "NYK", "SAS", game_type="finals", venue="MSG")
    ev = StatEvent("vw", "Wemby", "player", stat, float(val), ctx)
    f = StatFlag(ev, 0.99, "finals", 200, bits)
    return TemplateGenerator().generate([f], n=4)


def test_high_surprisal_passes_threshold():
    sel = SurpriseSelector(threshold=0.5)
    out = sel.select(_cands(bits=9.0))
    assert out is not None and out.surprise_score >= 0.5


def test_low_surprisal_filtered():
    sel = SurpriseSelector(threshold=0.9)
    assert sel.select(_cands(bits=0.1)) is None


def test_novelty_penalizes_repeats():
    nt = NoveltyTracker()
    assert nt.novelty("Wemby has 9 blocks") == 1.0
    nt.remember("Wemby has 9 blocks")
    assert nt.novelty("Wemby has 9 blocks") < 0.2


def test_feedback_update_moves_weights():
    sel = SurpriseSelector(threshold=0.5)
    out = sel.select(_cands(bits=8.0))
    before = list(sel.weights)
    sel.update_from_feedback(out, reused=True, lr=0.2)
    assert sel.weights != before
