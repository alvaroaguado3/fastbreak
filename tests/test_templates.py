from fastbreak.generation.templates import TemplateGenerator
from fastbreak.models import GameContext, StatEvent, StatFlag


def _flag(name, stat, val, pct=0.97, bits=5.0, ctx_key="finals"):
    ctx = GameContext("g", "NYK", "SAS", game_type="finals", venue="MSG")
    ev = StatEvent(name.lower(), name, "player", stat, float(val), ctx)
    return StatFlag(ev, pct, ctx_key, 200, bits)


def test_single_stat_headlines():
    cands = TemplateGenerator().generate([_flag("Jalen Brunson", "points", 41)], n=4)
    assert len(cands) >= 1
    assert all("Brunson" in c.text and "41" in c.text for c in cands)
    assert all(c.source == "template" for c in cands)


def test_multi_stat_same_player_combines():
    flags = [_flag("Wemby", "blocks", 9, bits=8.0), _flag("Wemby", "points", 38, bits=4.0)]
    cands = TemplateGenerator().generate(flags, n=4)
    assert any("9" in c.text and "38" in c.text for c in cands)


def test_multi_stat_two_players():
    flags = [_flag("Brunson", "points", 41, bits=6.0), _flag("Wemby", "blocks", 9, bits=8.0)]
    cands = TemplateGenerator().generate(flags, n=4)
    assert any("Brunson" in c.text and "Wemby" in c.text for c in cands)


def test_empty_flags():
    assert TemplateGenerator().generate([], n=4) == []
