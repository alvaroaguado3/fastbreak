from fastbreak.publish.x_connector import XPublisher
from fastbreak.generation.templates import TemplateGenerator
from fastbreak.selection.entropy import SurpriseSelector
from fastbreak.models import GameContext, StatEvent, StatFlag


def _selected(text_val=41):
    ctx = GameContext("g", "NYK", "SAS", game_type="finals", venue="MSG")
    ev = StatEvent("jb", "Jalen Brunson", "player", "points", float(text_val), ctx)
    f = StatFlag(ev, 0.99, "finals", 200, 7.0)
    cands = TemplateGenerator().generate([f], n=4)
    return SurpriseSelector(threshold=0.0).select(cands)


def test_dry_run_does_not_post():
    pub = XPublisher(dry_run=True)
    res = pub.publish(_selected())
    assert res["published"] is False and res["dry_run"] is True


def test_tweet_within_limit_and_has_hashtag():
    pub = XPublisher(dry_run=True, hashtags=("#NBAFinals",))
    tweet = pub.format_tweet(_selected())
    assert len(tweet) <= 280
    assert "#NBAFinals" in tweet


def test_urls_stripped_to_avoid_surcharge():
    pub = XPublisher(dry_run=True, strip_urls=True)
    sel = _selected()
    sel.candidate.text = "Brunson drops 41 see http://example.com/x now"
    tweet = pub.format_tweet(sel)
    assert "http" not in tweet
