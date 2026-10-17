import numpy as np
import pytest

from memebot.bot.storage import Storage
from memebot.data.build_dataset import find_duplicates
from memebot.evaluation.metrics import score
from memebot.evaluation.online import compute
from memebot.search.hybrid import rrf


def test_metrics_known_item():
    # Релевантный на 2-м месте: recall@1 = 0, recall@5 = 1, MRR = 1/2, nDCG = 1/log2(3)
    r = score([[("a", 1.0), ("b", 0.5)]], {"0": {"b": 1}})
    assert r["recall@1"] == 0 and r["recall@5"] == 1 and r["mrr@10"] == 0.5
    assert r["ndcg@10"] == pytest.approx(1 / np.log2(3), abs=1e-4)


def test_metrics_graded_ideal_order():
    r = score([[("x", 1.0), ("y", 0.9)]], {"0": {"x": 2, "y": 1}})
    assert r["ndcg@10"] == pytest.approx(1.0)


def test_rrf_prefers_consensus():
    a = [[("m1", 9.0), ("m2", 8.0)]]
    b = [[("m3", 0.9), ("m2", 0.8)]]
    fused = [m for m, _ in rrf([a, b])[0]]
    assert fused[0] == "m2"  # второй в обоих списках выигрывает у первых мест, найденных одним ретривером
    assert set(fused) == {"m1", "m2", "m3"}


def test_dedup_same_template_different_caption_kept_apart():
    h = np.array([0b1010, 0b1011, 0b1010], dtype=np.uint64)  # почти одинаковые картинки
    desc = ["кот: когда купил вино", "кот: когда купил наушники без проводов", "кот: когда купил вино"]
    canon = find_duplicates(h, desc)
    assert canon[1] == 1  # другой текст на том же шаблоне — отдельный мем
    assert canon[2] == 0  # та же картинка и то же описание — дубликат


def test_dedup_without_descriptions_needs_near_identical_image():
    h = np.array([0, 0b11, 0b111111], dtype=np.uint64)  # расстояния 2 и 6 от первого
    canon = find_duplicates(h, ["", "", ""])
    assert canon[1] == 0 and canon[2] == 2


@pytest.fixture
def storage(tmp_path):
    return Storage(tmp_path / "bot.db", salt="test")


def test_feedback_only_from_owner_and_once(storage):
    sid = storage.log_search(1, "q", ["a", "b"], {"bm25_ms": 1, "dense_ms": 2, "fuse_ms": 0})
    imp = storage.log_impression(sid, 1, "a", 0)
    assert storage.impression(imp, user_id=2) is None  # чужой показ не виден
    assert storage.impression(imp, user_id=1) == (sid, "a", 0)
    assert storage.log_feedback(imp, "like") is True
    assert storage.log_feedback(imp, "dislike") is False  # вторая реакция не принимается


def test_online_metrics(storage):
    t = {"bm25_ms": 1, "dense_ms": 2, "fuse_ms": 0}
    s1 = storage.log_search(1, "q1", ["a", "b"], t)
    storage.log_feedback(storage.log_impression(s1, 1, "a", 0), "next")
    storage.log_feedback(storage.log_impression(s1, 1, "b", 1), "like")
    s2 = storage.log_search(2, "q2", ["c"], t)
    storage.log_feedback(storage.log_impression(s2, 2, "c", 0), "dislike")
    m = compute(storage.db_path)
    assert m["searches"] == 2 and m["users"] == 2
    assert m["like_rate"] == 0.5 and m["hit@1"] == 0.0 and m["success_rate"] == 0.5
    assert m["depth"] == 1.5 and m["latency_p50_ms"] == 3.0
