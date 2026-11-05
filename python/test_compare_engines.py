"""Tests for the matched comparison of the three engines: each trained from
scratch for one hour with the same settings, then rated by a round robin."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import compare_engines as ce


def test_two_seeds_of_each_engine_with_the_best_tuned_settings():
    trials = ce.trials(seeds=(0, 1))
    assert [t["name"] for t in trials] == [
        "network_s0", "full_s0", "policy_s0", "network_s1", "full_s1", "policy_s1"]
    for t in trials:
        assert t["params"] == ce.BO19


def test_each_engine_trains_with_its_own_search():
    def cfg(engine):
        return ce.config({"name": "x", "engine": engine, "seed": 3, "params": ce.BO19},
                         "/w", "/tb", "/bin/sp")
    net, full, policy = cfg("network"), cfg("full"), cfg("policy")
    assert (net.enable_tier1, net.enable_material, net.policy_only) == (False, False, False)
    assert (full.enable_tier1, full.enable_material, full.policy_only) == (True, True, False)
    assert (policy.enable_tier1, policy.enable_material, policy.policy_only) == (True, True, True)
    for c in (net, full, policy):
        assert c.max_seconds == 3600 and c.syzygy_path == "/tb" and c.mate_checks == 5
        assert c.c_puct == ce.BO19["c_puct"] and c.simulations_per_move == 100
        assert c.seed == 3


def test_each_side_of_a_match_searches_as_it_was_trained():
    a = {"name": "network_s0", "engine": "network", "weights": "/a.pth", "params": ce.BO19}
    b = {"name": "policy_s0", "engine": "policy", "weights": "/b.pth", "params": ce.BO19}
    cmd = ce.match_command(a, b, 100, 100, 7, "/tb", "/bin/eval")
    assert cmd[:5] == ["/bin/eval", "/a.pth", "/b.pth", "100", "100"]
    assert "--candidate-disable-tier1" in cmd and "--candidate-disable-material" in cmd
    assert cmd[cmd.index("--current-policy-only") + 1] == "revisit"
    assert "--current-disable-tier1" not in cmd and "--disable-tier1" not in cmd
    assert "--candidate-policy-only" not in cmd
    assert cmd[cmd.index("--syzygy") + 1] == "/tb"
    assert cmd[cmd.index("--seed-offset") + 1] == "7"


def test_the_full_engine_needs_no_flags():
    a = {"name": "full_s0", "engine": "full", "weights": "/a.pth", "params": ce.BO19}
    b = {"name": "full_s1", "engine": "full", "weights": "/b.pth", "params": ce.BO19}
    cmd = ce.match_command(a, b, 10, 100, 0, "", "/bin/eval")
    assert not any("disable" in x or "policy-only" in x for x in cmd)


def test_the_round_robin_plays_each_pair_once():
    pairs = ce.round_robin(["a", "b", "c", "d"])
    assert len(pairs) == 6 and len(set(map(frozenset, pairs))) == 6
