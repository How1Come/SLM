"""Learning-signal detection without a model: python -m unittest tests.test_rp_learn"""
import json
import tempfile
import unittest

from rp.learn import ACK_PREFIX, Learner, clean_messages, parse_command
from rp.store import Store

SYS = [{"role": "system", "content": "Write 苏小糖's next reply."}, {"role": "system", "content": "[Start a new Chat]"}]
GREET = {"role": "assistant", "content": "哼，哪里来的小书生？"}


class Sim:
    """Mimics what SillyTavern sends and what the server logs."""

    def __init__(self):
        self.store = Store(tempfile.mkdtemp())
        self.learner = Learner(self.store)

    def gen(self, history, output, session_id=None):
        a = self.learner.analyze(history, session_id=session_id)
        self.store.add_generation(a.turn_key, a.messages, output, "base", {}, prefill=a.prefill,
                                  scope_id=a.scope_id, session_id=a.session_id)
        return a

    def kinds(self):
        return sorted((s["kind"], s["chosen"], s["rejected"]) for s in self.store.active_signals())


class LearnTest(unittest.TestCase):
    def test_commands(self):
        self.assertEqual(parse_command("!good"), ("good", ""))
        self.assertEqual(parse_command("！改 本座才不稀罕"), ("fix", "本座才不稀罕"))
        self.assertEqual(parse_command("!fix: 哼"), ("fix", "哼"))
        self.assertIsNone(parse_command("good morning!"))
        self.assertIsNone(parse_command("!goodbye"))

    def test_clean_merges_systems_and_drops_feedback(self):
        msgs = SYS + [GREET, {"role": "user", "content": "!bad"}, {"role": "assistant", "content": ACK_PREFIX + "ok"}]
        out = clean_messages(msgs)
        self.assertEqual([m["role"] for m in out], ["system", "assistant"])

    def test_accept_edit_swipe(self):
        s = Sim()
        h1 = SYS + [GREET, {"role": "user", "content": "你是谁？"}]
        s.gen(h1, "我是一个AI助手，有什么可以帮你？")
        h2 = h1 + [{"role": "assistant", "content": " 我是一个AI助手，有什么可以帮你？\n"}, {"role": "user", "content": "你住哪？"}]
        s.gen(h2, "我住在网上。")
        self.assertEqual(s.kinds(), [])
        edited = "*尾巴一甩* 本座自然住在桂花树下，小书生问这个做什么？"
        h3 = h2 + [{"role": "assistant", "content": edited}, {"role": "user", "content": "我饿了"}]
        s.gen(h3, "我给你做饭吧。")
        self.assertEqual(s.kinds(), [("edit", edited, "我住在网上。")])
        s.gen(h3, "饿了就去吃饭。")
        s.gen(h3, "*耳朵一动* 哼，本座这儿只有桂花糕。")
        kept = "*耳朵一动* 哼，本座这儿只有桂花糕。"
        s.learner.analyze(h3 + [{"role": "assistant", "content": kept}, {"role": "user", "content": "给我一块"}])
        swipes = [k for k in s.kinds() if k[0] == "swipe"]
        self.assertEqual(sorted(r for _, _, r in swipes), sorted(["我给你做饭吧。", "饿了就去吃饭。"]))
        self.assertTrue(all(c == kept for _, c, _ in swipes))
        self.assertEqual(sum(1 for k in s.kinds() if k[0] == "edit"), 1)

    def test_swiping_back_to_first_reply(self):
        s = Sim()
        h1 = SYS + [GREET, {"role": "user", "content": "你好"}]
        s.gen(h1, "A 回复")
        s.gen(h1, "B 回复")
        s.learner.analyze(h1 + [{"role": "assistant", "content": "A 回复"}, {"role": "user", "content": "嗯"}])
        self.assertEqual(s.kinds(), [("swipe", "A 回复", "B 回复")])

    def test_continue_matches_full_message(self):
        s = Sim()
        h1 = SYS + [GREET, {"role": "user", "content": "讲个故事"}]
        s.gen(h1, "从前有座山，")
        a = s.gen(h1 + [{"role": "assistant", "content": "从前有座山，"}], "山里有座庙。")
        self.assertEqual(a.kind, "continue")
        s.learner.analyze(h1 + [{"role": "assistant", "content": "从前有座山，山里有座庙。"}, {"role": "user", "content": "然后呢"}])
        self.assertEqual(s.kinds(), [])

    def test_explicit_feedback(self):
        s = Sim()
        h1 = SYS + [GREET, {"role": "user", "content": "你好"}]
        s.gen(h1, "你好！有什么可以帮你？")
        h2 = h1 + [{"role": "assistant", "content": "你好！有什么可以帮你？"}, {"role": "user", "content": "!fix 哼，本座可不是你的丫鬟。"}]
        a = s.learner.analyze(h2)
        self.assertEqual(a.kind, "command")
        s.learner.apply_command(a)
        self.assertEqual(s.kinds(), [("fix", "哼，本座可不是你的丫鬟。", "你好！有什么可以帮你？")])
        sig = s.store.active_signals()[0]
        self.assertEqual(json.loads(sig["messages"])[-1]["content"], "你好")

    def test_character_cards_with_identical_turns_do_not_collide(self):
        s = Sim()
        user = {"role": "user", "content": "你是谁？"}
        first = [{"role": "system", "content": "你是苏小糖"}, user]
        second = [{"role": "system", "content": "你是月亮侦探"}, user]
        first_analysis = s.gen(first, "本座是苏小糖。")
        second_analysis = s.gen(second, "我是侦探。")
        self.assertNotEqual(first_analysis.turn_key, second_analysis.turn_key)
        self.assertEqual(s.kinds(), [])
        s.learner.analyze(second + [{"role": "assistant", "content": "我是月亮侦探。"}, {"role": "user", "content": "明白"}])
        signal = s.store.active_signals()[0]
        self.assertEqual(signal["rejected"], "我是侦探。")
        self.assertIn("月亮侦探", json.loads(signal["messages"])[0]["content"])

    def test_explicit_sessions_with_same_card_do_not_collide(self):
        s = Sim()
        history = SYS + [GREET, {"role": "user", "content": "你好"}]
        first = s.gen(history, "A reply", session_id="first")
        second = s.gen(history, "B reply", session_id="second")
        self.assertNotEqual(first.turn_key, second.turn_key)
        s.learner.analyze(history + [{"role": "assistant", "content": "B reply"}, {"role": "user", "content": "继续"}], session_id="second")
        self.assertEqual(s.kinds(), [])

    def test_session_survives_system_updates_and_history_trimming(self):
        s = Sim()
        history = SYS + [GREET, {"role": "user", "content": "你好"},
                         {"role": "assistant", "content": "坐下吧。"}, {"role": "user", "content": "茶呢"}]
        original = s.gen(history, "没有茶。", session_id="lecture-chat")
        trimmed = [{"role": "system", "content": "角色卡附加新的世界信息"}, *history[-2:]]
        continued = s.learner.analyze(trimmed + [{"role": "assistant", "content": "桂花茶在桌上。"},
                                              {"role": "user", "content": "谢谢"}], session_id="lecture-chat")
        signal = s.store.active_signals()[0]
        self.assertEqual(signal["turn_key"], original.turn_key)
        self.assertEqual(signal["session_id"], "lecture-chat")
        self.assertEqual(signal["chosen"], "桂花茶在桌上。")
        self.assertEqual(continued.scope_id, original.scope_id)

    def test_long_inputs_with_identical_suffixes_have_distinct_keys(self):
        s = Sim()
        first = SYS + [{"role": "user", "content": "A" + "同" * 1000}]
        second = SYS + [{"role": "user", "content": "B" + "同" * 1000}]
        self.assertNotEqual(s.gen(first, "第一条").turn_key, s.gen(second, "第二条").turn_key)

    def test_reverted_edit_is_removed_from_training(self):
        s = Sim()
        history = SYS + [GREET, {"role": "user", "content": "你好"}]
        s.gen(history, "原回复")
        s.learner.analyze(history + [{"role": "assistant", "content": "临时修改"}, {"role": "user", "content": "继续"}])
        self.assertEqual(len(s.store.active_signals()), 1)
        s.learner.analyze(history + [{"role": "assistant", "content": "原回复"}, {"role": "user", "content": "继续"}])
        self.assertEqual(s.store.active_signals(), [])

    def test_reviewed_exclusion_survives_history_reprocessing(self):
        s = Sim()
        history = SYS + [GREET, {"role": "user", "content": "你好"}]
        s.gen(history, "原回复")
        edited = history + [{"role": "assistant", "content": "修正回复"}, {"role": "user", "content": "继续"}]
        s.learner.analyze(edited)
        identifier = s.store.active_signals()[0]["id"]
        s.store.review_signal(identifier, False)
        s.learner.analyze(edited)
        self.assertEqual(s.store.active_signals(), [])
        self.assertEqual(s.store.get_signal(identifier)["review_status"], "excluded")

    def test_swipe_choice_updates_previously_consumed_pairs(self):
        s = Sim()
        history = SYS + [GREET, {"role": "user", "content": "你好"}]
        s.gen(history, "A 回复")
        s.gen(history, "B 回复")
        next_user = {"role": "user", "content": "继续"}
        s.learner.analyze(history + [{"role": "assistant", "content": "B 回复"}, next_user])
        identifiers = [signal["id"] for signal in s.store.active_signals()]
        s.store.mark_consumed(identifiers, "v1")
        s.learner.analyze(history + [{"role": "assistant", "content": "A 回复"}, next_user])
        self.assertEqual(s.kinds(), [("swipe", "A 回复", "B 回复")])
        self.assertTrue(all(signal["consumed_in"] is None for signal in s.store.active_signals()))

    def test_repeated_commands_are_idempotent_and_opposite_feedback_is_superseded(self):
        s = Sim()
        history = SYS + [GREET, {"role": "user", "content": "你好"}]
        s.gen(history, "原回复")
        replied = history + [{"role": "assistant", "content": "原回复"}]
        good = s.learner.analyze(replied + [{"role": "user", "content": "!good"}])
        s.learner.apply_command(good)
        s.learner.apply_command(good)
        self.assertEqual(len(s.store.active_signals()), 1)
        bad = s.learner.analyze(replied + [{"role": "user", "content": "!bad"}])
        s.learner.apply_command(bad)
        self.assertEqual(s.kinds(), [("bad", None, "原回复")])

    def test_regenerating_a_settled_turn_clears_stale_swipe_preferences(self):
        s = Sim()
        history = SYS + [GREET, {"role": "user", "content": "你好"}]
        s.gen(history, "A 回复")
        s.gen(history, "B 回复")
        s.learner.analyze(history + [{"role": "assistant", "content": "B 回复"},
                                     {"role": "user", "content": "继续"}])
        self.assertEqual(s.kinds(), [("swipe", "B 回复", "A 回复")])
        s.gen(history, "C 回复")
        self.assertEqual(s.store.active_signals(), [])
        self.assertEqual(s.store.stats()["new_signals_since_sleep"], 0)
        s.learner.analyze(history + [{"role": "assistant", "content": "C 回复"},
                                     {"role": "user", "content": "继续"}])
        self.assertEqual(s.kinds(), [("swipe", "C 回复", "A 回复"), ("swipe", "C 回复", "B 回复")])

    def test_unscoped_legacy_generation_is_not_used_for_new_feedback(self):
        s = Sim()
        history = SYS + [GREET, {"role": "user", "content": "你好"}]
        s.store.add_generation("legacy-turn", history, "其他角色的回复", "base", {})
        s.learner.analyze(history + [{"role": "assistant", "content": "修正回复"}, {"role": "user", "content": "继续"}])
        self.assertEqual(s.kinds(), [])


if __name__ == "__main__":
    unittest.main()
