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

    def gen(self, history, output):
        a = self.learner.analyze(history)
        self.store.add_generation(a.turn_key, a.messages, output, "base", {}, prefill=a.prefill)
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
        # user keeps it unchanged (whitespace differences are ignored) and continues
        h2 = h1 + [{"role": "assistant", "content": " 我是一个AI助手，有什么可以帮你？\n"}, {"role": "user", "content": "你住哪？"}]
        s.gen(h2, "我住在网上。")
        self.assertEqual(s.kinds(), [])
        # user edits the second reply, then sends the next message
        edited = "*尾巴一甩* 本座自然住在桂花树下，小书生问这个做什么？"
        h3 = h2 + [{"role": "assistant", "content": edited}, {"role": "user", "content": "我饿了"}]
        s.gen(h3, "我给你做饭吧。")
        self.assertEqual(s.kinds(), [("edit", edited, "我住在网上。")])
        # swipe the last reply twice, keep the second swipe
        s.gen(h3, "饿了就去吃饭。")
        s.gen(h3, "*耳朵一动* 哼，本座这儿只有桂花糕。")
        kept = "*耳朵一动* 哼，本座这儿只有桂花糕。"
        s.learner.analyze(h3 + [{"role": "assistant", "content": kept}, {"role": "user", "content": "给我一块"}])
        swipes = [k for k in s.kinds() if k[0] == "swipe"]
        self.assertEqual(sorted(r for _, _, r in swipes), sorted(["我给你做饭吧。", "饿了就去吃饭。"]))
        self.assertTrue(all(c == kept for _, c, _ in swipes))
        # the edited message keeps being resent with every request: still exactly one edit signal
        self.assertEqual(sum(1 for k in s.kinds() if k[0] == "edit"), 1)

    def test_swiping_back_to_first_reply(self):
        s = Sim()
        h1 = SYS + [GREET, {"role": "user", "content": "你好"}]
        s.gen(h1, "A 回复")
        s.gen(h1, "B 回复")  # swipe
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


if __name__ == "__main__":
    unittest.main()
