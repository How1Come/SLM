import json
import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rp.settings import DEFAULTS, LearningSettings
from rp.store import Store


class StoreTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = Store(self.directory.name)
        self.messages = [{"role": "user", "content": "你好"}]

    def signal(self, chosen="更好的回复", rejected="原回复", kind="fix"):
        generation = self.store.add_generation("scope:turn", self.messages, "原回复", "base", {}, session_id="chat", scope_id="scope")
        return self.store.add_signal(kind, "scope:turn", generation, self.messages, chosen=chosen, rejected=rejected)

    def test_concurrent_connections_do_not_share_transactions(self):
        other = Store(self.directory.name)

        def add(index):
            store = self.store if index % 2 else other
            identifier = store.add_generation(f"scope:turn-{index}", self.messages, str(index), "base", {})
            store.add_signal("good", f"scope:turn-{index}", identifier, self.messages, chosen=str(index))
            store.stats()

        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(add, range(80)))
        self.assertEqual(self.store.stats()["generations"], 80)
        self.assertEqual(len(self.store.active_signals()), 80)

    def test_context_manager_and_close_leave_no_shared_connection(self):
        with Store(self.directory.name) as store:
            self.assertEqual(store.stats()["total_signals"], 0)
        store.close()
        self.signal()
        self.signal(chosen=None, kind="swipe")
        self.assertEqual(store.stats()["total_signals"], 2)
        self.assertEqual(len(store.active_signals()), 1)

    def test_updates_during_training_remain_unconsumed(self):
        identifier = self.signal()
        before = self.store.get_signal(identifier)
        self.store.update_signal(identifier, chosen="最新修正")
        self.store.mark_consumed([identifier], "v1", expected_versions={identifier: before["record_version"]})
        current = self.store.get_signal(identifier)
        self.assertIsNone(current["consumed_in"])
        self.assertGreater(current["record_version"], before["record_version"])
        self.store.mark_consumed([identifier], "v2", expected_versions={identifier: current["record_version"]})
        self.assertEqual(self.store.get_signal(identifier)["consumed_in"], "v2")
        self.store.update_signal(identifier, chosen="再修正")
        self.assertIsNone(self.store.get_signal(identifier)["consumed_in"])

    def test_review_excludes_and_reenables_signal(self):
        identifier = self.signal()
        self.store.mark_consumed([identifier], "v1")
        excluded = self.store.review_signal(identifier, False)
        self.assertEqual(excluded["review_status"], "excluded")
        self.assertEqual(self.store.active_signals(), [])
        self.store.review_signal(identifier, True)
        self.assertEqual(self.store.stats()["new_signals_since_sleep"], 1)
        self.assertIsNone(self.store.get_signal(identifier)["consumed_in"])

    def test_bare_swipes_do_not_train_or_trigger_sleep(self):
        self.signal(chosen=None, kind="swipe")
        self.assertEqual(self.store.active_signals(), [])
        self.assertEqual(self.store.stats()["new_signals_since_sleep"], 0)
        self.assertEqual(len(self.store.recent_signals()), 1)

    def test_export_and_pagination_preserve_normalized_data(self):
        first = self.signal()
        second = self.signal(chosen="后来的修正")
        self.store.review_signal(first, False)
        self.assertEqual(self.store.recent_signals(n=1)[0]["id"], second)
        self.assertEqual(self.store.recent_signals(n=1, offset=1)[0]["id"], first)
        self.assertEqual(self.store.recent_signals(active=False)[0]["id"], first)
        exported = self.store.export_feedback()
        self.assertEqual(len(exported), 1)
        self.assertEqual(exported[0]["messages"][-1], {"role": "assistant", "content": "后来的修正"})
        self.assertEqual(exported[0]["preference"]["prompt"], self.messages)
        self.assertEqual(exported[0]["metadata"]["session_id"], "chat")
        self.assertEqual(len(self.store.export_feedback(active_only=False)), 2)
        json.dumps(exported, ensure_ascii=False)

    def test_update_column_names_are_allowlisted(self):
        identifier = self.signal()
        with self.assertRaises(ValueError):
            self.store.update_signal(identifier, **{"active=0 WHERE id>0; --": 1})
        with self.assertRaises(ValueError):
            self.store.recent_signals(n=0)
        self.assertEqual(self.store.get_signal(identifier)["active"], 1)

    def test_migration_preserves_and_quarantines_legacy_feedback(self):
        with tempfile.TemporaryDirectory() as legacy_dir:
            with sqlite3.connect(Path(legacy_dir) / "rp.db") as database:
                database.execute("CREATE TABLE signals (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, "
                                 "kind TEXT NOT NULL, turn_key TEXT NOT NULL, gen_id TEXT, messages TEXT NOT NULL, "
                                 "chosen TEXT, rejected TEXT, weight REAL NOT NULL, active INTEGER NOT NULL DEFAULT 1, consumed_in TEXT)")
                database.execute("INSERT INTO signals (ts,kind,turn_key,messages,chosen,weight) VALUES (1,'edit','old','[]','legacy edit',1)")
            migrated = Store(legacy_dir)
            signal = migrated.recent_signals()[0]
            self.assertEqual(signal["chosen"], "legacy edit")
            self.assertEqual(signal["scope_id"], "legacy")
            self.assertEqual(signal["review_status"], "needs_review")
            self.assertEqual(migrated.active_signals(), [])
            migrated.review_signal(signal["id"], True)
            reopened = Store(legacy_dir)
            self.assertEqual(reopened.active_signals()[0]["chosen"], "legacy edit")


class LearningSettingsTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.settings = LearningSettings(self.directory.name)

    def test_default_and_persistent_updates(self):
        self.assertEqual(self.settings.get(), DEFAULTS)
        updated = self.settings.update({"mode": "background", "epochs": 3, "guard_max_drift": 0.1})
        self.assertEqual(LearningSettings(self.directory.name).get(), updated)
        self.assertTrue(updated["enabled"])
        self.settings.update({"enabled": False})
        self.assertEqual(self.settings.get()["mode"], "background")

    def test_invalid_settings_do_not_modify_disk(self):
        self.settings.update({"mode": "collect_only"})
        original = self.settings.path.read_bytes()
        invalid = [{"epochs": 0}, {"max_minutes": float("nan")}, {"threads": True}, {"enabled": "false"},
                   {"mode": "instant"}, {"threads": 1.5}, {"anything": 1}]
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.settings.update(values)
        self.assertEqual(self.settings.path.read_bytes(), original)

    def test_concurrent_partial_updates_are_not_lost(self):
        keys = ["enabled", "mode", "threads", "epochs"]
        values = [False, "background", 3, 5]
        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(lambda item: LearningSettings(self.directory.name).update(dict([item])), zip(keys, values)))
        result = self.settings.get()
        for key, value in zip(keys, values):
            self.assertEqual(result[key], value)
        self.assertEqual(list(Path(self.directory.name).glob(".learning-*.json")), [])

    def test_corrupt_file_has_actionable_error(self):
        self.settings.path.write_text("bad-json", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "restore or remove"):
            self.settings.get()


if __name__ == "__main__":
    unittest.main()
