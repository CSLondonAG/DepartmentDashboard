import ast
import unittest
from pathlib import Path
from typing import List, Tuple

import pandas as pd


def _load_interval_functions():
    """Load the real helpers without executing the Streamlit application."""
    source_path = Path(__file__).resolve().parents[1] / "department_dashboard.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    wanted = {
        "merge_intervals",
        "concurrent_slot_seconds",
        "residual_capacity_usage",
    }
    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    module = ast.Module(body=functions, type_ignores=[])
    namespace = {
        "pd": pd,
        "List": List,
        "Tuple": Tuple,
        "Interval": Tuple[pd.Timestamp, pd.Timestamp],
        "CHAT_CONCURRENCY_CAPACITY": 2,
    }
    exec(compile(module, str(source_path), "exec"), namespace)
    return (
        namespace["concurrent_slot_seconds"],
        namespace["residual_capacity_usage"],
    )


concurrent_slot_seconds, residual_capacity_usage = _load_interval_functions()


class ChatCapacityTests(unittest.TestCase):
    def setUp(self):
        self.start = pd.Timestamp("2026-10-20 09:00:00")
        self.end = self.start + pd.Timedelta(hours=1)
        self.available = [(self.start, self.end)]

    def test_one_chat_uses_half_of_two_slot_capacity(self):
        slots = concurrent_slot_seconds(
            [(self.start, self.end)], self.available, capacity=2
        )
        self.assertEqual(slots, 3600.0)
        self.assertEqual(slots / 2 / 3600, 0.5)

    def test_two_concurrent_chats_use_full_capacity(self):
        slots = concurrent_slot_seconds(
            [(self.start, self.end), (self.start, self.end)],
            self.available,
            capacity=2,
        )
        self.assertEqual(slots, 7200.0)
        self.assertEqual(slots / 2 / 3600, 1.0)

    def test_more_than_two_chats_is_capped(self):
        slots = concurrent_slot_seconds(
            [(self.start, self.end)] * 3, self.available, capacity=2
        )
        self.assertEqual(slots, 7200.0)

    def test_sequential_single_chats_remain_half_utilised(self):
        midpoint = self.start + pd.Timedelta(minutes=30)
        slots = concurrent_slot_seconds(
            [(self.start, midpoint), (midpoint, self.end)],
            self.available,
            capacity=2,
        )
        self.assertEqual(slots, 3600.0)

    def test_work_is_clipped_to_available_time(self):
        work_start = self.start - pd.Timedelta(hours=1)
        work_end = self.end + pd.Timedelta(hours=1)
        slots = concurrent_slot_seconds(
            [(work_start, work_end), (work_start, work_end)],
            self.available,
            capacity=2,
        )
        self.assertEqual(slots, 7200.0)

    def test_invalid_capacity_is_rejected(self):
        with self.assertRaises(ValueError):
            concurrent_slot_seconds([], self.available, capacity=0)


class EmailResidualCapacityTests(unittest.TestCase):
    def setUp(self):
        self.start = pd.Timestamp("2026-10-20 09:00:00")
        self.end = self.start + pd.Timedelta(hours=1)
        self.shared = [(self.start, self.end)]
        self.email = [(self.start, self.end)]

    def test_no_chat_leaves_full_email_capacity(self):
        capacity, used, excess = residual_capacity_usage(
            [], self.email, self.shared, primary_capacity=2
        )
        self.assertEqual((capacity, used, excess), (3600.0, 3600.0, 0.0))

    def test_one_chat_leaves_no_email_capacity(self):
        capacity, used, excess = residual_capacity_usage(
            [(self.start, self.end)], self.email, self.shared, primary_capacity=2
        )
        self.assertEqual((capacity, used, excess), (0.0, 0.0, 3600.0))

    def test_two_chats_leave_no_email_capacity(self):
        capacity, used, excess = residual_capacity_usage(
            [(self.start, self.end), (self.start, self.end)],
            self.email,
            self.shared,
            primary_capacity=2,
        )
        self.assertEqual((capacity, used, excess), (0.0, 0.0, 3600.0))

    def test_email_uses_only_chat_free_shared_time(self):
        midpoint = self.start + pd.Timedelta(minutes=30)
        capacity, used, excess = residual_capacity_usage(
            [(self.start, midpoint)],
            self.email,
            self.shared,
            primary_capacity=2,
        )
        self.assertEqual((capacity, used, excess), (1800.0, 1800.0, 1800.0))

    def test_invalid_chat_capacity_is_rejected(self):
        with self.assertRaises(ValueError):
            residual_capacity_usage([], [], self.shared, primary_capacity=0)


if __name__ == "__main__":
    unittest.main()
