import unittest

from base.services.agent.rag.planner import QueryPlanner
from base.services.agent.rag.text_utils import compute_overlap_score, extract_json_object


class RagTextUtilsTests(unittest.TestCase):
    def test_overlap_score_handles_khmer_substring_match(self):
        query = "របៀបដាក់ពាក្យឥណទាន"
        text = "ឯកសារនេះពន្យល់អំពីការរបៀបដាក់ពាក្យឥណទាននៅធនាគារ។"

        score = compute_overlap_score(query, text)

        self.assertGreater(score, 0.5)

    def test_overlap_score_handles_english_keyword_match(self):
        query = "loan policy 2024"
        text = "This loan policy applies for 2024 and later updates."

        score = compute_overlap_score(query, text)

        self.assertGreater(score, 0.7)

    def test_extract_json_object_from_fenced_response(self):
        raw = """```json
{"intent":"lookup","sub_queries":[],"needs_table":false}
```"""

        parsed = extract_json_object(raw)

        self.assertEqual(parsed["intent"], "lookup")
        self.assertEqual(parsed["sub_queries"], [])
        self.assertFalse(parsed["needs_table"])

    def test_extract_json_object_from_pythonish_response(self):
        raw = "Result: {'intent': 'lookup', 'sub_queries': [], 'needs_table': False}"

        parsed = extract_json_object(raw)

        self.assertEqual(parsed["intent"], "lookup")
        self.assertEqual(parsed["sub_queries"], [])
        self.assertFalse(parsed["needs_table"])

    def test_dedupe_sub_queries_preserves_order(self):
        items = ["Loan policy", "loan policy", "  sick leave  ", "sick leave"]

        deduped = QueryPlanner._dedupe_sub_queries(items)

        self.assertEqual(deduped, ["Loan policy", "sick leave"])
