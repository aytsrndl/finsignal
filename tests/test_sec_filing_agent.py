import unittest

from pydantic import ValidationError
from agents.sec_filing_agent import TenKAnalysis


class AnalysisSchemaTests(unittest.TestCase):
    def test_structured_findings_round_trip(self):
        payload = {
            'business_summary': 'Example business',
            'major_risks': [{'title': 'Supply risk', 'explanation': 'Depends on one supplier', 'evidence': 'We rely on a single supplier.'}],
            'business_trends': [], 'management_outlook': 'Stable', 'red_flags': [],
        }
        self.assertEqual(TenKAnalysis.model_validate(payload).model_dump(), payload)

    def test_finding_requires_evidence(self):
        with self.assertRaises(ValidationError):
            TenKAnalysis(business_summary='Example', major_risks=[{'title': 'Risk', 'explanation': 'Explanation'}], business_trends=[], management_outlook='Stable', red_flags=[])


if __name__ == '__main__':
    unittest.main()
