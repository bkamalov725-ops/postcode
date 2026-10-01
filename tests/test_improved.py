import json
from pathlib import Path
import tempfile
import unittest
from run_improved import choose_winner
from postcode_ml.evaluation import refined_candidate_specs


class ImprovedTests(unittest.TestCase):
    def test_search_includes_verified_winner_and_original(self):
        specs = refined_candidate_specs(4608)
        self.assertEqual(len(specs), len({s['name'] for s in specs}))
        self.assertTrue(any(s.get('C')==300 and s.get('gamma')==.25/4608 and s.get('epsilon')==1 for s in specs))
        self.assertTrue(any(s.get('C')==100 and s.get('gamma')=='scale' and s.get('epsilon')==2 for s in specs))

    def test_selects_only_comparable_complete_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            folders=[]
            for size,mae in [(392,8.1),(518,8.4)]:
                folder=Path(tmp)/str(size); folder.mkdir(); folders.append(folder)
                report={'status':'complete','selection':{'metrics':{'mae':mae}},'folds':{'exported_sha256':'same'}}
                (folder/'report.json').write_text(json.dumps(report))
            self.assertEqual(choose_winner(folders)[0]['directory'],'392')
            report['folds']['exported_sha256']='different'
            (folder/'report.json').write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,'different validation'):
                choose_winner(folders)
            report['status']='failed'
            (folder/'report.json').write_text(json.dumps(report))
            with self.assertRaisesRegex(RuntimeError,'Incomplete'):
                choose_winner(folders)


if __name__=='__main__':
    unittest.main()
