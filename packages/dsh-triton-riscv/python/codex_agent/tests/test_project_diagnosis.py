import tempfile
from pathlib import Path
import unittest
from codex_agent.project_diagnosis import *

class ProjectDiagnosisTests(unittest.TestCase):
 def test_normalize_and_group(self):
  a={'test':'x::test','source_file':'python/foo.py','log':'/tmp/tmpabc/foo.py:12 AssertionError expected 1'}
  b={'test':'y::test','source_file':'python/foo.py','log':'/tmp/tmpxyz/foo.py:99 AssertionError expected 1'}
  fs=analyze_failures([a,b]); self.assertEqual(len(fs),1); self.assertEqual(fs[0].occurrences,2)
 def test_plan_marks_compiler_review(self):
  f=ProjectFailure('x','compiler lowering failure','compiler','buddy-opt',1,['t'],['lib/a.cpp'])
  p=plan_repairs([f])[0]; self.assertTrue(p.requires_review); self.assertEqual(p.scope,'project-review')
 def test_report(self):
  with tempfile.TemporaryDirectory() as d:
   p=write_report(Path(d)/'r.json',[],[]); self.assertTrue(p.exists())

if __name__=='__main__': unittest.main()
