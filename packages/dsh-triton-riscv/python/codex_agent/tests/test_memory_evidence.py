import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from codex_agent.memory_evidence import EvidenceSource, assemble_receipt_evidence, collect_lifecycle_sources
from codex_agent.diagnostic_memory import record_from_validation_receipt, remember_validation, retrieve_memories, _public_item
from codex_agent.memory import MemoryStore, records_from_run, render_memory_context
from codex_agent.memory_view import bounded_chain, encode


def fixture():
    receipt = {"run_id":"run-a","operator":"demo","status":"failed","dry_run":False,
               "exit_code":1,"failure_stage":"mlir-translate","command":"pytest test_demo.py",
               "error_excerpt":["linalg.generic failed"],"log_path":"agent-results/logs/a.log",
               "remote_preflight":{"architecture":"riscv64","triton_version":"3.4.0"},
               "diagnosis":{"confidence":0.8,"recommended_actions":["try arithmetic"]}}
    proposal = {"proposal_id":"repair-a","run_id":"run-a","operator":"demo",
                "status":"applied","applied_at":"2026-09-01T02:00:00+00:00",
                "rationale":"avoid unsupported primitive", "diff":"@@ -1 +1 @@\n-old\n+new"}
    sources = [EvidenceSource("agent-results/receipts/a.json",json.dumps(receipt),available_at="2026-09-01T01:00:00+00:00"),
               EvidenceSource("agent-results/proposals/repair-a.json",json.dumps(proposal),available_at="2026-09-01T02:00:00+00:00"),
               EvidenceSource("agent-results/logs/a.log","error: linalg.generic\n1 failed in 0.1s",available_at="2026-09-01T01:00:00+00:00")]
    return receipt, sources


class EvidenceTests(unittest.TestCase):
    def test_explicit_links_not_same_operator_or_adjacent_file(self):
        r,s = fixture()
        unrelated = json.loads(s[1].content); unrelated["run_id"]="other"
        chain = assemble_receipt_evidence(r, [s[0],EvidenceSource(s[1].path,json.dumps(unrelated)),s[2]])
        self.assertFalse(any(e["kind"] in {"action","patch"} for e in chain["items"]))
        self.assertTrue(any(e["kind"]=="log" for e in chain["items"]))

    def test_state_and_version_not_inferred_from_pass_or_recommendation(self):
        r,s = fixture()
        chain = assemble_receipt_evidence(r,s)
        self.assertEqual(chain["repair_causality"],"not-established")
        self.assertEqual(chain["remote_version_binding"],"unknown")
        self.assertEqual(next(e for e in chain["items"] if e["kind"]=="recommendation")["state"],"not-executed")
        self.assertTrue(any(g["reason"]=="no-explicit-followup-validation-binding" for g in chain["gaps"]))
        proposal = json.loads(s[1].content); proposal["status"]="approved"
        chain = assemble_receipt_evidence(r,[s[0],EvidenceSource(s[1].path,json.dumps(proposal))])
        self.assertTrue(all(e["state"]!="host-recorded-applied" for e in chain["items"]))

    def test_asof_checks_each_source_not_only_receipt(self):
        r,s = fixture()
        cutoff="2026-09-01T01:30:00+00:00"
        chain=assemble_receipt_evidence(r,s,as_of=cutoff)
        self.assertFalse(any(e["kind"]=="patch" for e in chain["items"]))
        self.assertEqual(len(chain["gaps"]),1)
        self.assertIsNone(record_from_validation_receipt(r,sources=s,as_of="2026-08-01T00:00:00+00:00"))
        with self.assertRaises(ValueError):
            assemble_receipt_evidence(r,s,as_of="not-a-date")

    def test_planned_rejected_even_when_source_contains_patch(self):
        r,s=fixture(); r["status"]="planned"
        self.assertIsNone(record_from_validation_receipt(r,sources=s))

    def test_diagnosis_fallback_has_real_source_pointer(self):
        r,s=fixture();r.pop('diagnosis');r['likely_reason']='unsupported operation'
        chain=assemble_receipt_evidence(r,[EvidenceSource('receipt.json',json.dumps(r))])
        entry=next(e for e in chain['items'] if e['kind']=='diagnosis')
        self.assertEqual(entry['pointer'],'/likely_reason')
        self.assertEqual(entry['text'],r['likely_reason'])

    def test_environment_conflicts_retained_not_guessed(self):
        r,s=fixture(); r["architecture"]="x86_64"
        s[0]=EvidenceSource(s[0].path,json.dumps(r))
        record=record_from_validation_receipt(r,sources=s)
        self.assertIsNone(record.environment["architecture"])
        self.assertEqual(record.environment["triton"],"3.4.0")
        self.assertTrue(record.evidence["chain"]["conflicts"])
        self.assertTrue(any(e["kind"]=="conflict" for e in record.evidence["chain"]["items"]))

    def test_duplicate_and_changed_source_replace_chunks_without_cross_run_merge(self):
        r,s=fixture()
        with tempfile.TemporaryDirectory() as temp, MemoryStore(Path(temp)/"m.db") as store:
            first=record_from_validation_receipt(r,sources=s)
            mid,created=store.add(first); self.assertTrue(created)
            self.assertEqual(store.add(first),(mid,False))
            changed=json.loads(s[1].content); changed["diff"]="@@ -1 +1 @@\n-old\n+newer"
            updated=record_from_validation_receipt(r,sources=[s[0],EvidenceSource(s[1].path,json.dumps(changed))])
            self.assertEqual(store.add(updated),(mid,False))
            chunks=[dict(c) for c in store.connection.execute("SELECT * FROM memory_chunks")]
            self.assertFalse(any("+new\n" in c["text"] for c in chunks))
            r2={**r,"run_id":"run-b"}
            store.add(record_from_validation_receipt(r2,sources=[EvidenceSource("b.json",json.dumps(r2))]))
            self.assertEqual(len(store.list()),2)

    def test_long_unicode_and_second_projection_stay_bounded(self):
        r,s=fixture(); obj=json.loads(s[1].content); obj["diff"]="中"*20000
        record=record_from_validation_receipt(r,sources=[s[0],EvidenceSource(s[1].path,json.dumps(obj))])
        with tempfile.TemporaryDirectory() as temp, MemoryStore(Path(temp)/"m.db") as store:
            store.add(record); raw=store.list()[0]
        public=_public_item(raw)
        self.assertLessEqual(len(encode(public)),public["output_budget_chars"])
        self.assertTrue(public["output_truncated"])
        again=bounded_chain(public,max_chars=600)
        self.assertTrue(again["truncated"])
        self.assertLessEqual(len(encode(again)),600)
        for budget in (0,10,256,1000,6000):
            rendered=render_memory_context([public],max_chars=budget)
            self.assertLessEqual(len(rendered),budget)
        self.assertIn("truncated",render_memory_context([public],max_chars=1000))

    def test_real_public_model_and_context_retain_binding(self):
        from codex_agent.operator_lifecycle import MemoryRetrievalToolResult
        r,s=fixture()
        with tempfile.TemporaryDirectory() as temp:
            db=Path(temp)/"m.db"
            with MemoryStore(db) as store:
                store.add(record_from_validation_receipt(r,sources=s))
            with patch.dict("os.environ", {"TRITON_RISCV_MEMORY_DB":str(db),"TRITON_RISCV_EMBEDDING_PROVIDER":"none","TRITON_RISCV_MEMORY_RETRIEVAL_MODE":"legacy"}):
                result=retrieve_memories(Path(temp),operator_name="demo",failure_stage="mlir-translate")
            typed=MemoryRetrievalToolResult.model_validate(result).model_dump()
            self.assertTrue(typed["items"][0]["evidence_chain"]["items"])
            for key in ("run-a","repair-a","linalg.generic","host-recorded-applied","+new"):
                self.assertIn(key,typed["context_excerpt"])

    def test_missing_bad_sources_safe_and_path_escape_not_read(self):
        r,s=fixture(); r["log_path"]="/etc/passwd"
        with tempfile.TemporaryDirectory() as temp:
            sources=collect_lifecycle_sources(Path(temp),r)
            self.assertEqual(len(sources),1)
            chain=assemble_receipt_evidence(r,sources+[EvidenceSource("broken.json","not json")])
            self.assertTrue(any(g["reason"]=="referenced-log-unavailable" for g in chain["gaps"]))
            with patch.dict("os.environ", {"TRITON_RISCV_MEMORY_DB":str(Path(temp)/"m.db"),"TRITON_RISCV_EMBEDDING_PROVIDER":"none"}):
                self.assertEqual(remember_validation(Path(temp),r)["status"],"recorded")

    def test_public_second_cut_marks_patch_and_bad_confidence_degrades(self):
        r,s=fixture();r['diagnosis']={'confidence':'bad','evidence':None,'recommended_actions':None}
        record=record_from_validation_receipt(r)
        self.assertEqual(record.confidence_grade,'D')
        with tempfile.TemporaryDirectory() as temp, MemoryStore(Path(temp)/'m.db') as store:
            from dataclasses import replace
            store.add(replace(record,evidence={'patch_excerpt':'x'*900,'test_summary':{'unexpected':'中'*20000}}))
            public=_public_item(store.list()[0])
            self.assertTrue(public['patch_truncated'])
            self.assertLessEqual(len(encode(public)),public['output_budget_chars'])

    def test_development_patch_without_history_does_not_pair_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root/"operator-spec.json").write_text(json.dumps({"name":"demo","semantics":"x*x"}))
            (root/"final-result.json").write_text(json.dumps({"operator":"demo","status":"passed","validations":[{"status":"passed","iteration":2,"exit_code":0}]}))
            (root/"repair-1.patch").write_text("@@ -1 +1 @@\n-a\n+b")
            record=records_from_run(root)[0]
            patch_item=next(e for e in record.evidence["chain"]["items"] if e["kind"]=="patch")
            self.assertEqual(patch_item["state"],"saved-patch-application-unknown")
            self.assertEqual(patch_item["attempt"],1)
            self.assertEqual(record.memory_type,"successful-run")

    def test_two_repairs_keep_distinct_attempts_and_candidate_validations(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root/"operator-spec.json").write_text(json.dumps({"name":"demo"}))
            final={"operator":"demo","status":"passed", "validations":[
                {"iteration":2,"status":"failed","error_excerpt":["first"],"test_summary":"1 failed"},
                {"iteration":3,"status":"passed","test_summary":"1 passed"}],
                "repair_history":[{"attempt":1,"accepted":False,"outcome":"failed","candidate_validation_iteration":2},
                                  {"attempt":2,"accepted":True,"outcome":"passed","candidate_validation_iteration":3}]}
            (root/"final-result.json").write_text(json.dumps(final))
            for n in (1,2): (root/f"repair-{n}.patch").write_text(f"@@ -1 +1 @@\n-old\n+attempt{n}")
            repairs=[r for r in records_from_run(root) if r.memory_type.endswith("repair")]
            self.assertEqual(len(repairs),2)
            for r in repairs:
                n=r.evidence['attempt']; chain=r.evidence['chain']
                self.assertEqual([e['attempt'] for e in chain['items'] if e['kind']=='patch'],[n])
                self.assertEqual([e['attempt'] for e in chain['items'] if e['kind']=='validation'],[n+1])
            self.assertNotEqual(repairs[0].evidence['chain']['identity'],repairs[1].evidence['chain']['identity'])


if __name__ == "__main__":
    unittest.main()
