from __future__ import annotations

import unittest

from codex_agent.platform.conversation_context import (
    ContextPolicy,
    ConversationContextManager,
    ExtractiveConversationSummarizer,
    SummaryResult,
)


def message(number: int, role: str, content: str, **metadata: object) -> dict:
    return {
        "id": f"m{number}",
        "role": role,
        "content": content,
        "metadata": metadata,
    }


class RecordingSummarizer:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls: list[list[str]] = []
        self.fail = fail

    def summarize(self, previous_summary, messages, anchors):
        self.calls.append([item["id"] for item in messages])
        if self.fail:
            raise RuntimeError("offline")
        return SummaryResult(
            "Objective\n- repair operator\nConstraints\n- keep tests locked",
            input_tokens=30,
            output_tokens=12,
            provider="fake-model",
        )


class ConversationContextTests(unittest.TestCase):
    def policy(self) -> ContextPolicy:
        return ContextPolicy(
            max_input_tokens=400,
            safety_ratio=0.75,
            output_reserve_tokens=50,
            recent_turns=1,
            max_tool_result_tokens=50,
            max_rag_tokens=50,
        )

    def test_short_conversation_remains_verbatim(self) -> None:
        manager = ConversationContextManager(self.policy(), token_counter=len)
        bundle = manager.build(
            current_request="continue",
            messages=[message(1, "user", "inspect relu"), message(2, "assistant", "ready")],
        )

        self.assertFalse(bundle.compacted)
        self.assertIn("inspect relu", bundle.prompt)
        self.assertIn("ready", bundle.prompt)

    def test_long_history_is_summarized_but_recent_turns_remain_raw(self) -> None:
        summarizer = RecordingSummarizer()
        manager = ConversationContextManager(
            self.policy(), summarizer=summarizer, token_counter=len
        )
        messages = [
            message(1, "user", "old " * 180),
            message(2, "assistant", "run-123 failed at /tmp/a.log " * 60),
            message(3, "user", "recent question"),
            message(4, "assistant", "recent answer"),
        ]
        bundle = manager.build(current_request="continue", messages=messages)

        self.assertTrue(bundle.compacted)
        self.assertEqual(summarizer.calls, [["m1", "m2"]])
        self.assertEqual(bundle.summarized_message_ids, ("m1", "m2"))
        self.assertIn("recent question", bundle.prompt)
        self.assertIn("recent answer", bundle.prompt)
        self.assertIn("run-123", bundle.summary)
        self.assertIn("/tmp/a.log", bundle.summary)
        self.assertIn("run-123", bundle.prompt)
        self.assertIn("/tmp/a.log", bundle.prompt)
        self.assertLessEqual(bundle.estimated_tokens, bundle.input_budget + len("\n[truncated to context budget]"))

    def test_chinese_and_unbroken_text_are_bounded(self) -> None:
        manager = ConversationContextManager(self.policy(), token_counter=len)
        messages = [
            message(1, "user", "错误" * 300),
            message(2, "assistant", "A" * 600),
            message(3, "user", "保留近期问题"),
            message(4, "assistant", "保留近期回答"),
        ]
        bundle = manager.build(
            current_request="继续",
            messages=messages,
            tool_results="B" * 600,
            rag_context="C" * 600,
        )

        self.assertTrue(bundle.compacted)
        self.assertIn("truncated to context budget", bundle.prompt)

    def test_model_summary_failure_falls_back_without_losing_anchors(self) -> None:
        manager = ConversationContextManager(
            self.policy(), summarizer=RecordingSummarizer(fail=True), token_counter=len
        )
        bundle = manager.build(
            current_request="continue",
            messages=[
                message(1, "user", "proposal-abc is approved " * 30),
                message(2, "assistant", "python -m pytest test_x.py failed " * 20),
                message(3, "user", "recent"),
                message(4, "assistant", "answer"),
            ],
        )

        self.assertTrue(bundle.compacted)
        self.assertEqual(bundle.summary_usage["provider"], "deterministic-fallback")
        self.assertIn("proposal-abc", bundle.summary)
        self.assertIn("test_x.py", bundle.summary)

    def test_plan_boilerplate_is_not_sent_back_to_model(self) -> None:
        manager = ConversationContextManager(self.policy(), token_counter=len)
        bundle = manager.build(
            current_request="validate",
            messages=[
                message(1, "assistant", "confirm this", view="harness-plan"),
                message(2, "user", "real evidence"),
            ],
        )
        self.assertNotIn("confirm this", bundle.prompt)
        self.assertIn("real evidence", bundle.prompt)


if __name__ == "__main__":
    unittest.main()
