"""Tests for how the runner records an incoming message in session history."""

from __future__ import annotations

from omnigent.inner.native_attachments import framework_notice_block, text_notice_block
from omnigent.runner.app import _history_message_from_body


def test_history_drops_one_shot_text_notices() -> None:
    """Policy context is for the turn that carries it, not replayed on later turns."""
    resize = framework_notice_block({"width": 6000, "height": 4000})
    message = _history_message_from_body(
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "how do I deploy"},
                resize,
                text_notice_block("See notes/deploy.md"),
            ],
            "persisted_item_id": "item_1",
        }
    )
    assert message == {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "how do I deploy"}, resize],
        "id": "item_1",
    }
