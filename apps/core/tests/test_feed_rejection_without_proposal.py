"""A failed extraction must not trap its pending feed in the queue.

The reject handler has always accepted a pending feed without a proposal,
but the only rendered reject form lived inside the proposal-diff branch.
An extraction failure therefore left the operator with Retry as the only UI
action even though rejection was already a valid state transition.
"""
from __future__ import annotations

import pytest
from django.contrib.auth.models import User
from django.utils import timezone

from apps.feeds.models import Feed

pytestmark = pytest.mark.django_db


@pytest.fixture(autouse=True)
def configured_install(monkeypatch):
    from apps.brainconfig import setup_state

    monkeypatch.setattr(setup_state, "needs_first_admin", lambda: False)
    monkeypatch.setattr(setup_state, "is_complete", lambda: True)


@pytest.fixture
def operator(client):
    user = User.objects.create_user(
        "feed-op", password="x" * 20, is_staff=True, is_superuser=True
    )
    client.force_login(user)
    return user


def _failed_feed(**overrides) -> Feed:
    fields = {
        "source_id": "failed-capture",
        "channel": "ui",
        "status": "pending",
        "raw_payload": {"source_kind": "thought", "content": "captured text"},
        "proposal": None,
        "error": "extraction attempt 3/3 failed: NoStructuredOutput",
    }
    fields.update(overrides)
    return Feed.objects.create(**fields)


def test_failed_feed_without_proposal_renders_reject_control(client, operator) -> None:
    feed = _failed_feed()

    response = client.get(f"/ops/feeds/{feed.pk}/")

    assert response.status_code == 200
    html = response.content.decode()
    assert 'name="action" value="reject"' in html
    assert 'name="reason"' in html


def test_failed_feed_without_proposal_can_be_rejected(client, operator) -> None:
    feed = _failed_feed()

    response = client.post(
        f"/ops/feeds/{feed.pk}/",
        {"action": "reject", "reason": "Extraction failed; discard test capture."},
    )

    assert response.status_code == 302
    feed.refresh_from_db()
    assert feed.status == "rejected"
    assert feed.decision_note == "Extraction failed; discard test capture."
    assert feed.decided_at is not None


def test_feed_cannot_be_rejected_while_extraction_is_in_flight(client, operator) -> None:
    feed = _failed_feed(error="", extract_queued_at=timezone.now())

    response = client.get(f"/ops/feeds/{feed.pk}/")

    assert response.status_code == 200
    assert 'name="action" value="reject"' not in response.content.decode()

    response = client.post(
        f"/ops/feeds/{feed.pk}/",
        {"action": "reject", "reason": "Stale browser submission."},
    )

    assert response.status_code == 302
    feed.refresh_from_db()
    assert feed.status == "pending"
    assert feed.decision_note == ""
