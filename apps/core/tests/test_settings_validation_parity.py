"""/ops/settings/ must not accept what the wizard would refuse.

The wizard normalises a pasted repo (`myname/brain` →
`git@github.com:myname/brain.git`, browser URLs → ssh) and rejects
garbage with a usable message. /ops/settings/ stored the same paste
verbatim: the clone check went red and the offered repair failed
against a URL git cannot parse — strictly worse than being told at
save time.

Mechanism under test: `SettingSpec.clean`, enforced in the settings
page's save path. For the repo URL the spec points at the wizard's own
`normalise_repo_input`, so the two surfaces cannot drift. Numeric keys
get save-time rejection too — the read-side accessors already fall
back safely (test_numeric_settings_totality), but an operator who
typed "nan" should hear "not saved", not silently run on the default.
"""
from __future__ import annotations

import pytest
from django.contrib.auth.models import User

from apps.brainconfig import setup_services
from apps.brainconfig.models import AppSetting
from apps.brainconfig.services import REGISTRY, is_db_set

pytestmark = pytest.mark.django_db


@pytest.fixture
def operator(client):
    user = User.objects.create_user(
        "validation-op", password="x" * 20, is_staff=True, is_superuser=True
    )
    client.force_login(user)
    return user


def _save(client, key: str, value: str):
    return client.post("/ops/settings/", {f"value__{key}": value, "action": "save"})


def _stored(key: str) -> str:
    row = AppSetting.objects.filter(key=key).first()
    return row.value if row else ""


def test_repo_shorthand_normalises_exactly_like_the_wizard(client, operator) -> None:
    _save(client, "BRAIN_REPO_URL", "myname/brain")
    assert _stored("BRAIN_REPO_URL") == setup_services.normalise_repo_input("myname/brain")
    assert _stored("BRAIN_REPO_URL") == "git@github.com:myname/brain.git"


def test_the_repo_cleaner_is_the_wizard_function_itself(client, operator) -> None:
    """Parity by construction, not by parallel implementations: a browser
    URL with a branch path — the wizard's hardest case — must come out
    identical on both surfaces."""
    pasted = "https://github.com/myname/brain/tree/main"
    _save(client, "BRAIN_REPO_URL", pasted)
    assert _stored("BRAIN_REPO_URL") == setup_services.normalise_repo_input(pasted)


def test_garbage_repo_input_is_rejected_not_stored(client, operator) -> None:
    _save(client, "BRAIN_REPO_URL", "not a repository at all")
    assert not is_db_set("BRAIN_REPO_URL")


def test_nan_cap_is_rejected_at_the_door(client, operator) -> None:
    _save(client, "DAILY_COST_CAP", "nan")
    assert not is_db_set("DAILY_COST_CAP")


def test_non_numeric_turns_are_rejected_and_numeric_stored(client, operator) -> None:
    _save(client, "MAX_TURNS_READER", "many")
    assert not is_db_set("MAX_TURNS_READER")
    _save(client, "MAX_TURNS_READER", "12")
    assert _stored("MAX_TURNS_READER") == "12"


def test_every_wizard_validated_key_carries_a_cleaner() -> None:
    """Structural: the keys whose values some later surface parses must
    declare a cleaner, so a new numeric knob can't ship store-as-typed
    by forgetting this file exists."""
    cleaners = {s.key: s.clean for s in REGISTRY}
    for key in (
        "BRAIN_REPO_URL",
        "DAILY_COST_CAP",
        "MAX_BUDGET_USD_READER",
        "MAX_BUDGET_USD_FEEDER",
        "MAX_TURNS_READER",
        "MAX_TURNS_FEEDER",
        "SDK_TIMEOUT_SECONDS",
        "OPENROUTER_FALLBACK_PRICES",
    ):
        assert cleaners.get(key) is not None, f"{key} stores as typed"
