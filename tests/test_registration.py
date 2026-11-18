import io
import json

import pytest

from mbtopology import registration
from mbtopology.runtime import Fail


def test_public_tag_must_match_exact_annotated_object(monkeypatch):
    payload = {
        "ref": registration.TAG_REF,
        "object": {"type": "tag", "sha": "different-object"},
    }
    monkeypatch.setattr(
        registration, "urlopen", lambda *args, **kwargs: io.BytesIO(json.dumps(payload).encode())
    )
    registration.verify_public_tag.cache_clear()
    try:
        with pytest.raises(Fail, match="public annotated tag does not match"):
            registration.verify_public_tag()
        payload["object"]["sha"] = registration.TAG_OBJECT
        registration.verify_public_tag()
    finally:
        registration.verify_public_tag.cache_clear()
