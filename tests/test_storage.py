"""Storage providers and key safety (Phase 3, task 1). No database needed."""

from __future__ import annotations

import io
import uuid

import pytest

from app.storage import (StorageError, StorageKeyError, assert_key_for_tenant, new_upload_id,
                         upload_key)
from app.storage.local import LocalStorage
from app.storage.s3 import S3Storage


def test_keys_are_built_server_side_under_the_tenants_prefix():
    uid = new_upload_id()
    key = upload_key("acme", uid)
    assert key == f"tenants/acme/uploads/{uid}/data.csv"
    with pytest.raises(ValueError):
        upload_key("acme", "../../etc")            # not a UUID
    with pytest.raises(ValueError):
        upload_key("../evil", uid)                 # invalid tenant id


@pytest.mark.parametrize("key", [
    "tenants/acme/uploads/../../other/uploads/x/data.csv", "/etc/passwd", "data.csv",
    f"tenants/acme/uploads/{uuid.uuid4()}/../data.csv", f"tenants/acme/uploads/{uuid.uuid4()}/a/b.csv",
    f"tenants/other/uploads/{uuid.uuid4()}/data.csv", "",
])
def test_malformed_or_foreign_keys_are_refused(key):
    with pytest.raises(StorageKeyError):
        assert_key_for_tenant(key, "acme")


def test_local_round_trip_and_tenant_check(tmp_path):
    store = LocalStorage(tmp_path)
    key = upload_key("acme", new_upload_id())
    assert not store.exists(key, tenant_id="acme")
    store.put(key, io.BytesIO(b"a,b\n1,2\n"), tenant_id="acme")
    assert store.exists(key, tenant_id="acme")
    with store.open(key, tenant_id="acme") as f:
        assert f.read() == b"a,b\n1,2\n"
    assert not [p for p in (tmp_path / key).parent.iterdir() if p.name.startswith(".part")]
    with pytest.raises(StorageKeyError):
        store.open(key, tenant_id="other")
    store.delete(key, tenant_id="acme")
    assert not store.exists(key, tenant_id="acme")
    with pytest.raises(StorageError):
        store.open(key, tenant_id="acme")


class FakeS3:
    """Just enough of boto3's client for S3Storage."""
    def __init__(self): self.objects, self.fail = {}, False
    def upload_fileobj(self, f, bucket, key):
        if self.fail: raise RuntimeError("boom")
        self.objects[(bucket, key)] = f.read()
    def get_object(self, Bucket, Key):
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}
    def head_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects: raise KeyError(Key)
    def delete_object(self, Bucket, Key): self.objects.pop((Bucket, Key), None)
    def head_bucket(self, Bucket): raise KeyError(Bucket)
    def create_bucket(self, Bucket): self.created = Bucket


def test_s3_provider_uses_the_same_keys_and_checks(tmp_path):
    client = FakeS3()
    store = S3Storage(client, "iia-uploads")
    key = upload_key("acme", new_upload_id())
    store.ensure_bucket()
    assert client.created == "iia-uploads"
    store.put(key, io.BytesIO(b"x,y\n"), tenant_id="acme")
    assert ("iia-uploads", key) in client.objects and store.exists(key, tenant_id="acme")
    assert store.open(key, tenant_id="acme").read() == b"x,y\n"
    with pytest.raises(StorageKeyError):
        store.open(key, tenant_id="other")
    client.fail = True
    with pytest.raises(StorageError):
        store.put(key, io.BytesIO(b"z"), tenant_id="acme")
    store.delete(key, tenant_id="acme")
    assert not store.exists(key, tenant_id="acme")
