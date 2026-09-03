import json
import hashlib
import threading
from photocut.algorithms.v7.dataset_index import load_sample_metadata, materialize_sample, assign_splits, IndexedSample, FinalizationToken, FrozenVaultAccess
import pytest

def test_metadata_does_not_materialize_objects(tmp_path):
    obj = tmp_path / "object.bin"; obj.write_bytes(b"secret")
    record = {"schema_version":1,"image_id":"i","origin_group_id":"g","split":"train","photo_truth":[[0,0],[9,0],[9,9],[0,9]],"audited_labels":["photo"],"auditor":"a","timestamp":"t","source_annotation_id":"ann","expected_object_hash":"sha256:"+hashlib.sha256(b"secret").hexdigest()}
    p=tmp_path/"manifest.json"; p.write_text(json.dumps({"schema_version":1,"samples":[record]}))
    s=load_sample_metadata(p, image_id="i")
    assert s.image_id == "i" and not s.materialized
    # Object locations are supplied out-of-band by the materializer, never public metadata.
    bound = IndexedSample(s.image_id, s.origin_group_id, s.split, s.metadata, str(obj))
    assert materialize_sample(bound, decoder=lambda b: b) == b"secret"

def test_origin_group_split_is_deterministic_and_immutable():
    a=assign_splits(["g1","g2"], existing={"g1":"train"})
    assert a["g1"] == "train" and a["g2"] in {"train","validation"}


def test_finalization_authorization_is_immutable_and_one_time(tmp_path):
    vault = tmp_path / "vault.json"; vault.write_text("{}")
    token = FinalizationToken.issue(vault)
    with pytest.raises(AttributeError): token.value = "tampered"
    with pytest.raises(AttributeError): token.vault_root = tmp_path
    FrozenVaultAccess(vault, token, token.value)
    with pytest.raises(PermissionError): FrozenVaultAccess(vault, token, token.value)
    with pytest.raises(PermissionError): FinalizationToken("x", vault_root=vault)


def test_finalization_authorization_is_atomic_under_concurrency(tmp_path):
    vault = tmp_path / "vault.json"; vault.write_text("{}")
    token = FinalizationToken.issue(vault)
    outcomes = []
    lock = threading.Lock()
    barrier = threading.Barrier(20)

    def attempt():
        barrier.wait()
        try:
            FrozenVaultAccess(vault, token, token.value)
        except PermissionError:
            with lock: outcomes.append(False)
        else:
            with lock: outcomes.append(True)

    threads = [threading.Thread(target=attempt) for _ in range(20)]
    for thread in threads: thread.start()
    for thread in threads: thread.join()
    assert outcomes.count(True) == 1
