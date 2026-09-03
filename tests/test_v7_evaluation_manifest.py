import pytest
from photocut.algorithms.v7.evaluation_manifest import validate_public_manifest, FrozenManifestError, build_frozen_capacity_attestation, validate_attestation, make_candidate_seal, validate_candidate_seal

def test_manifest_accepts_basic_sample():
    s={"schema_version":1,"image_id":"i","origin_group_id":"g","split":"train","photo_truth":[[0,0],[9,0],[9,9],[0,9]],"audited_labels":["photo"],"auditor":"a","timestamp":"t","source_annotation_id":"ann","expected_object_hash":"sha256:"+"0"*64}
    assert validate_public_manifest({"schema_version":1,"samples":[s]})[0]["image_id"] == "i"

def test_manifest_rejects_duplicate():
    s={"schema_version":1,"image_id":"i","origin_group_id":"g","split":"train","photo_truth":[[0,0],[9,0],[9,9],[0,9]],"audited_labels":["photo"],"auditor":"a","timestamp":"t","source_annotation_id":"ann","expected_object_hash":"sha256:"+"0"*64}
    with pytest.raises(FrozenManifestError): validate_public_manifest({"schema_version":1,"samples":[s,s]})

def test_attestation_rejects_tampered_slice_counts():
    s={"origin_group_id":"g","audited_labels":{"slices":["outer"]}}
    a=build_frozen_capacity_attestation("sha256:"+"0"*64,[s],"prep")
    a["slice_counts"]={"outer":99}
    with pytest.raises(FrozenManifestError): validate_attestation(a,[s])
    a["schema_version"] = 2
    with pytest.raises(FrozenManifestError): validate_attestation(a,[s])


def test_manifest_rejects_missing_photo_truth_audit_identity_and_unknown_fields():
    s={"schema_version":1,"image_id":"i","origin_group_id":"g","split":"train","photo_truth":None,"audited_labels":["photo"],"auditor":"a","timestamp":"t","source_annotation_id":"ann","expected_object_hash":"sha256:"+"0"*64}
    with pytest.raises(FrozenManifestError): validate_public_manifest({"schema_version":1,"samples":[s]})
    s["photo_truth"]=[[0,0],[9,0],[9,9],[0,9]]; s["auditor"]=""
    with pytest.raises(FrozenManifestError): validate_public_manifest({"schema_version":1,"samples":[s]})
    s["auditor"]="a"; s["unexpected"] = 1
    with pytest.raises(FrozenManifestError): validate_public_manifest({"schema_version":1,"samples":[s]})


def test_candidate_seal_normalizes_aliases_without_forbidden_output_fields():
    h="sha256:"+"0"*64
    seal=make_candidate_seal(candidate_id="c",git_commit="g",parameter_json={"x":1},analysis_version="a",public_manifest_hash=h,frozen_vault_hash=h,split_hash=h,environment_fingerprint="e",powered_sample_requirements={},validation_gate_results={},final_benchmark_manifest_hash=h,final_benchmark_result_hash=h,final_benchmark_pass_state=True,timestamp="t")
    assert set(seal) == {"schema_version","candidate_id","git_commit","parameter_json","parameter_hash","analysis_version","analysis_hash","public_manifest_hash","frozen_attestation_hash","split_hash","environment_fingerprint","powered_sample_requirements","gate_results","final_benchmark_manifest_hash","final_benchmark_result_hash","final_benchmark_pass_state","seal_timestamp"}
    with pytest.raises(FrozenManifestError): validate_candidate_seal({**seal, "schema_version": 2}, seal)


@pytest.mark.parametrize("bad_seal", [None, {}, {"schema_version": 2}, {"schema_version": 1, "foo": "bar"}])
def test_public_manifest_rejects_invalid_candidate_seal(bad_seal):
    s={"schema_version":1,"image_id":"i","origin_group_id":"g","split":"train","photo_truth":[[0,0],[9,0],[9,9],[0,9]],"audited_labels":["photo"],"auditor":"a","timestamp":"t","source_annotation_id":"ann","expected_object_hash":"sha256:"+"0"*64}
    with pytest.raises(FrozenManifestError): validate_public_manifest({"schema_version":1,"candidate_seal":bad_seal,"samples":[s]})


@pytest.mark.parametrize("schema_version", [0, 2, "1", True, None])
def test_manifest_and_sample_schema_versions_require_exact_integer_one(schema_version):
    s={"schema_version":1,"image_id":"i","origin_group_id":"g","split":"train","photo_truth":[[0,0],[9,0],[9,9],[0,9]],"audited_labels":["photo"],"auditor":"a","timestamp":"t","source_annotation_id":"ann","expected_object_hash":"sha256:"+"0"*64}
    with pytest.raises(FrozenManifestError): validate_public_manifest({"schema_version":schema_version,"samples":[s]})
    s["schema_version"] = schema_version
    with pytest.raises(FrozenManifestError): validate_public_manifest({"schema_version":1,"samples":[s]})
