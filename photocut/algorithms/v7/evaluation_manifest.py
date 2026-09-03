"""Validation and sealing of metadata-only evaluation artifacts."""
from __future__ import annotations
import hashlib, json, math, re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

class FrozenManifestError(ValueError): pass

_PUBLIC_REQUIRED = {"schema_version","image_id","origin_group_id","split","photo_truth","audited_labels","auditor","timestamp","source_annotation_id","expected_object_hash"}
_PUBLIC_TOP_ALLOWED = {"schema_version", "samples", "candidate_seal"}
_PUBLIC_SAMPLE_ALLOWED = _PUBLIC_REQUIRED | {"outer_frame_truth"}
_CANDIDATE_SEAL_FIELDS = {"schema_version","candidate_id","git_commit","parameter_json","parameter_hash","analysis_version","analysis_hash","public_manifest_hash","frozen_attestation_hash","split_hash","environment_fingerprint","powered_sample_requirements","gate_results","final_benchmark_manifest_hash","final_benchmark_result_hash","final_benchmark_pass_state","seal_timestamp"}
_CANDIDATE_HASH_FIELDS = {"parameter_hash","public_manifest_hash","frozen_attestation_hash","split_hash","final_benchmark_manifest_hash","final_benchmark_result_hash"}
_SPLITS = {"train","validation"}
_FROZEN = {"historical_frozen","new_holdout","frozen"}

def _quad(q):
    if q is None: return False
    if not isinstance(q,(list,tuple)) or len(q)!=4: return False
    try: pts=[(float(p[0]),float(p[1])) for p in q]
    except Exception: return False
    if any(not math.isfinite(x) or not math.isfinite(y) for x,y in pts) or len(set(pts)) != 4: return False
    area=sum(pts[i][0]*pts[(i+1)%4][1]-pts[(i+1)%4][0]*pts[i][1] for i in range(4))/2
    if abs(area) < 1e-9: return False
    # A valid ordered quad is strictly convex: all turns have the same sign.
    def cross(a,b,c): return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    turns=[cross(pts[i],pts[(i+1)%4],pts[(i+2)%4]) for i in range(4)]
    return all(t > 1e-9 for t in turns) or all(t < -1e-9 for t in turns)

def _unreviewed(value):
    if isinstance(value, Mapping): return any(_unreviewed(k) or _unreviewed(v) for k,v in value.items())
    if isinstance(value, (list,tuple,set)): return any(_unreviewed(v) for v in value)
    return isinstance(value,str) and value.strip().lower() in {"unreviewed","pending","unknown"}


def _validate_candidate_seal_shape(candidate):
    if not isinstance(candidate, Mapping) or type(candidate.get("schema_version")) is not int or candidate.get("schema_version") != 1:
        raise FrozenManifestError("invalid candidate seal schema")
    if set(candidate) != _CANDIDATE_SEAL_FIELDS:
        raise FrozenManifestError("candidate seal has forbidden fields")
    for field_name in ("candidate_id", "git_commit", "analysis_version", "analysis_hash", "environment_fingerprint", "seal_timestamp"):
        if not isinstance(candidate[field_name], str) or not candidate[field_name]:
            raise FrozenManifestError("invalid seal field: " + field_name)
    for field_name in _CANDIDATE_HASH_FIELDS:
        if not isinstance(candidate[field_name], str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", candidate[field_name]):
            raise FrozenManifestError("invalid seal hash: " + field_name)
    for field_name in ("parameter_json", "powered_sample_requirements", "gate_results"):
        if not isinstance(candidate[field_name], dict):
            raise FrozenManifestError("invalid seal object: " + field_name)
    if candidate["final_benchmark_pass_state"] is not True:
        raise FrozenManifestError("benchmark did not pass")
    return True

def validate_public_manifest(manifest_or_path):
    if isinstance(manifest_or_path,(str,Path)): manifest=json.loads(Path(manifest_or_path).read_text())
    else: manifest=manifest_or_path
    if not isinstance(manifest,Mapping) or type(manifest.get("schema_version")) is not int or manifest.get("schema_version") != 1 or not isinstance(manifest.get("samples"),list): raise FrozenManifestError("invalid public manifest")
    if any(key not in _PUBLIC_TOP_ALLOWED for key in manifest): raise FrozenManifestError("unknown public manifest field")
    if "candidate_seal" in manifest: _validate_candidate_seal_shape(manifest["candidate_seal"])
    seen_i, groups = set(), defaultdict(set); out=[]
    for s in manifest["samples"]:
        if not isinstance(s,Mapping) or not _PUBLIC_REQUIRED.issubset(s): raise FrozenManifestError("missing sample metadata")
        if type(s.get("schema_version")) is not int or s.get("schema_version") != 1: raise FrozenManifestError("invalid sample schema")
        if not isinstance(s.get("split"),str) or s["split"] not in _SPLITS: raise FrozenManifestError("public manifest may contain train/validation only")
        if "object_path" in s: raise FrozenManifestError("public manifest cannot expose object_path")
        if any(key not in _PUBLIC_SAMPLE_ALLOWED for key in s): raise FrozenManifestError("unknown public field")
        if not isinstance(s["image_id"],str) or not isinstance(s["origin_group_id"],str) or not isinstance(s["split"],str) or not s["image_id"] or not s["origin_group_id"]: raise FrozenManifestError("invalid identity")
        for field_name in ("auditor", "timestamp", "source_annotation_id"):
            if not isinstance(s[field_name], str) or not s[field_name].strip(): raise FrozenManifestError("invalid audit identity")
        if s["image_id"] in seen_i: raise FrozenManifestError("duplicate image_id")
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", str(s["expected_object_hash"])): raise FrozenManifestError("invalid object hash")
        if not isinstance(s["audited_labels"], (Mapping,list)): raise FrozenManifestError("invalid labels")
        if not _quad(s.get("photo_truth")): raise FrozenManifestError("invalid photo_truth quad")
        if "outer_frame_truth" in s and s["outer_frame_truth"] is not None and not _quad(s["outer_frame_truth"]): raise FrozenManifestError("invalid outer_frame_truth quad")
        if _unreviewed(s.get("audited_labels")): raise FrozenManifestError("unreviewed hard label")
        cls=s.get("sample_class", "photo")
        seen_i.add(s["image_id"]); groups[s["origin_group_id"]].add((s["split"], cls)); out.append(dict(s))
    if any(len(v)>1 for v in groups.values()): raise FrozenManifestError("origin group crosses split")
    return out

def build_frozen_capacity_attestation(vault_hash: str, samples, preparation_signature: str = ""):
    if not isinstance(vault_hash,str) or not vault_hash.startswith("sha256:") or len(vault_hash) != 71 or not isinstance(preparation_signature,str) or not preparation_signature: raise FrozenManifestError("invalid attestation identity")
    # Deliberately discard all identities: this artifact is public and may only
    # disclose aggregate capacities, never group/sample membership.
    groups={"total": len({str(s["origin_group_id"]) for s in samples})}
    slices=Counter()
    for s in samples:
        labels=s.get("audited_labels", {})
        vals=labels.get("slices",[]) if isinstance(labels,Mapping) else []
        for x in vals: slices[str(x)] += 1
    return {"schema_version":1,"vault_hash":vault_hash,"origin_group_counts":groups,"slice_counts":dict(sorted(slices.items())),"preparation_signature":preparation_signature}

def make_candidate_seal(**kw):
    required_input=("candidate_id","git_commit","parameter_json","analysis_version","public_manifest_hash","split_hash","environment_fingerprint","powered_sample_requirements","final_benchmark_manifest_hash","final_benchmark_result_hash","final_benchmark_pass_state")
    if any(k not in kw or kw[k] in (None,"") for k in required_input): raise FrozenManifestError("incomplete candidate seal")
    if not (kw.get("frozen_vault_hash") or kw.get("frozen_attestation_hash")): raise FrozenManifestError("incomplete candidate seal")
    if not (kw.get("validation_gate_results") is not None or kw.get("gate_results") is not None): raise FrozenManifestError("incomplete candidate seal")
    if not (kw.get("timestamp") or kw.get("seal_timestamp")): raise FrozenManifestError("incomplete candidate seal")
    pj=kw["parameter_json"]; canonical=json.dumps(pj,sort_keys=True,separators=(",",":"),ensure_ascii=False)
    # Normalize legacy input aliases into the exact schema property names. Do
    # not return aliases: candidate seals use additionalProperties=false.
    return {
        "schema_version": 1,
        "candidate_id": kw["candidate_id"], "git_commit": kw["git_commit"],
        "parameter_json": pj, "parameter_hash": "sha256:" + hashlib.sha256(canonical.encode()).hexdigest(),
        "analysis_version": kw["analysis_version"], "analysis_hash": kw.get("analysis_hash", kw["analysis_version"]),
        "public_manifest_hash": kw["public_manifest_hash"],
        "frozen_attestation_hash": kw.get("frozen_attestation_hash", kw.get("frozen_vault_hash")),
        "split_hash": kw["split_hash"], "environment_fingerprint": kw["environment_fingerprint"],
        "powered_sample_requirements": kw["powered_sample_requirements"],
        "gate_results": kw.get("gate_results", kw.get("validation_gate_results")),
        "final_benchmark_manifest_hash": kw["final_benchmark_manifest_hash"],
        "final_benchmark_result_hash": kw["final_benchmark_result_hash"],
        "final_benchmark_pass_state": kw["final_benchmark_pass_state"],
        "seal_timestamp": kw.get("seal_timestamp", kw.get("timestamp")),
    }

def validate_candidate_seal(candidate, benchmark, attestation=None, samples=None):
    required=("candidate_id","git_commit","parameter_json","parameter_hash","analysis_version","analysis_hash","public_manifest_hash","frozen_attestation_hash","split_hash","environment_fingerprint","powered_sample_requirements","gate_results","final_benchmark_manifest_hash","final_benchmark_result_hash","final_benchmark_pass_state","seal_timestamp")
    _validate_candidate_seal_shape(candidate)
    if any(f not in candidate for f in required): raise FrozenManifestError("incomplete candidate seal")
    for f in ("candidate_id","git_commit","analysis_version","analysis_hash","public_manifest_hash","frozen_attestation_hash","split_hash","environment_fingerprint","final_benchmark_manifest_hash","final_benchmark_result_hash","seal_timestamp"):
        if not isinstance(candidate[f],str) or not candidate[f]: raise FrozenManifestError("invalid seal field: "+f)
    for f in ("public_manifest_hash","frozen_attestation_hash","split_hash","final_benchmark_manifest_hash","final_benchmark_result_hash"):
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", candidate[f]): raise FrozenManifestError("invalid hash: "+f)
    if not isinstance(candidate["parameter_json"],dict) or not isinstance(candidate["powered_sample_requirements"],dict) or not isinstance(candidate["gate_results"],dict): raise FrozenManifestError("invalid seal object")
    if candidate["final_benchmark_pass_state"] is not True: raise FrozenManifestError("benchmark did not pass")
    canonical=json.dumps(candidate["parameter_json"],sort_keys=True,separators=(",",":"),ensure_ascii=False)
    if candidate["parameter_hash"] != "sha256:"+hashlib.sha256(canonical.encode()).hexdigest(): raise FrozenManifestError("parameter hash mismatch")
    for f in ("git_commit","parameter_hash","environment_fingerprint","analysis_version","analysis_hash","public_manifest_hash","frozen_attestation_hash","split_hash"):
        if candidate.get(f) != benchmark.get(f): raise FrozenManifestError("benchmark identity mismatch: "+f)
    for f in ("final_benchmark_manifest_hash","final_benchmark_result_hash","final_benchmark_pass_state"):
        if candidate.get(f) != benchmark.get(f): raise FrozenManifestError("benchmark result mismatch: "+f)
    if attestation is not None:
        if attestation.get("vault_hash") != candidate.get("frozen_attestation_hash"): raise FrozenManifestError("attestation hash mismatch")
        if samples is None: raise FrozenManifestError("samples required for attestation")
        validate_attestation(attestation, samples)

def validate_attestation(attestation, samples):
    if not isinstance(attestation, Mapping): raise FrozenManifestError("invalid attestation")
    if type(attestation.get("schema_version")) is not int or attestation.get("schema_version") != 1: raise FrozenManifestError("unsupported attestation schema")
    if set(attestation) != {"schema_version","vault_hash","origin_group_counts","slice_counts","preparation_signature"}: raise FrozenManifestError("attestation has forbidden fields")
    if not isinstance(attestation,Mapping) or not re.fullmatch(r"sha256:[0-9a-f]{64}", str(attestation.get("vault_hash",""))) or not attestation.get("preparation_signature"): raise FrozenManifestError("invalid attestation")
    expected={"total":len({str(s["origin_group_id"]) for s in samples})}
    if attestation.get("origin_group_counts") != expected: raise FrozenManifestError("attestation aggregate mismatch")
    slices=Counter()
    for s in samples:
        labels=s.get("audited_labels", {})
        for sl in (labels.get("slices",[]) if isinstance(labels,Mapping) else []): slices[str(sl)] += 1
    if attestation.get("slice_counts") != dict(sorted(slices.items())): raise FrozenManifestError("attestation slice aggregate mismatch")
    return True
