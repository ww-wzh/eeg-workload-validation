"""Exercise checkpoint rejection without changing any scientific artifact."""
from pathlib import Path
from tempfile import TemporaryDirectory
import shutil
import run_nested as r


def main():
    stage=r.HERE/"jobs"/r.PILOT/"stages/inner_s03_feature_mlp"
    identity=r.load(stage/"complete.json")["identity"]
    assert r.read_stage(stage,identity) is not None
    altered={**identity,"seed":identity["seed"]+1}
    try:
        r.read_stage(stage,altered)
    except AssertionError:
        mismatch_rejected=True
    else:
        raise AssertionError("Mismatched scientific identity was accepted")
    before=r.sha(stage/"predictions.npz")
    with TemporaryDirectory(dir=r.HERE,prefix="checkpoint_test_") as td:
        dest=Path(td)/"copied_stage"
        shutil.copytree(stage,dest)
        with (dest/"predictions.npz").open("ab") as f:
            f.write(b"deliberate-corruption-in-test-copy")
        try:
            r.read_stage(dest,identity)
        except AssertionError:
            corruption_rejected=True
        else:
            raise AssertionError("Corrupt scientific checkpoint was accepted")
        (dest/"complete.json").unlink()
        assert r.read_stage(dest,identity) is None
    assert r.sha(stage/"predictions.npz")==before
    r.dump(r.HERE/"checkpoint_safety_verification.json",{"status":"passed","utc":r.now(),
        "valid_stage_loaded":True,"changed_seed_rejected":mismatch_rejected,
        "corrupt_copied_payload_rejected":corruption_rejected,"uncommitted_stage_not_reused":True,
        "scientific_artifact_unchanged":True,"test_source_sha256":r.sha(Path(__file__))})
    print("Checkpoint identity, integrity and uncommitted-stage tests passed; original artifacts unchanged.")


if __name__=="__main__":
    main()
