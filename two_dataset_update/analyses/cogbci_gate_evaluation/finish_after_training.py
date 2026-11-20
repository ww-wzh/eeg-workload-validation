"""Local continuation: after all fits finish, score and independently verify.

This process never starts another model trainer and never publishes files.
It stops on a new training failure. Re-running it is safe; completed verified
outputs are checked before exit. The computer must remain running.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path
import run_nested as r


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--wait", action="store_true")
    args=parser.parse_args()
    status_path=r.HERE/"continuation_status.json"
    started=time.time()
    freeze={"analysis":r.sha(r.HERE/"analyze_gates.py"),"verifier":r.sha(r.HERE/"verify_results.py")}
    r.dump(r.HERE/"continuation_freeze.json",{"utc":r.now(),"files_sha256":freeze,
        "training_started_separately":True,"remote_writes":False})
    while True:
        status=r.load(r.HERE/"run_status.json") if (r.HERE/"run_status.json").exists() else {}
        if status.get("mode")=="all" and status.get("status")=="selected_jobs_complete":
            break
        failure=r.HERE/"last_failure.json"
        if failure.exists() and failure.stat().st_mtime>=started:
            r.dump(status_path,{"status":"stopped_on_training_error","utc":r.now(),"failure":r.load(failure)})
            return
        r.dump(status_path,{"status":"waiting_for_all_3250_jobs","utc":r.now(),"training_status":status})
        if not args.wait:
            print("Training incomplete; analysis not started.",flush=True)
            return
        time.sleep(30)
    assert r.sha(r.HERE/"analyze_gates.py")==freeze["analysis"]
    assert r.sha(r.HERE/"verify_results.py")==freeze["verifier"]
    complete=r.HERE/"analysis/independent_verification.json"
    if complete.exists() and r.load(complete).get("status")=="passed":
        manifest=r.load(r.HERE/"analysis/verified_outputs_manifest.json")
        for name,h in manifest.items():
            assert r.sha(r.HERE/name)==h
        r.dump(status_path,{"status":"analysis_and_independent_verification_already_complete","utc":r.now()})
        return
    for script,args in [("analyze_gates.py",["analyze"]),("verify_results.py",[])]:
        r.dump(status_path,{"status":"running_postprocessing","script":script,"utc":r.now()})
        log=r.HERE/(script.removesuffix(".py")+".log")
        with log.open("a",encoding="utf-8") as stream:
            result=subprocess.run([sys.executable,"-u",str(r.HERE/script),*args],cwd=r.ROOT,
                                  stdout=stream,stderr=subprocess.STDOUT,check=False)
        if result.returncode:
            r.dump(status_path,{"status":"postprocessing_failed","script":script,"exit_code":result.returncode,
                                "log":str(log),"utc":r.now()})
            raise RuntimeError(f"{script} failed; inspect {log}")
    r.dump(status_path,{"status":"COG_gate_results_verified","utc":r.now(),
        "result_directory":"analysis"})


if __name__=="__main__":
    main()
