"""Retrieve official-source metadata for provenance verification."""
import hashlib
import json
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

HERE=Path(__file__).resolve().parent
OUT=HERE/"official_sources"


def read(url):
    req=urllib.request.Request(url,headers={"User-Agent":"COG-BCI-secondary-analysis-source-verification/1.0"})
    with urllib.request.urlopen(req,timeout=45) as response:
        return response.read()


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    entries=[]
    url="https://zenodo.org/api/records/7413650"
    content=read(url)
    (OUT/"zenodo_7413650.json").write_bytes(content)
    metadata=json.loads(content)["metadata"]
    selected={k:metadata.get(k) for k in ["title","doi","version","license","creators","related_identifiers"]}
    entries.append({"url":url,"local_file":"zenodo_7413650.json","sha256":hashlib.sha256(content).hexdigest()})
    url="https://www.ebi.ac.uk/europepmc/webservices/rest/PMC9918545/fullTextXML"
    content=read(url)
    (OUT/"PMC9918545.xml").write_bytes(content)
    root=ET.fromstring(content)
    passages=[]
    for node in root.iter():
        if node.tag in ["p","license-p"]:
            text=" ".join(" ".join(node.itertext()).split())
            if re.search(r"ethic|consent|institutional review|approved|license|licence",text,re.I):
                passages.append(text)
    entries.append({"url":url,"local_file":"PMC9918545.xml","sha256":hashlib.sha256(content).hexdigest()})
    report={"checked_utc":datetime.now(timezone.utc).isoformat(),"sources":entries,
            "dataset_metadata":selected,"article_ethics_and_license_passages":passages,
            "scope":"Original COG-BCI acquisition/public-source statements only; this is not an institutional determination for the present secondary analysis.",
            "article_doi":"10.1038/s41597-022-01898-y"}
    (OUT/"source_check.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,ensure_ascii=True,indent=2))


if __name__=="__main__":
    main()
