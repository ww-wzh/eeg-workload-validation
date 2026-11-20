"""Independently confirm selected cross-ID duplicates byte for byte."""
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from audit_inputs import HERE, ARCHIVES, dump

def load(p):return json.loads(p.read_text(encoding='utf-8'))

def main():
    audit=load(HERE/'feature_audit.json')
    rows={r['record_id']:r for r in audit['records']}
    official=load(HERE/'official_duplicate_source_checks.json')
    checked=[]
    for f in official['files']:
        subject=int(f['key'][4:6]); report=load(HERE/'features'/f'sub-{subject:02d}_complete.json')
        assert report['archive_md5']==f['checksum'].split(':')[1]
        assert (ARCHIVES/f['key']).stat().st_size==f['size']
    for group in audit['duplicate_selected_FDT_groups']:
        a,b=map(rows.get,group)
        with zipfile.ZipFile(ARCHIVES/f"sub-{a['subject']:02d}.zip") as za,zipfile.ZipFile(ARCHIVES/f"sub-{b['subject']:02d}.zip") as zb:
            ha,hb=hashlib.sha256(),hashlib.sha256();size=0;equal=True
            with za.open(a['fdt_member']) as fa,zb.open(b['fdt_member']) as fb:
                while True:
                    ba,bb=fa.read(1024*1024),fb.read(1024*1024)
                    if not ba and not bb:break
                    equal=equal and ba==bb;size+=len(ba);ha.update(ba);hb.update(bb)
            assert equal and ha.hexdigest()==hb.hexdigest()==a['fdt_sha256']==b['fdt_sha256']
            set_equal=za.read(a['set_member'])==zb.read(b['set_member'])
        checked.append({'records':group,'fdt_bytes':size,'fdt_sha256':ha.hexdigest(),'fdt_byte_for_byte_equal':equal,'set_files_byte_for_byte_equal':set_equal})
    result={'completed_utc':datetime.now(timezone.utc).isoformat(),'groups':checked,
            'n_pairs':len(checked),'n_file_instances':sum(len(g['records']) for g in checked),
            'affected_participant_labels':sorted({rows[r]['subject'] for g in checked for r in g['records']}),
            'official_archive_checksums_match':True,
            'scope':'All channels and samples of the 348 selected condition FDT files were hashed; discovered pairs independently streamed and byte-compared. Other tasks/files in the dataset were not fully audited. Cannot determine correct participant identity or source of duplication.'}
    dump(HERE/'duplicate_provenance_audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='groups'},indent=2))

if __name__=='__main__':main()
