"""Inspect failed covariance symmetry checks without changing historical caches."""
import hashlib
import json
import zipfile
import numpy as np
import torch
from scipy.signal import butter, iirnotch, resample_poly, sosfiltfilt, tf2sos
from threadpoolctl import threadpool_limits
from prepare_protocol import HERE, PARENT, ARCHIVES, load, BANDS, VariantConfig, extract_features


def main():
    torch.set_num_threads(1)
    records=[r for r in load(HERE/'record_selection.json')['records'] if r['subject']==2]
    b,a=iirnotch(50/(250/2),30)
    notch=tf2sos(b,a)
    band=butter(4,[.5/(250/2),55/(250/2)],btype='band',output='sos')
    rows=[]
    with zipfile.ZipFile(ARCHIVES/'sub-02.zip') as z,threadpool_limits(limits=1):
        for r in records:
            raw=z.read(r['fdt_member'])
            assert hashlib.sha256(raw).hexdigest()==r['fdt_sha256']
            signal=np.frombuffer(raw,dtype='<f4').reshape(r['pnts'],r['nbchan'])[:,r['selected_channel_indices']]
            with np.load(PARENT/'features'/(r['record_id']+'.npz'),allow_pickle=False) as d:
                ii=np.array(r['selected_cache_indices'])
                supports=d['raw_supports'][ii]
            for tile,(start,stop) in enumerate(supports):
                down=resample_poly(signal[start:stop],1,2,axis=0).astype(np.float32)
                filtered=sosfiltfilt(notch,down,axis=0).astype(np.float32)
                _,cov=extract_features(filtered,sfreq=250,win_sec=8.5,stride_sec=.5,bands=BANDS,
                    config=VariantConfig('neutral','none',1,1,1),device=torch.device('cpu'),chunk_size=16)
                c32=cov[4].astype(np.float64)
                central=sosfiltfilt(band,filtered,axis=0).astype(np.float32)[500:2625].T.astype(np.float64)
                c64=central@central.T/2125+1e-4*np.eye(8)
                scale=np.linalg.norm(c64,'fro')
                sym=np.allclose(c32,c32.T,atol=1e-5,rtol=1e-6)
                rows.append({'record_id':r['record_id'],'tile':tile,'old_elementwise_symmetry_pass':bool(sym),
                    'float32_max_abs_asymmetry':float(np.abs(c32-c32.T).max()),
                    'float32_relative_asymmetry_fro':float(np.linalg.norm(c32-c32.T,'fro')/scale),
                    'float32_vs_float64_relative_error_fro':float(np.linalg.norm(c32-c64,'fro')/scale),
                    'float64_max_abs_asymmetry':float(np.abs(c64-c64.T).max()),
                    'matrix_fro_norm':float(scale),'min_eigenvalue_float64':float(np.linalg.eigvalsh(c64).min())})
    result={'scope':'All selected subject02 supports, selected by ID before inspecting diagnostics',
        'n_windows':len(rows),'failed_old_symmetry_checks':sum(not r['old_elementwise_symmetry_pass'] for r in rows),
        'max_relative_float32_error':max(r['float32_vs_float64_relative_error_fro'] for r in rows),
        'max_relative_float32_asymmetry':max(r['float32_relative_asymmetry_fro'] for r in rows),
        'max_absolute_float64_asymmetry':max(r['float64_max_abs_asymmetry'] for r in rows),'windows':rows}
    (HERE/'covariance_precision_diagnostic.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='windows'},indent=2))


if __name__=='__main__':main()
