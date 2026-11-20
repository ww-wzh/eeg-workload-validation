"""Combine legacy 65 and new 14 exact-sign-flip contrasts for 79-item sensitivity BH."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
LEGACY=next(HERE.rglob("primary_comparisons_BH65.csv"))
NEW=HERE/"heldout_gain_statistics"/"comparisons_BH14.csv"
OUT=HERE/"heldout_gain_bh79"

def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''): h.update(b)
 return h.hexdigest()
def bh(x):
 x=np.asarray(x,float); o=np.argsort(x,kind='stable'); z=x[o]*len(x)/np.arange(1,len(x)+1); z=np.minimum.accumulate(z[::-1])[::-1]; y=np.empty(len(x)); y[o]=np.clip(z,0,1); return y
def main():
 old=pd.read_csv(LEGACY); new=pd.read_csv(NEW)
 if len(old)!=65 or len(new)!=14: raise AssertionError((len(old),len(new)))
 old2=old.copy(); old2['source_family']='legacy_65'; old2['contrast_label']=old2['setting_display'].astype(str)+' | '+old2['method_display'].astype(str)
 old2['mean_difference_pp']=old2['mean_paired_difference_pp']; old2['ci95_low_pp']=old2['ci95_percentile_low_pp']; old2['ci95_high_pp']=old2['ci95_percentile_high_pp']
 old2['primary_raw_p']=old2['exact_two_sided_sign_flip_p'].astype(float); old2['sensitivity_wilcoxon_raw_p']=old2['wilcoxon_raw_p'].astype(float)
 new2=new.copy(); new2['source_family']='new_heldout_gain_14'; new2['contrast_label']=new2['contrast'].astype(str); new2['primary_raw_p']=new2['exact_sign_flip_p'].astype(float); new2['sensitivity_wilcoxon_raw_p']=new2['wilcoxon_pratt_raw_p'].astype(float)
 cols=['source_family','contrast_label','n_participants','mean_difference_pp','ci95_low_pp','ci95_high_pp','primary_raw_p','sensitivity_wilcoxon_raw_p']
 allc=pd.concat([old2[cols],new2[cols]],ignore_index=True); allc['bh_adjusted_p_exact_79']=bh(allc.primary_raw_p.to_numpy()); allc['bh_adjusted_p_wilcoxon_79_sensitivity']=bh(allc.sensitivity_wilcoxon_raw_p.to_numpy()); allc['exact_bh_significant_0_05']=allc.bh_adjusted_p_exact_79<.05; allc['wilcoxon_bh_significant_0_05']=allc.bh_adjusted_p_wilcoxon_79_sensitivity<.05
 OUT.mkdir(parents=True,exist_ok=True); allc.to_csv(OUT/'comparisons_BH79.csv',index=False,float_format='%.17g')
 summary=pd.DataFrame([{'family':'exact_sign_flip_primary','n_comparisons':79,'raw_p_below_0_05':int((allc.primary_raw_p<.05).sum()),'bh_below_0_05':int(allc.exact_bh_significant_0_05.sum()),'minimum_raw_p':float(allc.primary_raw_p.min()),'minimum_bh_p':float(allc.bh_adjusted_p_exact_79.min())},{'family':'wilcoxon_sensitivity','n_comparisons':79,'raw_p_below_0_05':int((allc.sensitivity_wilcoxon_raw_p<.05).sum()),'bh_below_0_05':int(allc.wilcoxon_bh_significant_0_05.sum()),'minimum_raw_p':float(allc.sensitivity_wilcoxon_raw_p.min()),'minimum_bh_p':float(allc.bh_adjusted_p_wilcoxon_79_sensitivity.min())}]); summary.to_csv(OUT/'summary_BH79.csv',index=False,float_format='%.17g')
 manifest={'status':'passed','n_legacy':65,'n_new':14,'n_total':79,'primary':'exact sign-flip p values in one BH family','sensitivity':'Wilcoxon Pratt p values in a separate BH family','inputs':{str(LEGACY):sha(LEGACY),str(NEW):sha(NEW)}}; (OUT/'189_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
 print('189 79-item unified BH sensitivity completed'); print(summary.to_string(index=False)); print('Output:',OUT)
if __name__=='__main__': main()
