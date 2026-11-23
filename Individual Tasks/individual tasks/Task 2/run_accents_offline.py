"""Paste this ENTIRE file into one final cell of the already-run Colab notebook.
Uses the loaded Qwen3-ASR model. Does not rerun the 12 baseline recordings.
Adds four VCTK speakers with dataset-labelled English/Scottish accents.
"""
import io, json, time, hashlib, shutil, unicodedata
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import torch
import jiwer
from IPython.display import display

if 'model' not in globals():
    raise RuntimeError('Open the ORIGINAL notebook/session where the model is loaded. If it disconnected, rerun only the install, environment and model-loading cells first, then run this cell.')
ROOT = Path('/content/task2_qwen3_asr')
RESULTS = ROOT / 'results'; SAMPLES = ROOT / 'samples'
if not (RESULTS/'per_sample.csv').exists():
    raise RuntimeError('Baseline results are missing from this session. Upload and extract your saved Task_2_Experiment_Results.zip into /content/task2_qwen3_asr first. Do not fabricate baseline results.')

def normalize(text):
    text = unicodedata.normalize('NFC',str(text)).casefold()
    return ' '.join(''.join(' ' if unicodedata.category(ch)[0] in {'P','S'} else ch for ch in text).split())

OFFLINE_INPUTS = Path('/content/task2_offline_accents')
def get_rows(offset):
    return json.loads((OFFLINE_INPUTS/'metadata.json').read_text())[str(offset)]

# These offsets were inspected during preparation: p225/p229 English and
# p237/p241 Scottish. Every label and speaker is checked again at run time.
targets = [(0,'English'),(3000,'English'),(9000,'Scottish'),(12000,'Scottish')]
selected = []; used_speakers = set()
for offset, expected_accent in targets:
    page = get_rows(offset)
    for entry in page['rows']:
        row = entry['row']
        if row.get('accent') != expected_accent: continue
        if row['speaker_id'] in used_speakers: continue
        if 'mic1' not in row.get('file',''): continue
        if len(normalize(row['text']).split()) < 8: continue
        audio, sr = sf.read(OFFLINE_INPUTS / row['audio'][0]['src'],dtype='float32')
        if audio.ndim == 2: audio = audio.mean(axis=1)
        if sr != 16000: audio = librosa.resample(audio,orig_sr=sr,target_sr=16000)
        duration = len(audio)/16000
        if not 1 <= duration <= 30: continue
        sample_id = f"vctk_{row['speaker_id']}_{row['text_id']}_mic1"
        path = SAMPLES/(sample_id+'.wav'); SAMPLES.mkdir(parents=True,exist_ok=True)
        sf.write(path,audio,16000,subtype='PCM_16')
        selected.append({'sample_id':sample_id,'pair_id':sample_id,'path':str(path),
                         'reference':row['text'],'speaker':row['speaker_id'],
                         'accent':row['accent'],'region':row.get('region',''),
                         'accent_label_source':'VCTK speaker metadata via kth-tmh/vctk; not independently assessed',
                         'language':'English','condition':'vctk_accent_read_speech','snr_db':None,
                         'source':'https://huggingface.co/datasets/kth-tmh/vctk',
                         'source_row_index':entry['row_idx'],'license':'CC BY 4.0',
                         'duration_s':duration,'audio_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        used_speakers.add(row['speaker_id']); break
    else:
        raise RuntimeError(f'No matching sample at offset {offset}. Dataset may have changed; send this error so selection can be updated.')
assert len(selected)==4 and len(used_speakers)==4
assert {r['accent'] for r in selected}=={'English','Scottish'}
display(pd.DataFrame(selected)[['sample_id','accent','region','duration_s','reference']])

# Save sources BEFORE inference. No reference text is derived from predictions.
pd.DataFrame(selected).to_csv(RESULTS/'accent_manifest.csv',index=False)
warm,sr = sf.read(selected[0]['path'],dtype='float32')
with torch.inference_mode(): model.transcribe(audio=(warm,sr),language=None)
torch.cuda.synchronize()
new_rows = []; new_trials = []
for row in selected:
    audio,sr = sf.read(row['path'],dtype='float32'); trials=[]
    for trial in range(3):
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with torch.inference_mode(): result = model.transcribe(audio=(audio,sr),language=None)[0]
        torch.cuda.synchronize()
        item={'sample_id':row['sample_id'],'trial':trial+1,'processing_s':time.perf_counter()-start,
              'prediction':result.text,'detected_language':result.language,
              'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3}
        trials.append(item); new_trials.append(item)
    prediction=trials[0]['prediction']; ref=normalize(row['reference']); hyp=normalize(prediction)
    words=jiwer.process_words(ref,hyp); chars=jiwer.process_characters(ref,hyp)
    latency=float(np.median([x['processing_s'] for x in trials]))
    new_rows.append({**row,'prediction':prediction,'detected_language':trials[0]['detected_language'],
                     'reference_normalized':ref,'prediction_normalized':hyp,
                     'raw_wer':jiwer.wer(row['reference'],prediction),'raw_cer':jiwer.cer(row['reference'],prediction),
                     'wer':words.wer,'cer':chars.cer,'substitutions':words.substitutions,
                     'deletions':words.deletions,'insertions':words.insertions,'reference_words':len(ref.split()),
                     'median_processing_s':latency,'rtf':latency/row['duration_s'],
                     'peak_allocated_gib':max(x['peak_allocated_gib'] for x in trials),
                     'repeat_transcripts_identical':len({x['prediction'] for x in trials})==1})
    pd.DataFrame(new_rows).to_csv(RESULTS/'accent_per_sample.csv',index=False)
    pd.DataFrame(new_trials).to_csv(RESULTS/'accent_all_trials.csv',index=False)
    print(row['sample_id'],row['accent'],'WER',round(100*words.wer,2),'%')

# Idempotent merge: rerunning this cell replaces prior VCTK extension rows.
baseline=pd.read_csv(RESULTS/'per_sample.csv',keep_default_na=False)
baseline=baseline[~baseline.sample_id.astype(str).str.startswith('vctk_')]
combined=pd.concat([baseline,pd.DataFrame(new_rows)],ignore_index=True)
combined.to_csv(RESULTS/'per_sample.csv',index=False)
old_trials=pd.read_csv(RESULTS/'all_trials.csv',keep_default_na=False)
old_trials=old_trials[~old_trials.sample_id.astype(str).str.startswith('vctk_')]
pd.concat([old_trials,pd.DataFrame(new_trials)],ignore_index=True).to_csv(RESULTS/'all_trials.csv',index=False)
old_manifest=pd.read_csv(RESULTS/'manifest.csv',keep_default_na=False)
old_manifest=old_manifest[~old_manifest.sample_id.astype(str).str.startswith('vctk_')]
pd.concat([old_manifest,pd.DataFrame(selected)],ignore_index=True).to_csv(RESULTS/'manifest.csv',index=False)

def summarize(group,label):
    return {'group':label,'n':len(group),
            'corpus_wer':jiwer.wer(group.reference_normalized.tolist(),group.prediction_normalized.tolist()),
            'corpus_cer':jiwer.cer(group.reference_normalized.tolist(),group.prediction_normalized.tolist()),
            'audio_total_s':float(group.duration_s.sum()),
            'median_latency_s':float(group.median_processing_s.median()),
            'aggregate_rtf':float(group.median_processing_s.sum()/group.duration_s.sum()),
            'peak_allocated_gib':float(group.peak_allocated_gib.max())}
summary=pd.DataFrame([summarize(combined,'all_samples')]+[summarize(g,str(k)) for k,g in combined.groupby('condition')])
summary.to_csv(RESULTS/'summary.csv',index=False)
accent_df=pd.DataFrame(new_rows)
accent_summary=pd.DataFrame([summarize(g,str(k)) for k,g in accent_df.groupby('accent')])
accent_summary.to_csv(RESULTS/'accent_summary.csv',index=False)
environment=json.loads((RESULTS/'environment.json').read_text())
environment.update({'sample_count':len(combined),'unique_speakers':int(combined.speaker.nunique()),
                    'verified_accent_labels':['English','Scottish'],'accent_coverage_complete':True,
                    'accent_label_qualification':'Dataset-documented labels; not independent linguistic assessment',
                    'accent_extension_gpu':torch.cuda.get_device_name(0),'accent_extension_repeats':3,
                    'accent_extension_note':'Four additional VCTK speakers; two per labelled accent. Convenience sample, not a controlled causal accent comparison.'})
(RESULTS/'environment.json').write_text(json.dumps(environment,indent=2),encoding='utf-8')

# Refresh the plot so it includes all inputs; retain the original evidence CSVs.
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,2,figsize=(12,4))
combined.plot.bar(x='sample_id',y='wer',ax=axes[0],legend=False,title='Per-sample normalized WER')
axes[0].set_ylabel('Error rate (fraction)')
axes[1].scatter(combined.duration_s,combined.median_processing_s)
axes[1].set(xlabel='Audio duration (s)',ylabel='Median processing time (s)',title='Observed latency')
fig.tight_layout();fig.savefig(RESULTS/'results.png',dpi=180,bbox_inches='tight');plt.show()
display(summary);display(accent_summary)
print('DONE: 16 inputs, 10 speakers, clean/noise and two dataset-labelled accent groups.')
print('Accent groups have different utterances/speakers. Do not attribute score differences solely to accent.')
archive=shutil.make_archive('/content/Task_2_Completed_Results','zip',root_dir=ROOT)
from google.colab import files
files.download(archive)
print('Now save this EXECUTED notebook using File > Download > Download .ipynb.')
