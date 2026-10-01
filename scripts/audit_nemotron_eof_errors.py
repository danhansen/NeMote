#!/usr/bin/env python3
"""Classify utterance-ending errors without claiming they were caused by EOF.

Retain bounds across equally optimal WER alignments, so repeated words cannot
silently manufacture an apparent tail deletion or insertion.
"""
import argparse
import hashlib
import json
from pathlib import Path

from score_benchmark_wer import load_eval_helpers


def alignment(reference, hypothesis, tail_words=3):
    n, m = len(reference), len(hypothesis)
    if not n or tail_words < 1:
        raise ValueError('nonempty reference and positive tail length required')
    cutoff = max(0, n-tail_words)
    costs = [[0]*(m+1) for _ in range(n+1)]
    bounds = [[None]*(m+1) for _ in range(n+1)]
    previous = [[None]*(m+1) for _ in range(n+1)]
    bounds[0][0] = ((0,0), (0,0))  # tail-error min/max, final-word-region min/max
    for i in range(n+1):
        for j in range(m+1):
            if i == j == 0:
                continue
            choices = []
            if i and j:
                error = int(reference[i-1] != hypothesis[j-1])
                choices.append((i-1,j-1,'S' if error else 'M',error,
                                error*int(i-1>=cutoff),error*int(i==n)))
            if i:
                choices.append((i-1,j,'D',1,int(i-1>=cutoff),int(i==n)))
            if j:
                choices.append((i,j-1,'I',1,int(i>=cutoff),int(i==n)))
            best = min(costs[a][b]+error for a,b,op,error,t,f in choices)
            optimal = [c for c in choices if costs[c[0]][c[1]]+c[3] == best]
            costs[i][j] = best
            bounds[i][j] = tuple((min(bounds[a][b][k][0]+c[4+k] for c in optimal for a,b in [c[:2]]),
                                   max(bounds[a][b][k][1]+c[4+k] for c in optimal for a,b in [c[:2]]))
                                  for k in (0,1))
            # Deterministic tie order: match/substitution, deletion, insertion.
            previous[i][j] = optimal[0]
    operations = []
    i,j = n,m
    while i or j:
        a,b,op,error,t,f = previous[i][j]
        operations.append(dict(operation=op, reference_index=a,
                               reference_word=reference[a] if op!='I' else None,
                               hypothesis_word=hypothesis[b] if op!='D' else None,
                               tail=bool(t), final_word_region=bool(f),
                               trailing_insertion=op=='I' and a==n))
        i,j = a,b
    operations.reverse()
    return dict(edits=costs[n][m], tail_error_bounds=list(bounds[n][m][0]),
                final_word_region_error_bounds=list(bounds[n][m][1]), operations=operations)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, default=Path('build/performance-audit'))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    helpers = load_eval_helpers()
    names = dict(old='accuracy-ablation.json', new='frontend-flush-accuracy.json',
                 gguf='gguf-accuracy.json', manifest='accuracy-manifest.json')
    files = {k: args.directory/v for k,v in names.items()}
    data = {k:json.loads(v.read_text()) for k,v in files.items()}
    assert all(data[k]['status']=='complete' for k in ('old','new','gguf'))
    manifest = {r['utt_id']:r for r in data['manifest']}
    report = dict(status='complete', tail_words=3,
                  sources={k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in files.items()},
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  normalization='Same lowercase and punctuation-to-space normalization as corpus WER.',
                  limitations=['Ending errors do not establish EOF causality: acoustic, quantization, and decoding errors also occur at endings.',
                               '40 speakers / 914 words; ordinary recorded endings, not a controlled abrupt-stop stress test.',
                               'Minimum and maximum bounds cover all optimal edit alignments; operation counts use one deterministic alignment.',
                               'All four chunk modes reuse the same clips and are not 160 independent utterances.'],
                  records=[], summary=[])
    for c in (1,2,7,14):
        for label,source,precision in [('old-ort','old','original-quantized'),
                                       ('new-ort-flush','new','compact'),
                                       ('nvidia-gguf','gguf',None)]:
            rows = [r for r in data[source]['results'] if r['chunk_frames']==c and
                    (precision is None or r['label']==precision)]
            assert {r['clip'] for r in rows}==set(manifest)
            selected=[]
            for row in rows:
                ref=helpers.normalize_words(manifest[row['clip']]['reference'])
                hyp=helpers.normalize_words(row['text'])
                result=alignment(ref,hyp)
                assert result['edits']==row['edits']
                result.update(label=label,chunk_frames=c,clip=row['clip'],
                              reference=manifest[row['clip']]['reference'],hypothesis=row['text'])
                report['records'].append(result)
                selected.append(result)
            item=dict(chunk_frames=c,label=label,utterances=len(selected),
                      tail_reference_words=sum(min(3,len(helpers.normalize_words(manifest[r['clip']]['reference']))) for r in rows),
                      total_word_errors=sum(r['edits'] for r in selected),
                      tail_word_error_bounds=[sum(r['tail_error_bounds'][k] for r in selected) for k in (0,1)],
                      utterances_with_tail_error_bounds=[sum(r['tail_error_bounds'][k]>0 for r in selected) for k in (0,1)],
                      utterances_with_final_word_region_error_bounds=[sum(r['final_word_region_error_bounds'][k]>0 for r in selected) for k in (0,1)],
                      deterministic_tail_operation_counts={op:sum(v['tail'] and v['operation']==op for r in selected for v in r['operations']) for op in ('D','S','I')},
                      deterministic_trailing_insertions=sum(v['trailing_insertion'] for r in selected for v in r['operations']),
                      examples=[dict(clip=r['clip'],reference=r['reference'],hypothesis=r['hypothesis'],
                                     errors=[v for v in r['operations'] if v['tail']])
                                for r in selected if r['tail_error_bounds'][1]>0])
            report['summary'].append(item)
            print(json.dumps({k:v for k,v in item.items() if k!='examples'}))
    args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
