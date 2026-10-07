"""Summarize the manually reviewed sample and generate local audit artifacts."""
import base64
import collections
import csv
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t

OUT = Path(__file__).resolve().parent
sample = json.loads((OUT/'sample.json').read_text(encoding='utf-8'))
census = json.loads((OUT/'census.json').read_text(encoding='utf-8'))
overrides = json.loads((OUT/'review_overrides.json').read_text(encoding='utf-8'))
rows = []
for r in sample:
    for q in r['questions']:
        aid = q['audit_id']
        status = 'no_clear_defect'
        category = 'No clear defect found'
        reason = ('Manually checked the selected image, sanitized context, question, answer and explanation. '
                  'No clear violation found under reasonable visual-estimate and ordinary chart-reading allowances. '
                  'This is not independent verification against the original raw dataframe.')
        if aid in overrides['clear_failures']:
            status = 'clear_failure'
            category, reason = overrides['clear_failures'][aid]
        elif aid in overrides['borderline']:
            status = 'borderline'
            category, reason = overrides['borderline'][aid]
        rows.append({
            'audit_id': aid, 'sample_chart': r['sample_chart'], 'prefix_id':r['prefix_id'],
            'graph_id':r['id'], 'source_file':r['source'], 'source_line':r['line'],
            'question_index_1based':q['question_index'], 'image':r['image'],
            'vlisual_valid':q['vlisual_valid'], 'data_valid':q['data_valid'],
            'status':status, 'category':category, 'audit_reason':reason,
            'question':q['question'], 'answer':q['answer'], 'explanation':q.get('explanation',''),
            'stored_visual_reason':q.get('visual_reason',''), 'stored_data_reason':q.get('data_reason',''),
        })
assert len(rows) == 180 and len({r['audit_id'] for r in rows}) == 180
assert set(overrides['clear_failures']) | set(overrides['borderline']) <= {r['audit_id'] for r in rows}
assert all(r['vlisual_valid'] is True and r['data_valid'] is True for r in rows)

N = census['counts']['questions_passed']
C = census['counts']['charts_accepted']
n = len(sample)
assert sum(int(k)*v for k,v in census['passed_questions_per_chart'].items()) == N
assert sum(census['passed_questions_per_chart'].values()) == C

def estimate(include_borderline=False):
    totals, within = [], []
    for r in sample:
        qs = [q for q in rows if q['sample_chart'] == r['sample_chart']]
        y = np.array([q['status']=='clear_failure' or (include_borderline and q['status']=='borderline') for q in qs], dtype=float)
        M, m = r['passed_questions_on_chart'], len(qs)
        totals.append(M*y.mean())
        within.append(M*M*(1-m/M)*y.var(ddof=1)/m)
    total = C/n*sum(totals)
    # Unbiased two-stage SRS variance estimator: chart clustering plus within-chart sampling.
    variance = C*C*(1-n/C)*np.var(totals,ddof=1)/n + C/n*sum(within)
    se = np.sqrt(variance)
    margin = float(t.ppf(.975,n-1))*se
    ci = [max(0,float(total-margin)), min(N,float(total+margin))]
    return {'estimated_total':float(total), 'estimated_rate':float(total/N),
            'standard_error_total':float(se), 'approx_95pct_total_interval':ci,
            'approx_95pct_rate_interval':[x/N for x in ci]}

status_counts = collections.Counter(r['status'] for r in rows)
categories = collections.Counter(r['category'] for r in rows if r['status']=='clear_failure')
summary = {'population_passed_questions':N,'sample_questions':len(rows),'sample_charts':n,
           'review_counts':dict(status_counts),'clear_failure_categories':dict(categories),
           'clear_failures_estimate':estimate(), 'including_borderline_sensitivity':estimate(True),
           'method':'Horvitz-Thompson total: (337/30)*sum(M_i*f_i/6); rate uses the known 6455 passing-question total. Two-stage simple random sampling variance with t(29) approximate 95% interval.',
           'limitations':'One model-led manual audit of selected chart images and sanitized descriptions, supported by saved plotting code and validator reasons. Raw data not retrieved or independently recomputed. Intervals describe sampling uncertainty conditional on these labels; they do not include adjudication uncertainty. No-clear-defect labels are not proof of validity.'}
(OUT/'results.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
(OUT/'reviewed_sample.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding='utf-8')
with (OUT/'reviewed_sample.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig, axes = plt.subplots(1,2,figsize=(12,4.6),gridspec_kw={'width_ratios':[1.05,1.5]})
labels=['Clear failures','Borderline','No clear defect']
values=[status_counts['clear_failure'],status_counts['borderline'],status_counts['no_clear_defect']]
colors=['#bb4430','#d3a12b','#3b8275']
bars=axes[0].barh(labels[::-1],values[::-1],color=colors[::-1],height=.6)
for b,v in zip(bars,values[::-1]):axes[0].text(v+2,b.get_y()+b.get_height()/2,str(v),va='center',weight='bold')
axes[0].set_xlim(0,175); axes[0].set_xlabel('Reviewed questions'); axes[0].set_title('180 questions across 30 random charts',loc='left',fontsize=12,pad=16)
for j,(key,label,color) in enumerate([('clear_failures_estimate','Clear failures',colors[0]),('including_borderline_sensitivity','Including borderline',colors[1])]):
    e=summary[key]; rate=e['estimated_rate']*100; lo,hi=[x*100 for x in e['approx_95pct_rate_interval']]
    axes[1].errorbar(rate,1-j,xerr=[[rate-lo],[hi-rate]],fmt='o',color=color,capsize=6,markersize=9)
    axes[1].text(rate,1-j+.16,f'{rate:.1f}%  (~{e["estimated_total"]:.0f} questions)',ha='center',fontsize=11,color=color,weight='bold')
axes[1].set_yticks([1,0],['Clear failures','Sensitivity']); axes[1].set_ylim(-.45,1.55)
axes[1].set_xlim(0,35); axes[1].set_xlabel('Estimated share of 6,455 validator-passing questions (%)')
axes[1].set_title('Weighted estimate and approximate 95% interval',loc='left',fontsize=12,pad=16)
axes[1].grid(axis='x',alpha=.2)
fig.suptitle('xhigh validator audit',x=.07,ha='left',fontsize=19,weight='bold')
fig.text(.07,.025,'Seed 20261007 · 6 passing questions per chart · Borderline cases are separate from the primary estimate',fontsize=10,color='#52616a')
fig.tight_layout(rect=[.015,.06,.99,.9])
fig.savefig(OUT/'audit_summary.png',dpi=180,facecolor='white')
fig.savefig(OUT/'audit_summary.svg',facecolor='white')
plt.close(fig)

E=html.escape
def embedded(path):return 'data:image/png;base64,'+base64.b64encode(Path(path).read_bytes()).decode()
e=summary['clear_failures_estimate']; s=summary['including_borderline_sensitivity']
parts=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>xhigh validator audit</title>',
       '<style>body{font:16px/1.55 system-ui;margin:40px auto;padding:0 24px;max-width:1100px;color:#21313d;background:#f7f9fb}h1{font-size:34px}img{max-width:100%;height:auto;background:white}section{background:white;border:1px solid #dce2e7;border-radius:12px;padding:24px;margin:24px 0}summary{cursor:pointer;font-weight:650}article{border-left:4px solid #3b8275;padding:8px 18px;margin:20px 0}article.clear_failure{border-color:#bb4430}article.borderline{border-color:#d3a12b}.badge{font-weight:700}.muted{color:#63717c}pre{white-space:pre-wrap;font:13px/1.4 monospace}details{margin:14px 0}a{color:#28618a}</style>',
       '<h1>xhigh validator audit</h1>',
       f'<p><strong>{status_counts["clear_failure"]} clear failures among 180 sampled passing questions.</strong> Weighted projection: <strong>{e["estimated_rate"]:.1%}, approximately {e["estimated_total"]:.0f} of 6,455 questions</strong>. Approximate sampling interval: {e["approx_95pct_total_interval"][0]:.0f}–{e["approx_95pct_total_interval"][1]:.0f} questions.</p>',
       f'<p>Another {status_counts["borderline"]} cases are borderline. Including these projects {s["estimated_rate"]:.1%}, approximately {s["estimated_total"]:.0f} questions. All examples below had both stored flags true.</p>',
       f'<img alt="Audit sample counts and population estimate" src="{embedded(OUT/"audit_summary.png")}">',
       '<section><h2>Method and limits</h2><p>Read all 51 metadata shards: 341 charts, of which 337 were accepted. Their 7,892 candidate questions included 6,455 that passed both validators and 1,437 rejected by at least one validator. Sample: 30 charts selected uniformly without replacement, then 6 passing questions per chart selected uniformly without replacement; seed 20261007. Source file, line, graph ID and original one-based question index are retained in CSV/JSON.</p>',
       '<p>Each sampled question, answer and explanation was checked against the selected original image and sanitized dataset description. The rubric requires chart evidence, permits approximate readings, rejects unstated specialist facts and hidden observations supplied in question context, and rejects incorrect explanations even with correct final answers. Saved code and judge reasons supported targeted checks; their assertions were not treated as independent ground truth.</p>',
       f'<p>{E(summary["method"])}</p><p>{E(summary["limitations"])}</p><p>A slip means the whole question-answer-explanation record fails the rubric; it does not necessarily mean its final answer is numerically wrong.</p></section>']
for r in sample:
    qs=[q for q in rows if q['sample_chart']==r['sample_chart']]
    failures=sum(q['status']=='clear_failure' for q in qs)
    borderline=sum(q['status']=='borderline' for q in qs)
    parts.append(f'<section id="C{r["sample_chart"]:02d}"><h2>C{r["sample_chart"]:02d} · {E(r["prefix_id"])} · {E(r["chart_type"])}</h2><p>{failures} clear failures · {borderline} borderline · {len(qs)} reviewed</p><details><summary>Original graph and allowed dataset context</summary><img alt="Chart {r["sample_chart"]}" src="{embedded(r["image"])}"><p>{E(r["dataset"]["sanitized_description"])}</p></details>')
    for q in qs:
        parts.append(f'<article class="{q["status"]}"><p class="badge">{q["audit_id"]} · {E(q["status"].replace("_"," "))} · {E(q["category"])}</p><p><b>Question:</b> {E(q["question"])}</p><p><b>Saved answer:</b> {E(q["answer"])}</p><p><b>Saved explanation:</b> {E(q["explanation"])}</p><p><b>Audit:</b> {E(q["audit_reason"])}</p><details><summary>Stored validator reasons and provenance</summary><p><b>Visual:</b> {E(q["stored_visual_reason"])}</p><p><b>Data:</b> {E(q["stored_data_reason"])}</p><pre>{E(q["source_file"])}:{q["source_line"]}\nGraph ID: {q["graph_id"]}\nQuestion index: {q["question_index_1based"]}</pre></details></article>')
    parts.append('</section>')
parts.append('</html>')
(OUT/'report.html').write_text('\n'.join(parts),encoding='utf-8')
print(json.dumps(summary,indent=2))
