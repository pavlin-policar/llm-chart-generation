"""Create a reproducible, chart-clustered sample without changing source data."""
import collections
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'dataset/full_generation_xhigh'
OUT = ROOT / 'analysis/xhigh_validator_audit'
OUT.mkdir(parents=True, exist_ok=True)
records = []
counts = collections.Counter()
types = collections.Counter()
for source in sorted(DATA.glob('metadata*.jsonl')):
    for line_no, line in enumerate(source.open(encoding='utf-8'), 1):
        r = json.loads(line)
        counts['charts_total'] += 1
        if not r.get('accepted'):
            counts['charts_rejected'] += 1
            continue
        counts['charts_accepted'] += 1
        passed = []
        for qi, q in enumerate(r['graph'].get('questions', []), 1):
            counts['questions_candidates'] += 1
            flags = (q.get('vlisual_valid'), q.get('data_valid'))
            counts['flags_' + str(flags)] += 1
            if flags == (True, True):
                passed.append({'question_index': qi, **q})
                types[q.get('type', 'unknown')] += 1
        if not passed:
            continue
        counts['questions_passed'] += len(passed)
        image = next((im for im in r['images'] if im.get('selected')), r['images'][-1])
        path = DATA / image['path'].replace('\\', '/')
        if not path.exists():
            path = DATA / 'images' / Path(image['path']).name
        records.append({
            'id': r['id'], 'prefix_id': r['prefix_id'], 'source': str(source.relative_to(ROOT)),
            'line': line_no, 'image': str(path), 'dataset': r['dataset'],
            'chart_type': r['graph']['type'], 'code': r['graph']['code'],
            'structured_data': r['graph']['structured_data'], 'questions': passed,
        })

rng = random.Random(20261007)
passed_histogram = dict(collections.Counter(len(r['questions']) for r in records))
chosen = rng.sample(records, min(30, len(records)))
for i, r in enumerate(chosen, 1):
    r['sample_chart'] = i
    r['passed_questions_on_chart'] = len(r['questions'])
    r['questions'] = sorted(rng.sample(r['questions'], min(6, len(r['questions']))), key=lambda q:q['question_index'])
    for q in r['questions']:
        q['audit_id'] = f"C{i:02d}-Q{q['question_index']:02d}"
    assert Path(r['image']).is_file(), r['image']
summary = {'seed': 20261007, 'design': 'Simple random sample of 30 accepted charts with passing questions; simple random sample of up to 6 passing questions per sampled chart.', 'counts':dict(counts), 'passed_question_types': dict(types), 'passed_questions_per_chart':passed_histogram, 'sample_charts':len(chosen), 'sample_questions':sum(len(r['questions']) for r in chosen)}
(OUT/'census.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
(OUT/'sample.json').write_text(json.dumps(chosen, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps(summary, indent=2))
