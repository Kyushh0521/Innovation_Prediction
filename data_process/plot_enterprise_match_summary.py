# -*- coding: utf-8 -*-
"""
Generate pie charts that show how many enterprises have at least one match.

This script can process multiple sources (experts / achievements). It prints
both line-based and unique-enterprise statistics and saves PNGs under
`data_process_outputs/`.
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt


def load_jsonl(path: Path):
	with path.open('r', encoding='utf-8') as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			yield json.loads(line)


def compute_enterprise_match_stats(src: Path):
	"""Compute both line-based and unique-enterprise statistics.

	Returns a dict with keys:
	  total_lines, lines_with_match, total_enterprises, enterprises_with_match
	"""
	if not src.exists():
		raise FileNotFoundError(str(src))

	# line-based
	total_lines = 0
	lines_with_match = 0
	for obj in load_jsonl(src):
		total_lines += 1
		matches = obj.get('matches', [])
		if matches and len(matches) > 0:
			lines_with_match += 1

	# unique-enterprise based on enterprise_index
	by_enterprise = {}
	for obj in load_jsonl(src):
		ent_idx = obj.get('enterprise_index')
		matches = obj.get('matches', [])
		has_match = bool(matches and len(matches) > 0)
		if ent_idx is None:
			ent_idx = ('_line_', id(obj))
		prev = by_enterprise.get(ent_idx)
		by_enterprise[ent_idx] = bool(prev or has_match)

	total_enterprises = len(by_enterprise)
	enterprises_with_match = sum(1 for v in by_enterprise.values() if v)

	return {
		'total_lines': total_lines,
		'lines_with_match': lines_with_match,
		'total_enterprises': total_enterprises,
		'enterprises_with_match': enterprises_with_match,
	}


def draw_pie(matched: int, unmatched: int, out_path: Path, *,
			 labels=('Matched', 'Unmatched'), title='Enterprise Matching Summary'):
	"""Draw and save a pie chart showing matched vs unmatched counts.

	- matched, unmatched: integer counts
	- out_path: Path to save PNG
	- labels: 2-tuple of labels
	- title: chart title
	"""
	sizes = [matched, unmatched]
	colors = ['#66b3ff', '#ff9999']
	explode = (0.05, 0)

	plt.figure(figsize=(6, 6))
	plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, explode=explode, startangle=140)
	plt.title(title)
	plt.axis('equal')
	out_path.parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(str(out_path), dpi=300, bbox_inches='tight')
	plt.close()


def compute_presence_across_sources(expert_src: Path, achievement_src: Path):
	"""Compute per-enterprise presence of matches in both sources.

	Only enterprises with a non-None `enterprise_index` are considered for the
	cross-source presence check (we need a common key to join records).

	Returns a dict: total_indices, both_present, at_least_one_missing
	"""
	def build_presence_map(path: Path):
		m = {}
		if not path.exists():
			return m
		for obj in load_jsonl(path):
			ent_idx = obj.get('enterprise_index')
			if ent_idx is None:
				continue
			has_match = bool(obj.get('matches') and len(obj.get('matches')) > 0)
			prev = m.get(ent_idx)
			m[ent_idx] = bool(prev or has_match)
		return m

	experts_map = build_presence_map(expert_src)
	ach_map = build_presence_map(achievement_src)

	all_idxs = set(experts_map.keys()) | set(ach_map.keys())
	total = len(all_idxs)
	both = sum(1 for i in all_idxs if experts_map.get(i, False) and ach_map.get(i, False))
	missing = total - both
	return {'total_indices': total, 'both_present': both, 'at_least_one_missing': missing}


def main():
	# define sources: (src_path, out_filename, labels, title)
	sources = [
		(Path('data_process_outputs/enterprises_experts_matches.jsonl'),
		 'enterprise_expert_match_pie.png',
		 ('Matched (≥1 expert)', 'Unmatched'),
		 'Enterprise Expert Matching Summary'),
		(Path('data_process_outputs/enterprises_achievements_matches.jsonl'),
		 'enterprise_achievement_match_pie.png',
		 ('Matched (≥1 achievement)', 'Unmatched'),
		 'Enterprise Achievement Matching Summary'),
	]

	for src, out_name, labels, title in sources:
		if not src.exists():
			print(f"Skipping missing file: {src}")
			continue

		stats = compute_enterprise_match_stats(src)
		total_lines = stats['total_lines']
		lines_with_match = stats['lines_with_match']
		total_enterprises = stats['total_enterprises']
		enterprises_with_match = stats['enterprises_with_match']
		unmatched = total_enterprises - enterprises_with_match

		def pct(n, d):
			return (n / d * 100) if d else 0.0

		print(f'Processing {src.name}:')
		print(f'  Line-based: total lines = {total_lines}, lines with ≥1 match = {lines_with_match} ({pct(lines_with_match, total_lines):.2f}%)')
		print(f'  Unique-enterprise: total unique enterprises = {total_enterprises}, enterprises with ≥1 match = {enterprises_with_match} ({pct(enterprises_with_match, total_enterprises):.2f}%)')
		print(f'  Unique-enterprise: enterprises without match = {unmatched} ({pct(unmatched, total_enterprises):.2f}%)')

		out_path = src.parent / out_name
		draw_pie(enterprises_with_match, unmatched, out_path, labels=labels, title=title)
		print(f'  Pie chart saved: {out_path}\n')

	# --- cross-source presence: expert vs achievement ---
	expert_file = Path('data_process_outputs/enterprises_experts_matches.jsonl')
	ach_file = Path('data_process_outputs/enterprises_achievements_matches.jsonl')
	if expert_file.exists() and ach_file.exists():
		presence = compute_presence_across_sources(expert_file, ach_file)
		total_idxs = presence['total_indices']
		both = presence['both_present']
		missing = presence['at_least_one_missing']

		print('Cross-source presence (by enterprise_index):')
		print(f'  Total enterprise_index values considered: {total_idxs}')
		print(f'  Both expert and achievement present: {both} ({(both/total_idxs*100) if total_idxs else 0:.2f}%)')
		print(f'  At least one missing: {missing} ({(missing/total_idxs*100) if total_idxs else 0:.2f}%)')

		out_cross = expert_file.parent / 'enterprise_expert_achievement_presence_pie.png'
		draw_pie(both, missing, out_cross, labels=('Both present', 'At least one missing'), title='Expert & Achievement Presence')
		print(f'  Cross-source pie chart saved: {out_cross}\n')
	else:
		print('Skipping cross-source presence chart: one or both source files missing.')


if __name__ == '__main__':
	main()

