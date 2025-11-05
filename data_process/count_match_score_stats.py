import json
from pathlib import Path


def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main():
    src = Path('data_process_outputs/enterprises_achievements_matches.jsonl')
    if not src.exists():
        print(f"文件未找到: {src}")
        return

    total_pairs = 0
    gt_08_pairs = 0
    gt_075_pairs = 0

    total_enterprises = 0
    enterprises_with_gt_08 = 0
    enterprises_with_gt_075 = 0

    for obj in load_jsonl(src):
        total_enterprises += 1
        matches = obj.get('matches', [])
        has_gt_08 = False
        has_gt_075 = False
        for m in matches:
            score = float(m.get('score', 0))
            total_pairs += 1
            if score > 0.80:
                gt_08_pairs += 1
                has_gt_08 = True
            if score > 0.73:
                gt_075_pairs += 1
                has_gt_075 = True
        if has_gt_08:
            enterprises_with_gt_08 += 1
        if has_gt_075:
            enterprises_with_gt_075 += 1

    def pct(n, d):
        return (n / d * 100) if d else 0.0

    print('配对级别统计:')
    print(f'  总配对数: {total_pairs}')
    print(f'  分数 > 0.80 的配对: {gt_08_pairs} ({pct(gt_08_pairs, total_pairs):.2f}%)')
    print(f'  分数 > 0.75 的配对: {gt_075_pairs} ({pct(gt_075_pairs, total_pairs):.2f}%)')

    print('\n企业级别统计:')
    print(f'  总企业数（行数）: {total_enterprises}')
    print(f'  至少有一个 match > 0.80 的企业数: {enterprises_with_gt_08} ({pct(enterprises_with_gt_08, total_enterprises):.2f}%)')
    print(f'  至少有一个 match > 0.75 的企业数: {enterprises_with_gt_075} ({pct(enterprises_with_gt_075, total_enterprises):.2f}%)')


if __name__ == '__main__':
    main()