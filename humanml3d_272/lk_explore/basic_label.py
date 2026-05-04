"""
文本标签解析脚本 - 解析和可视化文本标注
"""
import re
from pathlib import Path
from collections import Counter


class TextAnnotation:
    """文本标注类"""

    def __init__(self, line):
        parts = line.strip().split('#')
        self.raw_text = parts[0] if len(parts) > 0 else ""
        self.pos_text = parts[1] if len(parts) > 1 else ""
        self.start_time = float(parts[2]) if len(parts) > 2 else 0.0
        self.end_time = float(parts[3]) if len(parts) > 3 else 0.0

        # 解析词性标注
        self.tokens = []
        if self.pos_text:
            for token_pos in self.pos_text.split():
                if '/' in token_pos:
                    word, pos = token_pos.rsplit('/', 1)
                    self.tokens.append({'word': word, 'pos': pos})

    def __repr__(self):
        return f"Text: {self.raw_text}\nTokens: {len(self.tokens)}\nTime: [{self.start_time}, {self.end_time}]"


def parse_text_file(text_file):
    """解析单个文本文件"""
    annotations = []
    with open(text_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                annotations.append(TextAnnotation(line))
    return annotations


def analyze_text_annotations():
    """分析文本标注的统计信息"""
    data_root = Path('../')
    text_dir = data_root / 'texts'

    if not text_dir.exists():
        print("❌ texts 目录不存在")
        return

    text_files = sorted(text_dir.glob('*.txt'))
    print("=" * 60)
    print("文本标注分析")
    print("=" * 60)
    print(f"\n📁 文本文件总数: {len(text_files)}")

    # 统计信息
    all_pos_tags = []
    all_words = []
    all_descriptions = []
    annotation_counts = []

    print("\n🔍 解析文本文件...")
    for i, text_file in enumerate(text_files[:100]):  # 分析前100个文件
        annotations = parse_text_file(text_file)
        annotation_counts.append(len(annotations))

        for ann in annotations:
            all_descriptions.append(ann.raw_text)
            for token in ann.tokens:
                all_pos_tags.append(token['pos'])
                all_words.append(token['word'].lower())

        # 显示前几个示例
        if i < 3:
            print(f"\n📄 示例文件 {text_file.name}:")
            for j, ann in enumerate(annotations[:2]):  # 每个文件显示前2个标注
                print(f"   标注 {j+1}:")
                print(f"      原始文本: {ann.raw_text}")
                print(f"      词性标注: {ann.pos_text[:100]}...")
                print(f"      时间范围: [{ann.start_time}, {ann.end_time}]")
                print(f"      词数: {len(ann.tokens)}")

    # 统计分析
    print(f"\n📊 统计摘要:")
    print(f"   - 总标注数: {len(all_descriptions)}")
    print(
        f"   - 平均每个文件的标注数: {sum(annotation_counts)/len(annotation_counts):.2f}")
    print(f"   - 总词数: {len(all_words)}")
    print(f"   - 唯一词数: {len(set(all_words))}")

    # 词性标签分布
    print(f"\n🏷️ 词性标签分布 (前10):")
    pos_counter = Counter(all_pos_tags)
    for pos, count in pos_counter.most_common(10):
        percentage = (count / len(all_pos_tags)) * 100
        print(f"   {pos:10s}: {count:6d} ({percentage:5.2f}%)")

    # 高频词
    print(f"\n📝 高频词 (前20):")
    word_counter = Counter(all_words)
    for word, count in word_counter.most_common(20):
        print(f"   {word:15s}: {count:4d}")

    # 描述长度分析
    desc_lengths = [len(desc.split()) for desc in all_descriptions]
    import numpy as np
    desc_lengths = np.array(desc_lengths)
    print(f"\n📏 描述长度统计:")
    print(f"   - 最短: {desc_lengths.min()} 词")
    print(f"   - 最长: {desc_lengths.max()} 词")
    print(f"   - 平均: {desc_lengths.mean():.2f} 词")
    print(f"   - 中位数: {np.median(desc_lengths):.2f} 词")

    # 提取动作关键词
    print(f"\n🏃 常见动作动词 (VERB):")
    action_verbs = [word for word, pos in zip(
        all_words, all_pos_tags) if pos == 'VERB']
    verb_counter = Counter(action_verbs)
    for verb, count in verb_counter.most_common(15):
        print(f"   {verb:15s}: {count:4d}")

    # 提取身体部位关键词
    body_parts = ['hand', 'leg', 'arm', 'foot',
                  'head', 'body', 'knee', 'shoulder', 'hip']
    print(f"\n🦵 身体部位词频:")
    for part in body_parts:
        count = all_words.count(part)
        if count > 0:
            print(f"   {part:15s}: {count:4d}")


def extract_action_patterns():
    """提取动作模式"""
    data_root = Path('..')
    text_dir = data_root / 'texts'

    if not text_dir.exists():
        return

    print("\n" + "=" * 60)
    print("动作模式提取")
    print("=" * 60)

    text_files = sorted(text_dir.glob('*.txt'))[:50]

    # 收集不同类型的动作描述
    walking_actions = []
    jumping_actions = []
    kicking_actions = []
    arm_actions = []

    for text_file in text_files:
        annotations = parse_text_file(text_file)
        for ann in annotations:
            text_lower = ann.raw_text.lower()
            if 'walk' in text_lower:
                walking_actions.append(ann.raw_text)
            if 'jump' in text_lower:
                jumping_actions.append(ann.raw_text)
            if 'kick' in text_lower:
                kicking_actions.append(ann.raw_text)
            if 'arm' in text_lower or 'hand' in text_lower:
                arm_actions.append(ann.raw_text)

    print(f"\n🚶 行走相关动作 ({len(walking_actions)} 个):")
    for action in walking_actions[:5]:
        print(f"   - {action}")

    print(f"\n🦘 跳跃相关动作 ({len(jumping_actions)} 个):")
    for action in jumping_actions[:5]:
        print(f"   - {action}")

    print(f"\n🦵 踢腿相关动作 ({len(kicking_actions)} 个):")
    for action in kicking_actions[:5]:
        print(f"   - {action}")

    print(f"\n🙌 手臂相关动作 ({len(arm_actions)} 个):")
    for action in arm_actions[:5]:
        print(f"   - {action}")


if __name__ == '__main__':
    analyze_text_annotations()
    extract_action_patterns()
