#!/usr/bin/env python3
"""
Анализ статистики корпуса и построение графиков
"""

import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path

def load_and_plot(csv_path, output_dir):
    """Загружает CSV и строит графики"""
    
    # 1. Гистограмма длин
    df = pd.read_csv(csv_path)
    df['count'] = pd.to_numeric(df['count'])
    
    # Извлекаем начало бакета
    df['bucket_start'] = df['bucket'].str.split('-').str[0].astype(int)
    
    plt.figure(figsize=(14, 8))
    plt.bar(df['bucket_start'], df['count'], width=80, alpha=0.7, edgecolor='black')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Длина предложения (символы, логарифмическая шкала)', fontsize=12)
    plt.ylabel('Количество предложений (логарифмическая шкала)', fontsize=12)
    plt.title('Распределение длин предложений в корпусе', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'length_distribution.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/length_distribution.png")

def plot_language_composition(stats_json_path, output_dir):
    """Строит круговую диаграмму языкового состава"""
    
    with open(stats_json_path, 'r') as f:
        stats = json.load(f)
    
    languages = ['Russian only', 'English only', 'Mixed (Ru+En)', 'Other']
    counts = [
        stats['russian_only'],
        stats['english_only'],
        stats['mixed_cyrillic_latin'],
        stats['other']
    ]
    
    plt.figure(figsize=(10, 8))
    plt.pie(counts, labels=languages, autopct='%1.1f%%', startangle=90)
    plt.title('Языковой состав корпуса', fontsize=14)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'language_composition.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/language_composition.png")

def plot_quality_metrics(stats_json_path, output_dir):
    """Строит диаграмму метрик качества"""
    
    with open(stats_json_path, 'r') as f:
        stats = json.load(f)
    
    total = stats['total_sentences']
    
    metrics = {
        'Too short (<20)': stats['too_short_20'],
        'Too long (>1000)': stats['too_long_1000'],
        'High digit (>30%)': stats['high_digit_30'],
        'List marker': stats['list_marker'],
        'Has ISBN': stats['has_isbn'],
        'Has УДК': stats['has_udk'],
        'Has ББК': stats['has_bbk'],
    }
    
    names = list(metrics.keys())
    values = [metrics[n] / total * 100 for n in names]
    
    plt.figure(figsize=(12, 6))
    bars = plt.barh(names, values, color='skyblue', edgecolor='black')
    plt.xlabel('Доля от всех предложений (%)', fontsize=12)
    plt.title('Метрики качества текста', fontsize=14)
    for bar, val in zip(bars, values):
        plt.text(val + 0.5, bar.get_y() + bar.get_height()/2, f'{val:.2f}%', va='center')
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'quality_metrics.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/quality_metrics.png")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True, help='stats_length_histogram.csv')
    parser.add_argument('--json', type=str, required=True, help='stats_summary.json')
    parser.add_argument('--output', type=str, default='data/analysis', help='выходная директория')
    args = parser.parse_args()
    
    load_and_plot(args.csv, args.output)
    plot_language_composition(args.json, args.output)
    plot_quality_metrics(args.json, args.output)
    
    print("\n✅ Графики сохранены!")

if __name__ == "__main__":
    main()