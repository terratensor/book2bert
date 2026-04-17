```
============================================================
РЕЗУЛЬТАТЫ (покрытие уникальных слов)
============================================================
 vocab_size  coverage  actual_vocab
     100000   0.31211         86717
     120000   0.36133        104788
     150000   0.42762        132205
     200000   0.51459        177653

График сохранён: data/analysis/vocab_coverage_real_1m.png
```

```
(venv) audetv@home:/mnt/work/audetv/go/src/github.com/terratensor/book2bert$ head -n 10000000 /mnt/archive/book2bert/data/processed/corpus_full.txt > data/corpus_10m.txt
(venv) audetv@home:/mnt/work/audetv/go/src/github.com/terratensor/book2bert$ python scripts/test_vocab_size_real.py     --corpus data/corpus_10m.txt     --vocab-sizes 100000,120000,150000,200000     --max-test
-words 100000
```
```
============================================================
РЕЗУЛЬТАТЫ (покрытие уникальных слов)
============================================================
 vocab_size  coverage  actual_vocab
     100000   0.30446         86484
     120000   0.35172        104689
     150000   0.41110        132081
     200000   0.48678        177743

График сохранён: data/analysis/vocab_coverage_real.png
```
```
============================================================
РЕЗУЛЬТАТЫ (покрытие уникальных слов)
============================================================
 vocab_size  coverage  actual_vocab
     100000  0.301157         86079
     120000  0.347713        104147
     150000  0.406627        131293
     200000  0.482211        176654

График сохранён: data/analysis/vocab_coverage_real.png
```

```
============================================================
РЕЗУЛЬТАТЫ (покрытие уникальных слов)
============================================================
 vocab_size  coverage  actual_vocab
     250000  0.537315        222138
     300000  0.577496        267618
     350000  0.609380        313025
     400000  0.634398        358533

График сохранён: data/analysis/vocab_coverage_real.png
```
