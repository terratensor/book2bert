```bash
=== Processing flibusta 2023 ===
2026/04/06 23:34:23 === Corpus Processor v3 (with CJK filtering) ===
2026/04/06 23:34:23 Corpus dir: /mnt/archive/corpus/flibusta_2023_143861_txt
2026/04/06 23:34:23 Output dir: /mnt/archive/book2bert/data/processed/sentences_full
2026/04/06 23:34:23 Segmenter URL: http://localhost:8090
2026/04/06 23:34:23 Workers: 10
2026/04/06 23:34:23 Extensions: [.txt .txt.gz]
2026/04/06 23:34:23 Collecting files...
2026/04/06 23:34:29 Found 143861 files
...
2026/04/07 01:31:09 [Worker 4] finished
2026/04/07 01:31:09 
=== Summary ===
2026/04/07 01:31:09 Total files: 143861
2026/04/07 01:31:09 Processed: 143861
2026/04/07 01:31:09 Errors: 0
2026/04/07 01:31:09 Skipped: 359
2026/04/07 01:31:09 Total sentences (approx): 776606995
2026/04/07 01:31:09 CJK log saved to: /mnt/archive/book2bert/data/processed/sentences_full/cjk_filtered.log
2026/04/07 01:31:09 Done!
=== Processing flibusta 2025 ===
2026/04/07 01:31:10 === Corpus Processor v3 (with CJK filtering) ===
2026/04/07 01:31:10 Corpus dir: /mnt/archive/corpus/flibusta_2025_txt
2026/04/07 01:31:10 Output dir: /mnt/archive/book2bert/data/processed/sentences_full
2026/04/07 01:31:10 Segmenter URL: http://localhost:8090
2026/04/07 01:31:10 Workers: 10
2026/04/07 01:31:10 Extensions: [.txt .txt.gz]
2026/04/07 01:31:10 Collecting files...
2026/04/07 01:31:11 Found 21388 files
...
2026/04/07 02:01:20 [Worker 8] finished
2026/04/07 02:01:20 
=== Summary ===
2026/04/07 02:01:20 Total files: 21388
2026/04/07 02:01:20 Processed: 21388
2026/04/07 02:01:20 Errors: 0
2026/04/07 02:01:20 Skipped: 25
2026/04/07 02:01:20 Total sentences (approx): 189639159
2026/04/07 02:01:20 CJK log saved to: /mnt/archive/book2bert/data/processed/sentences_full/cjk_filtered.log
2026/04/07 02:01:20 Done!
=== Processing militera ===
2026/04/07 02:01:20 === Corpus Processor v3 (with CJK filtering) ===
2026/04/07 02:01:20 Corpus dir: /mnt/archive/corpus/militera_2023_11359_txt
2026/04/07 02:01:20 Output dir: /mnt/archive/book2bert/data/processed/sentences_full
...
2026/04/07 02:18:23 [Worker 3] finished
2026/04/07 02:18:23 
=== Summary ===
2026/04/07 02:18:23 Total files: 11361
2026/04/07 02:18:23 Processed: 11361
2026/04/07 02:18:23 Errors: 0
2026/04/07 02:18:23 Skipped: 10
2026/04/07 02:18:23 Total sentences (approx): 107900756
2026/04/07 02:18:23 CJK log saved to: /mnt/archive/book2bert/data/processed/sentences_full/cjk_filtered.log
2026/04/07 02:18:23 Done!
=== Processing geography ===
2026/04/07 02:18:23 === Corpus Processor v3 (with CJK filtering) ===
2026/04/07 02:18:23 Corpus dir: /mnt/archive/corpus/geomatrix_geo_library_txt
2026/04/07 02:18:23 Output dir: /mnt/archive/book2bert/data/processed/sentences_full
...
2026/04/07 02:19:09 
=== Summary ===
2026/04/07 02:19:09 Total files: 191
2026/04/07 02:19:09 Processed: 191
2026/04/07 02:19:09 Errors: 0
2026/04/07 02:19:09 Skipped: 0
2026/04/07 02:19:09 Total sentences (approx): 7247353
2026/04/07 02:19:09 CJK log saved to: /mnt/archive/book2bert/data/processed/sentences_full/cjk_filtered.log
2026/04/07 02:19:09 Done!
=== All done ===

```
```bash
watch -n 60 'find /mnt/archive/book2bert/data/processed/sentences_full -name "*.jsonl" | wc -l'
176407
```
```bash
watch -n 30 'du -sh /mnt/archive/book2bert/data/processed/sentences_full'
408G    /mnt/archive/book2bert/data/processed/sentences_full
```


### Total sentences: 1 081 394 263
`776606995+189639159+107900756+7247353`
