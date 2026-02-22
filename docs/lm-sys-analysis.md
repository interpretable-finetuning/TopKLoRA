# Analysis of LMSYS-chat-1M dataset



Analysis of [lmsys/lmsys-chat-1m](https://huggingface.co/datasets/lmsys/lmsys-chat-1m). Output from `python3 scripts/analyze_lmsys_dataset.py`.

```
======================================================================
SAMPLE COUNTS
======================================================================
Total samples:                       1,000,000
English samples:                       777,453 (77.75%)
Non-redacted samples:                  733,740 (73.37%)
English AND non-redacted:              511,193 (51.12%)

======================================================================
TOKEN STATISTICS (ALL SAMPLES)
======================================================================
Total tokens:                         508,923,348
Mean tokens per sample:                     508.9
Std dev:                                  4,194.1
Median tokens per sample:                   326.0
Min tokens:                                    11
Max tokens:                             1,578,133

======================================================================
TOKEN STATISTICS (ENGLISH & NON-REDACTED ONLY)
======================================================================
Total tokens:                         189,177,521
Mean tokens per sample:                     370.1
Std dev:                                  1,202.2
Median tokens per sample:                   271.0
Min tokens:                                    12
Max tokens:                               663,129

======================================================================
TOKEN LENGTH DISTRIBUTION (PERCENTILES)
======================================================================
Percentile       All Samples   English Non-Redacted
----------------------------------------------------------------------
P10                       61                     61
P25                      131                    109
P50                      326                    271
P75                      566                    470
P90                    1,028                    755
P95                    1,491                  1,043
P99                    2,821                  1,899
======================================================================
```
