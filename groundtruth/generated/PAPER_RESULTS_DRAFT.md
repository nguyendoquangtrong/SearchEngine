# Draft results for the revised evaluation

## Evaluation set

The survey supplied 183 queries from 61 participants. After owner review, 9 non-informative or unsupported queries were excluded. The primary evaluation contains 174 user-reviewed target labels: 145 supported and 29 partly supported. A participant-level split (seed 4204) reserves 15 participants / 42 queries for development and 45 participants / 132 queries for test. Duplicate-query clusters were kept within a single split.

Metrics are Success@1, Success@5, and MRR@5 at movie level. The submitted V1 configuration and all baselines were frozen before computing test metrics.

## Test results: all kept queries (n=132)

| System | Success@1 | Success@5 | MRR@5 |
|---|---:|---:|---:|
| BM25 | 0.5152 | 0.7348 | 0.5995 |
| SBERT | 0.4773 | 0.6439 | 0.5431 |
| CLIP Text | 0.3182 | 0.3561 | 0.3359 |
| CLIP Image | 0.0758 | 0.3485 | 0.1678 |
| V1 PT2 | **0.5530** | **0.7576** | **0.6318** |

The participant-cluster bootstrap 95% CI for the MRR@5 difference V1 PT2 minus BM25 is [-0.0515, 0.1094], and minus SBERT is [-0.0000, 0.1750]. Thus V1 PT2 has the highest observed point estimates, but this dataset does not establish a statistically reliable advantage over BM25 or SBERT at 95% confidence.

## Sensitivity analysis: supported queries only (n=108 test queries)

| System | Success@1 | Success@5 | MRR@5 |
|---|---:|---:|---:|
| BM25 | 0.5370 | 0.7593 | 0.6225 |
| SBERT | 0.5093 | 0.6574 | 0.5677 |
| CLIP Text | 0.3241 | 0.3519 | 0.3364 |
| CLIP Image | 0.0833 | 0.3519 | 0.1716 |
| V1 PT2 | **0.5926** | **0.7685** | **0.6603** |

The result pattern remains: V1 PT2 has the highest point estimates. The V1 PT2 minus BM25 MRR@5 CI is [-0.0541, 0.1187], so the sensitivity analysis also does not support a 95%-confidence claim of superiority.

## Reporting language

Use “user-reviewed target labels” in the paper. Do not call them independently adjudicated ground truth unless a second reviewer independently labels the queries and agreement is reported.
