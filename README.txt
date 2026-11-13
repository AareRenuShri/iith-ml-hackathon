IITH FOUNDATIONS OF MACHINE LEARNING 2026
KAGGLE HACKATHON - TEAM CRUCH

PHASE 1: DATA EXPLORATION AND UNDERSTANDING
-----------------------------------------------
1. DATASET DIMENSIONS

Training dataset:
- Rows: 202689
- Columns: 5
- Unique sentences: 12439

Testing dataset:
- Rows: 25094
- Columns: 4
- Unique sentences: 2077

Sample submission:
- Rows: 227783
- Columns: 2

2. COLUMN DESCRIPTIONS

Training columns:
- ID: Unique record identifier.
- sentence_id: Identifies the sentence.
- token_id: Position of the word in its sentence.
- token: Actual word or punctuation.
- target: Dependency annotation (head_id|relation_type).

Testing columns:
- ID
- sentence_id
- token_id
- token

Sample submission columns:
- ID
- Label

NOTE: Submission column must be named Label, not target.

3. TARGET FORMAT

Format: head_id|relation_type

Examples:
- 10|punct
- 3|det
- 0|root

4. DATASET STATISTICS

- Total training words: 202689
- Total training sentences: 12439
- Total testing words: 25094
- Total testing sentences: 2077
- Unique dependency relations: 51

5. RELATION FREQUENCY DISTRIBUTION

1
punct            23335
case             17315
nsubj            15808
det              15597
root             12439
advmod           11145
obj               9802
amod              9520
obl               8735
conj              7557
compound          7138
mark              7053
nmod              6817
cc                6700
aux               6420
cop               4683
advcl             3763
nmod:poss         3658
xcomp             3055
ccomp             2088
acl:relcl         1901
flat              1847
parataxis         1525
acl               1487
nummod            1464
appos             1383
aux:pass          1343
nsubj:pass        1171
nmod:unmarked     1120
obl:unmarked       994
discourse          802
compound:prt       736
list               665
iobj               642
expl               598
fixed              519
nmod:desc          353
obl:agent          312
csubj              275
nsubj:outer        213
det:predet         174
vocative           137
goeswith           123
advcl:relcl        119
cc:preconj          79
reparandum          36
orphan              26
csubj:pass           7
dislocated           5
dep                  3
csubj:outer          2

6. UNIQUE RELATION TYPES

acl, acl:relcl, advcl, advcl:relcl, advmod, amod, appos, aux, aux:pass, case, cc, cc:preconj, ccomp, compound, compound:prt, conj, cop, csubj, csubj:outer, csubj:pass, dep, det, det:predet, discourse, dislocated, expl, fixed, flat, goeswith, iobj, list, mark, nmod, nmod:desc, nmod:poss, nmod:unmarked, nsubj, nsubj:outer, nsubj:pass, nummod, obj, obl, obl:agent, obl:unmarked, orphan, parataxis, punct, reparandum, root, vocative, xcomp

7. TARGET VALIDATION

- Missing targets: 0
- Invalid head IDs: 0
- Missing relation types: 0
- Root relation count: 12439
- Unique relation count: 51

8. OBSERVATIONS

- The dataset contains 51 dependency relation types.
- Relation frequencies are highly imbalanced.
- Punctuation is the most frequent relation.
- Some relations have very few training examples.
- Root relation count matches the number of training sentences.


PHASE 2 & 3: MODEL TRAINING AND ITERATION
-----------------------------------------------
1. BASELINE MODEL (Phase 2):
- Model: Logistic Regression (L-BFGS, max_iter=2000)
- Features: lowercase, word_length, is_uppercase, is_digit, prefix (3 chars), suffix (3 chars)
- Feature dimensions: 23,455
- Training accuracy: 0.7146
- Validation accuracy: 0.6499
- Majority baseline: 0.1151

2. IMPROVED MODEL (Phase 3):
- Additional Features: is_title, is_lower, shape (ALLCAPS/lower/mixed), len_bucket (1, 2-3, 4-5, 6+)
- Feature dimensions: 23,464
- Training accuracy: 0.7150
- Validation accuracy: 0.6506
- Iterations: 1483

3. SUBMISSIONS GENERATED:
- submission_phase3_rel.csv: Relation predictions (25,094 rows)
- submission_phase3_head_rel.csv: Head + numeric relation predictions (25,094 rows)
- submission_phase3_named.csv: Head + textual relation predictions (25,094 rows, ready for Kaggle evaluation)
- Total test predictions shifted: 764 (3.04%) compared to Phase 2 baseline.
