# Hard & High-Disagreement Examples — ICL Stance Analysis

**Dataset:** SemEval 2016 Task 6 test set (1,249 examples)
**Models:** 13 (5 shot counts each: 0, 3, 6, 9, 12)
**Note:** All examples are from the test set. Not to be used as demonstrations.

**Matrix key:** FAV=FAVOR · AGN=AGAINST · NON=NONE · ✓=correct · ✗=wrong
**Metrics:** `err` = mean error rate across all models × shots · `instab` = mean label flips per model · `disagree` = mean prediction entropy (max ≈ 1.58)

---

## A · Highest Cross-Model Disagreement

Top 4 examples per topic ranked by mean prediction entropy across models and shot counts.


### Topic: Atheism

### doc 10004 · Atheism · Gold: **AGAINST**
> #God is utterly powerless without Human intervention... #SemST
*err=0.69  instab=0.69  disagree=1.50  wrong@12: 9/13*
  Model         |  0s   3s   6s   9s  12s
  --------------|------------------------------
  Q1.7B         | NON✗  NON✗  NON✗  NON✗  NON✗
  Ph4-mini      | AGN✓  AGN✓  AGN✓  AGN✓  AGN✓
  Nem4B         | NON✗  AGN✓  AGN✓  AGN✓  AGN✓
  Q4B           | FAV✗  FAV✗  FAV✗  FAV✗  FAV✗
  Q4B-i         | AGN✓  AGN✓  AGN✓  AGN✓  AGN✓
  Llama8B       | AGN✓  FAV✗  FAV✗  FAV✗  FAV✗
  DS-Llama8     | NON✗  FAV✗  NON✗  NON✗  NON✗
  Q8B           | FAV✗  FAV✗  FAV✗  FAV✗  FAV✗
  DS-Q14        | NON✗  NON✗  FAV✗  FAV✗  FAV✗
  Q14B          | NON✗  NON✗  NON✗  NON✗  FAV✗
  Ph4           | AGN✓  FAV✗  AGN✓  AGN✓  AGN✓
  GPT5m         | AGN✓  FAV✗  FAV✗  FAV✗  FAV✗
  GPT5          | FAV✗  FAV✗  FAV✗  FAV✗  FAV✗

### doc 10010 · Atheism · Gold: **AGAINST**
> #BIBLE = Big Irrelevant Book of Lies and Exaggerations ~ #Judaism #God #TeamJesus #Islam ~ #Truth & #Freedom = #SemST
*err=0.61  instab=1.08  disagree=1.47  wrong@12: 9/13*
  Model         |  0s   3s   6s   9s  12s
  --------------|------------------------------
  Q1.7B         | NON✗  NON✗  NON✗  NON✗  NON✗
  Ph4-mini      | AGN✓  AGN✓  AGN✓  AGN✓  AGN✓
  Nem4B         | AGN✓  AGN✓  AGN✓  AGN✓  AGN✓
  Q4B           | FAV✗  FAV✗  FAV✗  NON✗  FAV✗
  Q4B-i         | AGN✓  AGN✓  AGN✓  AGN✓  AGN✓
  Llama8B       | AGN✓  FAV✗  FAV✗  FAV✗  FAV✗
  DS-Llama8     | NON✗  NON✗  NON✗  NON✗  NON✗
  Q8B           | FAV✗  FAV✗  AGN✓  AGN✓  FAV✗
  DS-Q14        | NON✗  FAV✗  FAV✗  AGN✓  FAV✗
  Q14B          | NON✗  AGN✓  FAV✗  AGN✓  FAV✗
  Ph4           | AGN✓  FAV✗  AGN✓  AGN✓  AGN✓
  GPT5m         | FAV✗  FAV✗  FAV✗  FAV✗  FAV✗
  GPT5          | FAV✗  FAV✗  FAV✗  FAV✗  FAV✗

### doc 10013 · Atheism · Gold: **AGAINST**
> Let my house be built by wisdom and become strong through good sense -Prov. 24:3 #SemST
*err=0.65  instab=1.31  disagree=1.35  wrong@12: 8/13*
  Model         |  0s   3s   6s   9s  12s
  --------------|------------------------------
  Q1.7B         | NON✗  FAV✗  FAV✗  FAV✗  FAV✗
  Ph4-mini      | NON✗  AGN✓  AGN✓  NON✗  NON✗
  Nem4B         | NON✗  FAV✗  FAV✗  NON✗  NON✗
  Q4B           | NON✗  AGN✓  AGN✓  NON✗  NON✗
  Q4B-i         | NON✗  AGN✓  AGN✓  AGN✓  AGN✓
  Llama8B       | NON✗  FAV✗  FAV✗  FAV✗  FAV✗
  DS-Llama8     | NON✗  NON✗  NON✗  NON✗  NON✗
  Q8B           | NON✗  AGN✓  NON✗  NON✗  AGN✓
  DS-Q14        | NON✗  NON✗  NON✗  NON✗  NON✗
  Q14B          | FAV✗  NON✗  NON✗  AGN✓  NON✗
  Ph4           | NON✗  NON✗  AGN✓  AGN✓  AGN✓
  GPT5m         | NON✗  AGN✓  AGN✓  AGN✓  AGN✓
  GPT5          | AGN✓  AGN✓  AGN✓  AGN✓  AGN✓

### doc 10002 · Atheism · Gold: **AGAINST**
> RT @prayerbullets: I remove Nehushtan -previous moves of God that have become idols, from the high places -2 Kings 18:4 #SemST
*err=0.51  instab=0.77  disagree=1.35  wrong@12: 6/13*
  Model         |  0s   3s   6s   9s  12s
  --------------|------------------------------
  Q1.7B         | NON✗  NON✗  NON✗  NON✗  NON✗
  Ph4-mini      | AGN✓  AGN✓  AGN✓  AGN✓  AGN✓
  Nem4B         | NON✗  AGN✓  AGN✓  AGN✓  AGN✓
  Q4B           | NON✗  FAV✗  NON✗  NON✗  AGN✓
  Q4B-i         | AGN✓  FAV✗  FAV✗  FAV✗  FAV✗
  Llama8B       | AGN✓  FAV✗  FAV✗  FAV✗  FAV✗
  DS-Llama8     | NON✗  NON✗  NON✗  NON✗  NON✗
  Q8B           | NON✗  AGN✓  AGN✓  AGN✓  AGN✓
  DS-Q14        | NON✗  NON✗  AGN✓  AGN✓  NON✗
  Q14B          | NON✗  NON✗  NON✗  NON✗  NON✗
  Ph4           | NON✗  AGN✓  AGN✓  AGN✓  AGN✓
  GPT5m         | AGN✓  AGN✓  AGN✓  AGN✓  AGN✓
  GPT5          | AGN✓  AGN✓  AGN✓  AGN✓  AGN✓


---
## B1 · Universally Hard

Majority of models wrong at 12-shot AND mean error rate > 0.65.

### doc 10003 · Atheism · Gold: **AGAINST**
> @Brainman365 @heidtjj @BenjaminLives I have sought the truth of my soul and found it strong enough to stand on its own merits. #Se…
*err=1.00  instab=0.92  disagree=0.90  wrong@12: 13/13*
  Model         |  0s   3s   6s   9s  12s
  --------------|------------------------------
  Q1.7B         | NON✗  FAV✗  NON✗  NON✗  NON✗
  Ph4-mini      | NON✗  FAV✗  FAV✗  FAV✗  FAV✗
  Nem4B         | FAV✗  FAV✗  FAV✗  NON✗  FAV✗
  Q4B           | NON✗  FAV✗  FAV✗  FAV✗  NON✗
  Q4B-i         | NON✗  FAV✗  FAV✗  FAV✗  FAV✗
  Llama8B       | FAV✗  FAV✗  FAV✗  FAV✗  FAV✗
  DS-Llama8     | NON✗  NON✗  NON✗  NON✗  NON✗
  Q8B           | NON✗  FAV✗  FAV✗  NON✗  FAV✗
  DS-Q14        | NON✗  NON✗  NON✗  NON✗  NON✗
  Q14B          | NON✗  NON✗  NON✗  NON✗  NON✗
  Ph4           | FAV✗  FAV✗  FAV✗  FAV✗  FAV✗
  GPT5m         | FAV✗  FAV✗  FAV✗  FAV✗  FAV✗
  GPT5          | NON✗  FAV✗  FAV✗  FAV✗  FAV✗

### doc 10007 · Atheism · Gold: **AGAINST**
> Morality is not derived from religion, it precedes it. -Christopher 'The Hitch' Hitchens #freethinkers #SemST
*err=1.00  instab=0.23  disagree=0.16  wrong@12: 13/13*
  Model         |  0s   3s   6s   9s  12s
  --------------|------------------------------
  Q1.7B         | FAV✗  FAV✗  FAV✗  FAV✗  FAV✗
  Ph4-mini      | FAV✗  FAV✗  FAV✗  FAV✗  FAV✗
  Nem4B         | FAV✗  FAV✗  FAV✗  FAV✗  FAV✗
  Q4B           | FAV✗  FAV✗  FAV✗  FAV✗  FAV✗
  Q4B-i         | FAV✗  FAV✗  FAV✗  FAV✗  FAV✗
  Llama8B       | FAV✗  FAV✗  FAV✗  FAV✗  FAV✗
  DS-Llama8     | FAV✗  FAV✗  FAV✗  NON✗  FAV✗
  Q8B           | FAV✗  FAV✗  FAV✗  FAV✗  FAV✗
  DS-Q14        | FAV✗  FAV✗  FAV✗  FAV✗  FAV✗
  Q14B          | NON✗  FAV✗  FAV✗  FAV✗  FAV✗
  Ph4           | FAV✗  FAV✗  FAV✗  FAV✗  FAV✗
  GPT5m         | FAV✗  FAV✗  FAV✗  FAV✗  FAV✗
  GPT5          | FAV✗  FAV✗  FAV✗  FAV✗  FAV✗

### doc 10009 · Atheism · Gold: **AGAINST**
> @SecularDutchess I'll be your huckleberry @DeanModified #SemST
*err=1.00  instab=1.00  disagree=0.79  wrong@12: 13/13*
  Model         |  0s   3s   6s   9s  12s
  --------------|------------------------------
  Q1.7B         | NON✗  NON✗  NON✗  NON✗  NON✗
  Ph4-mini      | NON✗  FAV✗  NON✗  NON✗  NON✗
  Nem4B         | FAV✗  NON✗  NON✗  NON✗  NON✗
  Q4B           | FAV✗  NON✗  FAV✗  FAV✗  FAV✗
  Q4B-i         | NON✗  FAV✗  FAV✗  NON✗  NON✗
  Llama8B       | NON✗  FAV✗  FAV✗  FAV✗  FAV✗
  DS-Llama8     | NON✗  NON✗  NON✗  NON✗  NON✗
  Q8B           | NON✗  FAV✗  FAV✗  NON✗  FAV✗
  DS-Q14        | NON✗  NON✗  NON✗  FAV✗  NON✗
  Q14B          | NON✗  NON✗  NON✗  NON✗  NON✗
  Ph4           | NON✗  NON✗  NON✗  NON✗  NON✗
  GPT5m         | NON✗  NON✗  NON✗  NON✗  NON✗
  GPT5          | NON✗  NON✗  NON✗  NON✗  NON✗

### doc 10011 · Atheism · Gold: **AGAINST**
> If only dreams were real, now it's gone. #SingleBecause #getonyourfeet #SemST
*err=1.00  instab=0.00  disagree=0.00  wrong@12: 13/13*
  Model         |  0s   3s   6s   9s  12s
  --------------|------------------------------
  Q1.7B         | NON✗  NON✗  NON✗  NON✗  NON✗
  Ph4-mini      | NON✗  NON✗  NON✗  NON✗  NON✗
  Nem4B         | NON✗  NON✗  NON✗  NON✗  NON✗
  Q4B           | NON✗  NON✗  NON✗  NON✗  NON✗
  Q4B-i         | NON✗  NON✗  NON✗  NON✗  NON✗
  Llama8B       | NON✗  NON✗  NON✗  NON✗  NON✗
  DS-Llama8     | NON✗  NON✗  NON✗  NON✗  NON✗
  Q8B           | NON✗  NON✗  NON✗  NON✗  NON✗
  DS-Q14        | NON✗  NON✗  NON✗  NON✗  NON✗
  Q14B          | NON✗  NON✗  NON✗  NON✗  NON✗
  Ph4           | NON✗  NON✗  NON✗  NON✗  NON✗
  GPT5m         | NON✗  NON✗  NON✗  NON✗  NON✗
  GPT5          | NON✗  NON✗  NON✗  NON✗  NON✗

### doc 10012 · Atheism · Gold: **AGAINST**
> Happy Independence Day to America and her beautiful constitution #IndependenceDay #USA #SemST
*err=1.00  instab=0.15  disagree=0.08  wrong@12: 13/13*
  Model         |  0s   3s   6s   9s  12s
  --------------|------------------------------
  Q1.7B         | NON✗  NON✗  NON✗  NON✗  NON✗
  Ph4-mini      | NON✗  NON✗  NON✗  NON✗  NON✗
  Nem4B         | NON✗  FAV✗  NON✗  NON✗  NON✗
  Q4B           | NON✗  NON✗  NON✗  NON✗  NON✗
  Q4B-i         | NON✗  NON✗  NON✗  NON✗  NON✗
  Llama8B       | NON✗  NON✗  NON✗  NON✗  NON✗
  DS-Llama8     | NON✗  NON✗  NON✗  NON✗  NON✗
  Q8B           | NON✗  NON✗  NON✗  NON✗  NON✗
  DS-Q14        | NON✗  NON✗  NON✗  NON✗  NON✗
  Q14B          | NON✗  NON✗  NON✗  NON✗  NON✗
  Ph4           | NON✗  NON✗  NON✗  NON✗  NON✗
  GPT5m         | NON✗  NON✗  NON✗  NON✗  NON✗
  GPT5          | NON✗  NON✗  NON✗  NON✗  NON✗

### doc 10014 · Atheism · Gold: **AGAINST**
> These days, the cool kids are atheists.  #freethinker #SemST
*err=1.00  instab=0.15  disagree=0.08  wrong@12: 13/13*
  Model         |  0s   3s   6s   9s  12s
  --------------|------------------------------
  Q1.7B         | FAV✗  FAV✗  FAV✗  FAV✗  FAV✗
  Ph4-mini      | FAV✗  FAV✗  FAV✗  FAV✗  FAV✗
  Nem4B         | FAV✗  FAV✗  FAV✗  FAV✗  FAV✗
  Q4B           | FAV✗  FAV✗  FAV✗  FAV✗  FAV✗
  Q4B-i         | FAV✗  FAV✗  FAV✗  FAV✗  FAV✗
  Llama8B       | FAV✗  FAV✗  FAV✗  FAV✗  FAV✗
  DS-Llama8     | FAV✗  FAV✗  FAV✗  FAV✗  FAV✗
  Q8B           | FAV✗  FAV✗  NON✗  FAV✗  FAV✗
  DS-Q14        | FAV✗  FAV✗  FAV✗  FAV✗  FAV✗
  Q14B          | FAV✗  FAV✗  FAV✗  FAV✗  FAV✗
  Ph4           | FAV✗  FAV✗  FAV✗  FAV✗  FAV✗
  GPT5m         | FAV✗  FAV✗  FAV✗  FAV✗  FAV✗
  GPT5          | FAV✗  FAV✗  FAV✗  FAV✗  FAV✗

### doc 10020 · Atheism · Gold: **AGAINST**
> America, like all of us, has both beauty and brutality coursing through its blood. #4thofjuly #spirituality #SemST
*err=1.00  instab=0.00  disagree=0.00  wrong@12: 13/13*
  Model         |  0s   3s   6s   9s  12s
  --------------|------------------------------
  Q1.7B         | NON✗  NON✗  NON✗  NON✗  NON✗
  Ph4-mini      | NON✗  NON✗  NON✗  NON✗  NON✗
  Nem4B         | NON✗  NON✗  NON✗  NON✗  NON✗
  Q4B           | NON✗  NON✗  NON✗  NON✗  NON✗
  Q4B-i         | NON✗  NON✗  NON✗  NON✗  NON✗
  Llama8B       | NON✗  NON✗  NON✗  NON✗  NON✗
  DS-Llama8     | NON✗  NON✗  NON✗  NON✗  NON✗
  Q8B           | NON✗  NON✗  NON✗  NON✗  NON✗
  DS-Q14        | NON✗  NON✗  NON✗  NON✗  NON✗
  Q14B          | NON✗  NON✗  NON✗  NON✗  NON✗
  Ph4           | NON✗  NON✗  NON✗  NON✗  NON✗
  GPT5m         | NON✗  NON✗  NON✗  NON✗  NON✗
  GPT5          | NON✗  NON✗  NON✗  NON✗  NON✗

### doc 10042 · Atheism · Gold: **AGAINST**
> Watch @Dame_Lillard bring the Blazers to the playoff #Beast #SemST
*err=1.00  instab=0.00  disagree=0.00  wrong@12: 13/13*
  Model         |  0s   3s   6s   9s  12s
  --------------|------------------------------
  Q1.7B         | NON✗  NON✗  NON✗  NON✗  NON✗
  Ph4-mini      | NON✗  NON✗  NON✗  NON✗  NON✗
  Nem4B         | NON✗  NON✗  NON✗  NON✗  NON✗
  Q4B           | NON✗  NON✗  NON✗  NON✗  NON✗
  Q4B-i         | NON✗  NON✗  NON✗  NON✗  NON✗
  Llama8B       | NON✗  NON✗  NON✗  NON✗  NON✗
  DS-Llama8     | NON✗  NON✗  NON✗  NON✗  NON✗
  Q8B           | NON✗  NON✗  NON✗  NON✗  NON✗
  DS-Q14        | NON✗  NON✗  NON✗  NON✗  NON✗
  Q14B          | NON✗  NON✗  NON✗  NON✗  NON✗
  Ph4           | NON✗  NON✗  NON✗  NON✗  NON✗
  GPT5m         | NON✗  NON✗  NON✗  NON✗  NON✗
  GPT5          | NON✗  NON✗  NON✗  NON✗  NON✗

### doc 10043 · Atheism · Gold: **AGAINST**
> The Irish national school system is secular under law. We can reaffirm secularism by going through the courts! @HumanismIreland #S…
*err=1.00  instab=1.39  disagree=0.73  wrong@12: 13/13*
  Model         |  0s   3s   6s   9s  12s
  --------------|------------------------------
  Q1.7B         | FAV✗  FAV✗  NON✗  NON✗  NON✗
  Ph4-mini      | FAV✗  FAV✗  FAV✗  FAV✗  FAV✗
  Nem4B         | FAV✗  FAV✗  FAV✗  FAV✗  FAV✗
  Q4B           | NON✗  FAV✗  NON✗  FAV✗  NON✗
  Q4B-i         | FAV✗  FAV✗  FAV✗  FAV✗  FAV✗
  Llama8B       | NON✗  FAV✗  FAV✗  FAV✗  FAV✗
  DS-Llama8     | NON✗  FAV✗  NON✗  FAV✗  NON✗
  Q8B           | NON✗  FAV✗  FAV✗  NON✗  FAV✗
  DS-Q14        | FAV✗  FAV✗  NON✗  NON✗  FAV✗
  Q14B          | NON✗  FAV✗  FAV✗  NON✗  NON✗
  Ph4           | NON✗  FAV✗  FAV✗  FAV✗  FAV✗
  GPT5m         | FAV✗  FAV✗  FAV✗  FAV✗  FAV✗
  GPT5          | FAV✗  FAV✗  FAV✗  FAV✗  FAV✗

### doc 10046 · Atheism · Gold: **AGAINST**
> Currently doing a piece on logical fallacies for my blog @weebly #philosophy #logicism #blog #humanist #SemST
*err=1.00  instab=0.15  disagree=0.08  wrong@12: 13/13*
  Model         |  0s   3s   6s   9s  12s
  --------------|------------------------------
  Q1.7B         | NON✗  NON✗  NON✗  NON✗  NON✗
  Ph4-mini      | NON✗  NON✗  NON✗  NON✗  NON✗
  Nem4B         | NON✗  FAV✗  NON✗  NON✗  NON✗
  Q4B           | NON✗  NON✗  NON✗  NON✗  NON✗
  Q4B-i         | NON✗  NON✗  NON✗  NON✗  NON✗
  Llama8B       | NON✗  NON✗  NON✗  NON✗  NON✗
  DS-Llama8     | NON✗  NON✗  NON✗  NON✗  NON✗
  Q8B           | NON✗  NON✗  NON✗  NON✗  NON✗
  DS-Q14        | NON✗  NON✗  NON✗  NON✗  NON✗
  Q14B          | NON✗  NON✗  NON✗  NON✗  NON✗
  Ph4           | NON✗  NON✗  NON✗  NON✗  NON✗
  GPT5m         | NON✗  NON✗  NON✗  NON✗  NON✗
  GPT5          | NON✗  NON✗  NON✗  NON✗  NON✗

### doc 10050 · Atheism · Gold: **AGAINST**
> I officiate #humanist & most #nonreligious weddings-starting at $99. Ask me about #student discounts! #216 #celebrant #weddings #S…
*err=1.00  instab=0.77  disagree=0.92  wrong@12: 13/13*
  Model         |  0s   3s   6s   9s  12s
  --------------|------------------------------
  Q1.7B         | NON✗  FAV✗  NON✗  NON✗  NON✗
  Ph4-mini      | NON✗  NON✗  NON✗  NON✗  NON✗
  Nem4B         | NON✗  FAV✗  FAV✗  FAV✗  FAV✗
  Q4B           | NON✗  FAV✗  FAV✗  FAV✗  NON✗
  Q4B-i         | FAV✗  FAV✗  FAV✗  FAV✗  FAV✗
  Llama8B       | NON✗  FAV✗  FAV✗  FAV✗  FAV✗
  DS-Llama8     | NON✗  NON✗  NON✗  NON✗  NON✗
  Q8B           | NON✗  NON✗  FAV✗  FAV✗  FAV✗
  DS-Q14        | NON✗  NON✗  NON✗  NON✗  NON✗
  Q14B          | NON✗  NON✗  FAV✗  NON✗  NON✗
  Ph4           | NON✗  FAV✗  FAV✗  FAV✗  FAV✗
  GPT5m         | FAV✗  FAV✗  FAV✗  FAV✗   ?? 
  GPT5          | FAV✗  FAV✗  FAV✗  FAV✗  FAV✗

### doc 10054 · Atheism · Gold: **AGAINST**
> RT @br_holden: Atheists have every right to exist in the public sphere.   #freethinker #SemST
*err=1.00  instab=0.00  disagree=0.00  wrong@12: 13/13*
  Model         |  0s   3s   6s   9s  12s
  --------------|------------------------------
  Q1.7B         | FAV✗  FAV✗  FAV✗  FAV✗  FAV✗
  Ph4-mini      | FAV✗  FAV✗  FAV✗  FAV✗  FAV✗
  Nem4B         | FAV✗  FAV✗  FAV✗  FAV✗  FAV✗
  Q4B           | FAV✗  FAV✗  FAV✗  FAV✗  FAV✗
  Q4B-i         | FAV✗  FAV✗  FAV✗  FAV✗  FAV✗
  Llama8B       | FAV✗  FAV✗  FAV✗  FAV✗  FAV✗
  DS-Llama8     | FAV✗  FAV✗  FAV✗  FAV✗  FAV✗
  Q8B           | FAV✗  FAV✗  FAV✗  FAV✗  FAV✗
  DS-Q14        | FAV✗  FAV✗  FAV✗  FAV✗  FAV✗
  Q14B          | FAV✗  FAV✗  FAV✗  FAV✗  FAV✗
  Ph4           | FAV✗  FAV✗  FAV✗  FAV✗  FAV✗
  GPT5m         | FAV✗  FAV✗  FAV✗  FAV✗  FAV✗
  GPT5          | FAV✗  FAV✗  FAV✗  FAV✗  FAV✗


---
## B2 · SLM-Only Hard

SLM error > 0.5 and LLM error < 0.3 — capability gap examples.

### doc 11123 · Legalization of Abortion · Gold: **FAVOR**
> Yea, let's make a woman suffer 18 years and raise a child with an unfit parent who hates it because of "morality". #SemST
*err=0.68  instab=0.39  disagree=0.90  wrong@12: 9/13  SLM-err=0.95  LLM-err=0.24*
  Model         |  0s   3s   6s   9s  12s
  --------------|------------------------------
  Q1.7B         | FAV✓  AGN✗  AGN✗  AGN✗  AGN✗
  Ph4-mini      | AGN✗  AGN✗  AGN✗  AGN✗  AGN✗
  Nem4B         | AGN✗  AGN✗  AGN✗  AGN✗  AGN✗
  Q4B           | AGN✗  AGN✗  FAV✓  AGN✗  AGN✗
  Q4B-i         | AGN✗  AGN✗  AGN✗  AGN✗  AGN✗
  Llama8B       | AGN✗  AGN✗  AGN✗  AGN✗  AGN✗
  DS-Llama8     | AGN✗  AGN✗  AGN✗  AGN✗  AGN✗
  Q8B           | AGN✗  AGN✗  AGN✗  AGN✗  AGN✗
  DS-Q14        | AGN✗  AGN✗  AGN✗  AGN✗  AGN✗
  Q14B          | FAV✓  FAV✓  FAV✓  FAV✓  FAV✓
  Ph4           | FAV✓  FAV✓  AGN✗  FAV✓  FAV✓
  GPT5m         | FAV✓  FAV✓  FAV✓  FAV✓  FAV✓
  GPT5          | FAV✓  FAV✓  FAV✓  FAV✓  FAV✓

### doc 10899 · Hillary Clinton · Gold: **AGAINST**
> @HillaryClinton : Women deserve a better candidate for the HIGH HONOR of first woman President. We ALL do! #SemST
*err=0.69  instab=0.46  disagree=0.95  wrong@12: 9/13  SLM-err=0.94  LLM-err=0.28*
  Model         |  0s   3s   6s   9s  12s
  --------------|------------------------------
  Q1.7B         | FAV✗  FAV✗  FAV✗  FAV✗  FAV✗
  Ph4-mini      | FAV✗  FAV✗  FAV✗  FAV✗  FAV✗
  Nem4B         | FAV✗  FAV✗  FAV✗  FAV✗  FAV✗
  Q4B           | FAV✗  AGN✓   ??   FAV✗  FAV✗
  Q4B-i         | FAV✗  FAV✗  FAV✗  FAV✗  FAV✗
  Llama8B       | AGN✓  FAV✗  FAV✗  FAV✗  FAV✗
  DS-Llama8     | FAV✗  FAV✗  FAV✗  FAV✗  FAV✗
  Q8B           | NON✗  FAV✗   ??   FAV✗  FAV✗
  DS-Q14        | FAV✗  FAV✗  FAV✗  FAV✗  FAV✗
  Q14B          | AGN✓  AGN✓  FAV✗  FAV✗  AGN✓
  Ph4           | AGN✓  AGN✓  AGN✓  AGN✓  AGN✓
  GPT5m         | AGN✓  AGN✓  AGN✓  AGN✓  AGN✓
  GPT5          | AGN✓  AGN✓  AGN✓  AGN✓  AGN✓

### doc 10298 · Climate Change is a Real Concern · Gold: **FAVOR**
> @FoxNews LOL. Too bad all the other scientists agree with him. #ClimateChangeDenier #SemST
*err=0.66  instab=0.69  disagree=1.11  wrong@12: 7/13  SLM-err=0.93  LLM-err=0.24*
  Model         |  0s   3s   6s   9s  12s
  --------------|------------------------------
  Q1.7B         | AGN✗  NON✗  NON✗  AGN✗  AGN✗
  Ph4-mini      | AGN✗  AGN✗  AGN✗  AGN✗  AGN✗
  Nem4B         | AGN✗  AGN✗  AGN✗  AGN✗  FAV✓
  Q4B           | AGN✗  AGN✗  FAV✓  AGN✗  AGN✗
  Q4B-i         | AGN✗  AGN✗  AGN✗  AGN✗  AGN✗
  Llama8B       | AGN✗  AGN✗  AGN✗  AGN✗  AGN✗
  DS-Llama8     | AGN✗  AGN✗  AGN✗  AGN✗  FAV✓
  Q8B           | AGN✗  AGN✗  AGN✗  AGN✗  AGN✗
  DS-Q14        | FAV✓  AGN✗  AGN✗  AGN✗  NON✗
  Q14B          | AGN✗  AGN✗  FAV✓  FAV✓  FAV✓
  Ph4           | FAV✓  FAV✓  FAV✓  FAV✓  FAV✓
  GPT5m         | FAV✓  FAV✓  FAV✓  FAV✓  FAV✓
  GPT5          | FAV✓  FAV✓  FAV✓  FAV✓  FAV✓

### doc 11115 · Legalization of Abortion · Gold: **FAVOR**
> #antichoice NEVER try to police men over sex, but punishing women puts  them into a frenzy. #SemST
*err=0.66  instab=1.08  disagree=1.03  wrong@12: 8/13  SLM-err=0.90  LLM-err=0.28*
  Model         |  0s   3s   6s   9s  12s
  --------------|------------------------------
  Q1.7B         | NON✗  AGN✗  AGN✗  AGN✗  AGN✗
  Ph4-mini      | NON✗  AGN✗  AGN✗  AGN✗  AGN✗
  Nem4B         | AGN✗  AGN✗  AGN✗  AGN✗  AGN✗
  Q4B           | AGN✗  AGN✗  AGN✗  AGN✗  AGN✗
  Q4B-i         | AGN✗  AGN✗  AGN✗  AGN✗  AGN✗
  Llama8B       | AGN✗  AGN✗  FAV✓  FAV✓  FAV✓
  DS-Llama8     | NON✗  AGN✗  AGN✗  AGN✗  AGN✗
  Q8B           | AGN✗  AGN✗  FAV✓  AGN✗  AGN✗
  DS-Q14        | AGN✗  FAV✓  FAV✓  AGN✗  AGN✗
  Q14B          | FAV✓  FAV✓  FAV✓  AGN✗  FAV✓
  Ph4           | AGN✗  FAV✓  FAV✓  FAV✓  FAV✓
  GPT5m         | AGN✗  FAV✓  AGN✗  FAV✓  FAV✓
  GPT5          | FAV✓  FAV✓  FAV✓  FAV✓  FAV✓

### doc 11124 · Legalization of Abortion · Gold: **FAVOR**
> Pro-life laws only effect women so anti choicers are fine with these laws because the laws are only hurting are females  #SemST
*err=0.66  instab=0.77  disagree=1.19  wrong@12: 8/13  SLM-err=0.90  LLM-err=0.28*
  Model         |  0s   3s   6s   9s  12s
  --------------|------------------------------
  Q1.7B         | NON✗  NON✗  AGN✗  NON✗  NON✗
  Ph4-mini      | AGN✗  AGN✗  AGN✗  AGN✗  AGN✗
  Nem4B         | AGN✗  FAV✓  AGN✗  AGN✗  AGN✗
  Q4B           | AGN✗  AGN✗  AGN✗  AGN✗  AGN✗
  Q4B-i         | AGN✗  AGN✗  AGN✗  AGN✗  AGN✗
  Llama8B       | AGN✗  AGN✗  AGN✗  AGN✗  FAV✓
  DS-Llama8     | AGN✗  AGN✗  AGN✗  AGN✗  AGN✗
  Q8B           | FAV✓  AGN✗  FAV✓  AGN✗  AGN✗
  DS-Q14        | AGN✗  AGN✗  AGN✗  AGN✗  FAV✓
  Q14B          | FAV✓  FAV✓  FAV✓  FAV✓  FAV✓
  Ph4           | AGN✗  AGN✗  AGN✗  FAV✓  FAV✓
  GPT5m         | FAV✓  FAV✓  FAV✓  FAV✓   ?? 
  GPT5          | FAV✓  FAV✓  FAV✓  FAV✓  FAV✓

### doc 11197 · Legalization of Abortion · Gold: **AGAINST**
> @GBPstaff 1999 Meet the Press admitted to being Very Pro-Choice even with late term & Partial Birth Abortions.  He's sick!  #SemST
*err=0.65  instab=0.15  disagree=0.92  wrong@12: 8/13  SLM-err=0.90  LLM-err=0.24*
  Model         |  0s   3s   6s   9s  12s
  --------------|------------------------------
  Q1.7B         | FAV✗  FAV✗  FAV✗  FAV✗  FAV✗
  Ph4-mini      | FAV✗  FAV✗  FAV✗  FAV✗  FAV✗
  Nem4B         | FAV✗  AGN✓  AGN✓  AGN✓  AGN✓
  Q4B           | FAV✗  FAV✗  FAV✗  FAV✗  FAV✗
  Q4B-i         | FAV✗  FAV✗  FAV✗  FAV✗  FAV✗
  Llama8B       | FAV✗  FAV✗  FAV✗  FAV✗  FAV✗
  DS-Llama8     | FAV✗  FAV✗  FAV✗  FAV✗  FAV✗
  Q8B           | FAV✗  FAV✗  FAV✗  FAV✗  FAV✗
  DS-Q14        | FAV✗  FAV✗  FAV✗  FAV✗  FAV✗
  Q14B          | FAV✗  AGN✓  AGN✓  AGN✓  AGN✓
  Ph4           | AGN✓  AGN✓  AGN✓  AGN✓  AGN✓
  GPT5m         | AGN✓  AGN✓  AGN✓  AGN✓  AGN✓
  GPT5          | AGN✓  AGN✓  AGN✓  AGN✓  AGN✓

### doc 10948 · Hillary Clinton · Gold: **AGAINST**
> @lylafmills Simple. A revolution Two Independence Days And a clean slate. #2ndamendment #REMEMBERBENGHAZI2016 #PATRIOTSWILLRISE #S…
*err=0.63  instab=0.69  disagree=1.41  wrong@12: 8/13  SLM-err=0.88  LLM-err=0.24*
  Model         |  0s   3s   6s   9s  12s
  --------------|------------------------------
  Q1.7B         | NON✗  NON✗  NON✗  NON✗  NON✗
  Ph4-mini      | NON✗  FAV✗  FAV✗  FAV✗  FAV✗
  Nem4B         | FAV✗  FAV✗  AGN✓  AGN✓  AGN✓
  Q4B           | NON✗  NON✗  NON✗  NON✗  NON✗
  Q4B-i         | NON✗  NON✗  NON✗  NON✗  NON✗
  Llama8B       | AGN✓  FAV✗  FAV✗  FAV✗  FAV✗
  DS-Llama8     | NON✗  NON✗  AGN✓  NON✗  NON✗
  Q8B           | NON✗  NON✗  NON✗  NON✗  NON✗
  DS-Q14        | NON✗  AGN✓  AGN✓  AGN✓  AGN✓
  Q14B          | NON✗  NON✗  NON✗  AGN✓  NON✗
  Ph4           | NON✗  AGN✓  AGN✓  AGN✓  AGN✓
  GPT5m         | AGN✓  AGN✓  AGN✓  AGN✓  AGN✓
  GPT5          | AGN✓  AGN✓  AGN✓  AGN✓  AGN✓

### doc 10038 · Atheism · Gold: **AGAINST**
> You stay the same through the ages, Your love never changes...Your love never fails. #SemST
*err=0.61  instab=1.23  disagree=0.96  wrong@12: 7/13  SLM-err=0.85  LLM-err=0.24*
  Model         |  0s   3s   6s   9s  12s
  --------------|------------------------------
  Q1.7B         | NON✗  FAV✗  NON✗  NON✗  NON✗
  Ph4-mini      | NON✗  NON✗  NON✗  NON✗  NON✗
  Nem4B         | NON✗  NON✗  NON✗  NON✗  NON✗
  Q4B           | NON✗  AGN✓  NON✗  NON✗  NON✗
  Q4B-i         | NON✗  AGN✓  NON✗  NON✗  NON✗
  Llama8B       | NON✗  FAV✗  NON✗  NON✗  NON✗
  DS-Llama8     | NON✗  NON✗  NON✗  NON✗  NON✗
  Q8B           | NON✗  AGN✓  AGN✓  AGN✓  AGN✓
  DS-Q14        | NON✗  AGN✓  AGN✓  NON✗  AGN✓
  Q14B          | NON✗  AGN✓  NON✗  NON✗  AGN✓
  Ph4           | NON✗  AGN✓  AGN✓  AGN✓  AGN✓
  GPT5m         | AGN✓  AGN✓  AGN✓  AGN✓  AGN✓
  GPT5          | AGN✓  AGN✓  AGN✓  AGN✓  AGN✓

### doc 10474 · Feminist Movement · Gold: **FAVOR**
> We live in a sad world when wanting equality makes you a troll... #SemST
*err=0.57  instab=0.54  disagree=1.39  wrong@12: 8/13  SLM-err=0.85  LLM-err=0.12*
  Model         |  0s   3s   6s   9s  12s
  --------------|------------------------------
  Q1.7B         | NON✗  NON✗  NON✗  NON✗  NON✗
  Ph4-mini      | AGN✗  AGN✗  AGN✗  AGN✗  AGN✗
  Nem4B         | AGN✗  AGN✗  AGN✗  AGN✗  AGN✗
  Q4B           | AGN✗  AGN✗  AGN✗  AGN✗  AGN✗
  Q4B-i         | AGN✗  AGN✗  AGN✗  AGN✗  AGN✗
  Llama8B       | AGN✗  FAV✓  FAV✓  FAV✓  FAV✓
  DS-Llama8     | AGN✗  AGN✗  AGN✗  AGN✗  AGN✗
  Q8B           | FAV✓  AGN✗  NON✗  FAV✓  AGN✗
  DS-Q14        | NON✗  NON✗  FAV✓  FAV✓  FAV✓
  Q14B          | FAV✓  FAV✓  FAV✓  FAV✓  AGN✗
  Ph4           | FAV✓  FAV✓  FAV✓  FAV✓  FAV✓
  GPT5m         | FAV✓  FAV✓  FAV✓  FAV✓  FAV✓
  GPT5          | FAV✓  FAV✓  FAV✓  FAV✓  FAV✓

### doc 10158 · Atheism · Gold: **FAVOR**
> How do they come up with this insanity? Oh wait, they believe in a Sky-God.  #fyilive, #SemST
*err=0.58  instab=0.46  disagree=1.42  wrong@12: 7/13  SLM-err=0.82  LLM-err=0.20*
  Model         |  0s   3s   6s   9s  12s
  --------------|------------------------------
  Q1.7B         | NON✗  NON✗  NON✗  NON✗  NON✗
  Ph4-mini      | AGN✗  AGN✗  AGN✗  AGN✗  AGN✗
  Nem4B         | AGN✗  AGN✗  AGN✗  AGN✗  AGN✗
  Q4B           | AGN✗  FAV✓  FAV✓  FAV✓  FAV✓
  Q4B-i         | AGN✗  AGN✗  AGN✗  AGN✗  AGN✗
  Llama8B       | AGN✗  FAV✓  FAV✓  AGN✗  AGN✗
  DS-Llama8     | NON✗  AGN✗  NON✗  NON✗  NON✗
  Q8B           | AGN✗  AGN✗  AGN✗  AGN✗  FAV✓
  DS-Q14        | FAV✓  FAV✓  FAV✓  FAV✓  FAV✓
  Q14B          | FAV✓  FAV✓  FAV✓  FAV✓  FAV✓
  Ph4           | AGN✗  AGN✗  AGN✗  AGN✗  AGN✗
  GPT5m         | FAV✓  FAV✓  FAV✓  FAV✓  FAV✓
  GPT5          | FAV✓  FAV✓  FAV✓  FAV✓  FAV✓


---
## B3 · Persistently Unstable

6 or more models have trajectory=unstable (oscillate across shot counts).

### doc 10446 · Feminist Movement · Gold: **AGAINST**
> It wouldn't be a #FIFA event without women in tight dresses. Wrong place. Wrong time. #womensworldcup #SemST
*err=0.65  instab=1.92  disagree=1.40  wrong@12: 10/13  unstable_models=8*
  Model         |  0s   3s   6s   9s  12s
  --------------|------------------------------
  Q1.7B         | NON✗  NON✗  NON✗  NON✗  NON✗
  Ph4-mini      | FAV✗  NON✗  NON✗  FAV✗  NON✗
  Nem4B         | AGN✓  FAV✗  AGN✓  AGN✓  NON✗
  Q4B           | FAV✗  NON✗  FAV✗  NON✗  FAV✗
  Q4B-i         | FAV✗  FAV✗  FAV✗  NON✗  NON✗
  Llama8B       | AGN✓  AGN✓  AGN✓  AGN✓  AGN✓
  DS-Llama8     | NON✗  FAV✗  NON✗  NON✗  NON✗
  Q8B           | AGN✓  AGN✓  NON✗  AGN✓  NON✗
  DS-Q14        | NON✗  NON✗  NON✗  AGN✓  NON✗
  Q14B          | NON✗  NON✗  AGN✓  NON✗  NON✗
  Ph4           | AGN✓  NON✗  NON✗  AGN✓  NON✗
  GPT5m         | FAV✗  AGN✓  AGN✓  AGN✓  AGN✓
  GPT5          | FAV✗  AGN✓  AGN✓  AGN✓  AGN✓

### doc 10067 · Atheism · Gold: **AGAINST**
> What really builds a relationship is when we relate to what we don't understand about God.  #Emunah #SemST
*err=0.48  instab=1.77  disagree=1.36  wrong@12: 6/13  unstable_models=8*
  Model         |  0s   3s   6s   9s  12s
  --------------|------------------------------
  Q1.7B         | NON✗  FAV✗  NON✗  NON✗  NON✗
  Ph4-mini      | NON✗  FAV✗  FAV✗  NON✗  NON✗
  Nem4B         | NON✗  AGN✓  AGN✓  NON✗  AGN✓
  Q4B           | AGN✓  AGN✓  AGN✓  AGN✓  AGN✓
  Q4B-i         | FAV✗  FAV✗  AGN✓  AGN✓  NON✗
  Llama8B       | FAV✗  FAV✗  FAV✗  FAV✗  FAV✗
  DS-Llama8     | NON✗  AGN✓  NON✗  NON✗  NON✗
  Q8B           | AGN✓  AGN✓  AGN✓  AGN✓  AGN✓
  DS-Q14        | NON✗  AGN✓  NON✗  AGN✓  NON✗
  Q14B          | AGN✓  NON✗  AGN✓  NON✗  AGN✓
  Ph4           | AGN✓  NON✗  AGN✓  NON✗  AGN✓
  GPT5m         | AGN✓  AGN✓  AGN✓  AGN✓  AGN✓
  GPT5          | AGN✓  AGN✓  AGN✓  AGN✓  AGN✓

### doc 10206 · Atheism · Gold: **FAVOR**
> AtheistQ "God's guidance is the only excuse for a man who can't find a better justification for his actions. #SemST
*err=0.54  instab=1.77  disagree=1.49  wrong@12: 8/13  unstable_models=9*
  Model         |  0s   3s   6s   9s  12s
  --------------|------------------------------
  Q1.7B         | NON✗  NON✗  NON✗  NON✗  NON✗
  Ph4-mini      | AGN✗  AGN✗  AGN✗  AGN✗  AGN✗
  Nem4B         | AGN✗  FAV✓  AGN✗  AGN✗  AGN✗
  Q4B           | AGN✗  FAV✓  FAV✓  NON✗  AGN✗
  Q4B-i         | AGN✗  FAV✓  FAV✓  AGN✗  FAV✓
  Llama8B       | AGN✗  FAV✓  FAV✓  FAV✓  AGN✗
  DS-Llama8     | FAV✓  NON✗  AGN✗  NON✗  NON✗
  Q8B           | FAV✓  AGN✗  FAV✓  FAV✓  FAV✓
  DS-Q14        | AGN✗  FAV✓  NON✗  FAV✓  FAV✓
  Q14B          | NON✗  NON✗  NON✗  FAV✓  NON✗
  Ph4           | FAV✓  AGN✗  FAV✓  AGN✗  AGN✗
  GPT5m         | FAV✓  FAV✓  FAV✓  FAV✓  FAV✓
  GPT5          | FAV✓  FAV✓  FAV✓  FAV✓  FAV✓

### doc 10544 · Feminist Movement · Gold: **AGAINST**
> Not catering to a woman that doesn't bring = amount to the table as I do. #equality  #SemST
*err=0.61  instab=1.77  disagree=1.51  wrong@12: 8/13  unstable_models=8*
  Model         |  0s   3s   6s   9s  12s
  --------------|------------------------------
  Q1.7B         | NON✗  NON✗  NON✗  NON✗  AGN✓
  Ph4-mini      | FAV✗  FAV✗  FAV✗  AGN✓  FAV✗
  Nem4B         | AGN✓  AGN✓  FAV✗  AGN✓  AGN✓
  Q4B           | AGN✓  AGN✓  FAV✗  AGN✓  NON✗
  Q4B-i         | FAV✗  AGN✓  AGN✓  AGN✓  AGN✓
  Llama8B       | AGN✓  AGN✓  AGN✓  AGN✓  AGN✓
  DS-Llama8     | FAV✗  NON✗  AGN✓  NON✗  NON✗
  Q8B           | FAV✗  NON✗  NON✗  FAV✗  FAV✗
  DS-Q14        | FAV✗  NON✗  NON✗  AGN✓  NON✗
  Q14B          | NON✗  NON✗  NON✗  NON✗  NON✗
  Ph4           | NON✗  AGN✓  AGN✓  AGN✓  FAV✗
  GPT5m         | FAV✗  FAV✗  FAV✗  FAV✗  FAV✗
  GPT5          | AGN✓  NON✗  FAV✗  NON✗  AGN✓

### doc 10669 · Feminist Movement · Gold: **FAVOR**
> The more you push me, the more I resist. Don't bother #strongwomen #women #SemST
*err=0.63  instab=1.77  disagree=1.46  wrong@12: 9/13  unstable_models=8*
  Model         |  0s   3s   6s   9s  12s
  --------------|------------------------------
  Q1.7B         | NON✗  NON✗  NON✗  NON✗  NON✗
  Ph4-mini      | NON✗  FAV✓  FAV✓  AGN✗  FAV✓
  Nem4B         | AGN✗  AGN✗  AGN✗  AGN✗  AGN✗
  Q4B           | FAV✓  AGN✗  FAV✓  FAV✓  NON✗
  Q4B-i         | FAV✓  FAV✓  FAV✓  AGN✗  FAV✓
  Llama8B       | FAV✓  FAV✓  AGN✗  FAV✓  FAV✓
  DS-Llama8     | FAV✓  NON✗  NON✗  AGN✗  NON✗
  Q8B           | NON✗  AGN✗  AGN✗  AGN✗  NON✗
  DS-Q14        | FAV✓  NON✗  NON✗  NON✗  NON✗
  Q14B          | NON✗  FAV✓  AGN✗  NON✗  NON✗
  Ph4           | AGN✗  NON✗  AGN✗  NON✗  NON✗
  GPT5m         | FAV✓  FAV✓  FAV✓  FAV✓  FAV✓
  GPT5          | FAV✓  FAV✓  NON✗  NON✗  NON✗

### doc 10109 · Atheism · Gold: **AGAINST**
> Steppin' out in faith and love.....have a safe and pleasant day, tweeple!  #blessed #love #happyday #SemST
*err=0.57  instab=1.69  disagree=1.26  wrong@12: 6/13  unstable_models=7*
  Model         |  0s   3s   6s   9s  12s
  --------------|------------------------------
  Q1.7B         | NON✗  FAV✗  NON✗  FAV✗  NON✗
  Ph4-mini      | NON✗  NON✗  FAV✗  NON✗  NON✗
  Nem4B         | NON✗  FAV✗  FAV✗  NON✗  NON✗
  Q4B           | AGN✓  AGN✓  NON✗  AGN✓  AGN✓
  Q4B-i         | NON✗  AGN✓  AGN✓  AGN✓  AGN✓
  Llama8B       | NON✗  FAV✗  AGN✓  FAV✗  FAV✗
  DS-Llama8     | NON✗  NON✗  NON✗  NON✗  NON✗
  Q8B           | NON✗  AGN✓  AGN✓  AGN✓  AGN✓
  DS-Q14        | NON✗  NON✗  AGN✓  NON✗  AGN✓
  Q14B          | NON✗  AGN✓  NON✗  NON✗  NON✗
  Ph4           | NON✗  AGN✓  AGN✓  AGN✓  AGN✓
  GPT5m         | AGN✓  AGN✓  AGN✓  AGN✓  AGN✓
  GPT5          | NON✗  NON✗  AGN✓  AGN✓  AGN✓

### doc 10594 · Feminist Movement · Gold: **AGAINST**
> girl: *bikini photo* girls: slut Justin Bieber: *butt naked* girls: " "lemme see the front" LOL BUT YOU'RE A FEMINIST #SemST
*err=0.39  instab=1.69  disagree=1.21  wrong@12: 4/13  unstable_models=6*
  Model         |  0s   3s   6s   9s  12s
  --------------|------------------------------
  Q1.7B         | NON✗  NON✗  NON✗  AGN✓  AGN✓
  Ph4-mini      | AGN✓  AGN✓  AGN✓  AGN✓  AGN✓
  Nem4B         | AGN✓  NON✗  NON✗  NON✗  NON✗
  Q4B           | AGN✓  NON✗  FAV✗  AGN✓  AGN✓
  Q4B-i         | AGN✓  AGN✓  AGN✓  AGN✓  AGN✓
  Llama8B       | AGN✓  AGN✓  AGN✓  AGN✓  AGN✓
  DS-Llama8     | FAV✗  NON✗  FAV✗  AGN✓  NON✗
  Q8B           | AGN✓  NON✗  AGN✓  NON✗  AGN✓
  DS-Q14        | NON✗  NON✗  AGN✓  NON✗  AGN✓
  Q14B          | AGN✓  AGN✓  AGN✓  AGN✓  FAV✗
  Ph4           | AGN✓  NON✗  NON✗  AGN✓  AGN✓
  GPT5m         | AGN✓  AGN✓  AGN✓  AGN✓  AGN✓
  GPT5          | AGN✓  FAV✗  NON✗  FAV✗  FAV✗

### doc 10353 · Climate Change is a Real Concern · Gold: **NONE**
> @WEETFEvents Jonathan Tillson from @DefraGovUK highlights how economic stability will be undermined by #SemST
*err=0.54  instab=1.61  disagree=1.21  wrong@12: 8/13  unstable_models=7*
  Model         |  0s   3s   6s   9s  12s
  --------------|------------------------------
  Q1.7B         | NON✓  NON✓  NON✓  NON✓  NON✓
  Ph4-mini      | NON✓  FAV✗  AGN✗  FAV✗  NON✓
  Nem4B         | NON✓  FAV✗  NON✓  NON✓  NON✓
  Q4B           | FAV✗  FAV✗  FAV✗  NON✓  FAV✗
  Q4B-i         | NON✓  NON✓  NON✓  AGN✗  FAV✗
  Llama8B       | AGN✗  AGN✗  AGN✗  FAV✗  FAV✗
  DS-Llama8     | NON✓  NON✓  NON✓  FAV✗  AGN✗
  Q8B           | NON✓  NON✓  FAV✗  FAV✗  NON✓
  DS-Q14        | NON✓  NON✓  FAV✗  FAV✗  FAV✗
  Q14B          | NON✓  NON✓  FAV✗  NON✓  NON✓
  Ph4           | NON✓  FAV✗  FAV✗  FAV✗  FAV✗
  GPT5m         | NON✓  FAV✗  FAV✗  FAV✗  FAV✗
  GPT5          | NON✓  FAV✗  FAV✗  FAV✗  FAV✗


---
## Summary by Topic

| Topic | N | Majority-wrong@12 | Univ-hard | SLM-only-hard | Unstable≥6 | Mean-err | Mean-disagree |
|---|---|---|---|---|---|---|---|
| Atheism | 220 | 68 | 51 | 33 | 13 | 0.469 | 0.933 |
| Climate Change is a Real Concern | 169 | 42 | 27 | 5 | 9 | 0.308 | 0.587 |
| Feminist Movement | 285 | 84 | 59 | 20 | 23 | 0.380 | 0.787 |
| Hillary Clinton | 295 | 57 | 33 | 25 | 11 | 0.297 | 0.710 |
| Legalization of Abortion | 280 | 95 | 72 | 21 | 8 | 0.411 | 0.700 |
| **Total** | **1249** | **346** | **242** | **104** | **64** | **0.373** | **0.748** |