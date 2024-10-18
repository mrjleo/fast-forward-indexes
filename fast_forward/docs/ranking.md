Rankings (or runs) are represented using the `fast_forward.ranking.Ranking` class. Each ranking contains query IDs, document/passage IDs, and corresponding scores. Rankings may be instantiated from TREC runfiles using `fast_forward.ranking.Ranking.from_file`:

```python
r = Ranking.from_file(Path("/path/to/TREC/run.tsv"))
```

Alternatively, rankings can be created from nested dictionaries using `fast_forward.ranking.Ranking.from_run`:

```python
run = {
    "q1": {"d0": 1, "d1": 2, "d2": 3},
    "q2": {"d0": 4, "d1": 5, "d2": 6},
}
r = Ranking.from_run(run)
```

Optionally, rankings can have queries attached using `fast_forward.ranking.Ranking.attach_queries`:

```python
r = r.attach_queries(
    {
        "q1": "query 1",
        "q2": "query 2",
        # ...
        "qn": "query n",
    }
)
```

Rankings implement addition and multiplication operators, for example:

```python
ranking_3 = 0.1 * ranking_1 + ranking_2
```

`fast_forward.ranking.Ranking.interpolate` allows to interpolate the scores of two rankings (i.e., retrieval and re-ranking):

```python
first_stage_ranking = Ranking.from_file(Path("/path/to/TREC/run.tsv"))
semantic_scores = my_index(first_stage_ranking)
interpolated_ranking = first_stage_ranking.interpolate(semantic_scores, 0.1)
# equivalent to (but slightly more efficient than):
interpolated_ranking = first_stage_ranking * 0.1 + semantic_scores * 0.9
```

Additionally, `fast_forward.ranking.Ranking.rr_scores` recomputes a ranking's scores based on the reciprocal rank. This allows, for example, to perform [reciprocal rank fusion (RRF)](https://dl.acm.org/doi/10.1145/1571941.1572114) as follows:

```python
rrf_ranking = first_stage_ranking.rr_scores() + semantic_scores.rr_scores()
```

Further information about rank fusion can be found [here](https://dl.acm.org/doi/10.1145/3596512).

Finally, a ranking can have a name set and be saved as a TREC runfile:

```python
interpolated_ranking.name = "Fast-Forward, alpha=0.1"
interpolated_ranking.save(Path("/path/to/output/run.tsv"))
```
