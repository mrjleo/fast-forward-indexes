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

`fast_forward.ranking.Ranking.interpolate` allows to interpolate the scores of two rankings (i.e., retrieval and re-ranking):

```python
first_stage_ranking = Ranking.from_file(Path("/path/to/TREC/run.tsv"))
semantic_scores = my_index(first_stage_ranking)
interpolated_ranking = first_stage_ranking.interpolate(semantic_scores, 0.1)
```

Finally, a ranking can have a name set and be saved as a TREC runfile:

```python
interpolated_ranking.name = "Fast-Forward, alpha=0.1"
interpolated_ranking.save(Path("/path/to/output/run.tsv"))
```
