Pairwizer
===

Pairwizer is a binary which computes a variety of rankings using a few different algorithms.  It can scale up to 100s of millions of matches and 10s of millions of teams.

Algorithms Implemented
---

- [x] Bradley-Terry Model (Using MM Method)
- [x] Glicko2
- [x] Rate
- [ ] Bradley-Terry Model (Logistic Method)
- [ ] ELO
- [ ] TrueSkill
- [ ] Blade/Chest

Installation
---

You'll need the latest version of the Rust compiler [toolchain](http://www.rustup.rs).

    # Will add the `pairwizer` to ~/.cargo/bin
    cargo install -f --path .

Data Format
---

Data is expected in the following, line delimited format:

```
    ID_OF_WINNER ID_OF_LOSER [WEIGHT]
    ID_OF_WINNER ID_OF_LOSER [WEIGHT]
    ID_OF_WINNER ID_OF_LOSER [WEIGHT]
    
    ID_OF_WINNER ID_OF_LOSER [WEIGHT]
    ID_OF_WINNER ID_OF_LOSER [WEIGHT]
    ID_OF_WINNER ID_OF_LOSER [WEIGHT]
```

where weight is optional and only taken into consideration for Bradley-Terry Models.  Empty lines designate separate batch delimiters: in the case of `glicko2`, each batch will be considered an update against previous batches.  BTM and rate statistics will flatten multiple batches as they don't support updates.

To treat batches as completely independent rankings, users can use the `--groups-are-separate` in which case `pairwizer` will emit separate scores per batch.  This is useful when processing a large number of independent tournaments.

Example
---

We've provided the 2018 baseball season as an example dataset.  After installation:
    
    cd example
    bash run.sh

