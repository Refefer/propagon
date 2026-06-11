# Node importance on a link graph

`articles` is a directed, weighted link graph among ~330 Wikipedia
articles (`src dst weight` per line, weight = link count) — century pages,
historical topics, and related categories. It is small enough to eyeball
and lumpy enough that the rankings genuinely disagree.

`./run.sh` runs every graph algorithm. How to read the outputs:

- **page-rank** — global importance under the random-surfer model;
  `--sink-dispersion` picks the dangling-page treatment (`uniform` is the
  textbook choice, `reverse`/`all` are propagon traditions, `none` lets
  mass drain).
- **hits** — two scores per node: *authorities* (pointed at by good hubs)
  then *hubs* (pointing at good authorities), blank-line separated.
- **katz-centrality** — all incoming walks, discounted by `--alpha`; keep
  α below `1/λ_max` or the run aborts with advice (this graph's weights
  are large — hence `--alpha 0.05`).
- **degree** — the counting baseline. If page-rank's order barely differs
  from in-degree's, the link structure isn't adding much signal.
- **k-core** — coreness per node: which articles sit in the densely
  interlinked center vs. the periphery (direction/weights ignored).
- **birank** — two-sided scores treating the edges as a bipartite
  src↔dst interaction matrix.
- **components** — utility: splits the graph into connected components,
  one edge file each.

## Provenance

`articles` ships from propagon v1's example set (extracted from Wikipedia
link data); kept verbatim so old and new outputs stay comparable.
