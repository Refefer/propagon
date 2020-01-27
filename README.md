Propagon
===

Propagon is a CLI application which provides a suite of algorithms for processing graphs.  It can scale up to 100s of millions of edges and 10s of millions of nodes on a single laptop.

Algorithms Implemented
---

Propagon implements a selection of algorithms useful for a variety of use case.

### Tournament Ranking

Tournament ranking is useful for computing an absolute ordering of nodes according to wins and losses.  Use cases might include ranking sports teams, chess opponents, and more.

#### Bradley-Terry Model (MM Method)

Extremely fast BTM computation on pairs of games using the minorization-maximization method.  Works best with fully connected graphs though attempts have been made to stablize scores across disjoint graphs with random edges and other methods.  Requires a good number of comparisons between teams to learn accurate rankings.

Outputs a ranking score per node where higher is better.

#### Bradley-Terry Model (Logistic Method)

Computes BTM uses logistic regression.  While not as fast as the minorization method, it provides scores which are more natural to work with.  It also suffers less from disjoint graphs than the MM method.  As with the MM algorithm, it requires a good number of comparisons between teams to learn accurate rankings.

Outputs a ranking score per node where higher is better.

#### Glicko2

Implements the glicko2 ranking system for pairs of teams.  Generally provides good rankings.  This is rooted in Bayesian methodology, making it more robust to fewer games than the BTM models.

Outputs a ranking score per node where higher is better.

#### Rate

Computes rankings based on win/loss record, ignoring relative strength of the opponent.  This should only ever be used as a baseline since the above methods will almost certainly work better in nearly all cases.

### Node Importance

Node Importance attempts to learn authority measures of nodes within a directed graph.  In the classic case, Page Rank was used in search engines to learn which websites were considered most trustywortyh.


#### Page Rank

Computes Page Rank across a directed graph.  This can be usefully applied to generate popularity metrics or authority scores for nodes.

Outputs a score per node, where the higher the score the more "important" it is.

#### BiRank

Interesting algorithm for computing rankings of nodes.  Unlike PageRank, it only operates on symmetric graphs, but it provides greater flexibility on how importance is propagated.  In this implementation, we only support bipartite graphs.  Read the paper for additional information on how to formulate a problem to benefit from this algorithm.

Outputs a score per node, where the higher the score, the more "imporant" it is

He, Xiangnan, et al. "Birank: Towards ranking on bipartite graphs." IEEE Transactions on Knowledge and Data Engineering 29.1 (2016): 57-71.


### Vector Propagation

Useful for learning embeddings based on graph relationships.  Algorithms are semi-supervised, with some or all of the nodes providing some initial feature vector which are propagated through the graph.

#### VecProp

Experimental algorithm (paper pending) which generates node embeddings via a propagation algorithm.  The user provides two files: a weighted edges file and a "prior" file, which contains a subset of nodes and initial feature vectors, propagating the vectors across the graph to all nodes.  It's fast, converging fairly quickly, and produces intuitive embeddings for each node.  Think of it as a blend of VPCG [1] and node authority algorithms all graphs.  Works best for problems where network homophily is valuable.

Outputs an embedding, encoded as a json, for each node within the network.

[1] Jiang, Shan, et al. "Learning query and document relevance from a web-scale click graph." Proceedings of the 39th International ACM SIGIR conference on Research and Development in Information Retrieval. 2016.

#### VecWalk

VecWalk is another experimental take on VecProp, but instead uses random walks with a context window rather than local neighborhood to generate embeddings.  Like VecProp, it requires a "prior" file for consumption.  Think of this similar to places you might use DeepWalk [1], except it is semi-supervised and uses node averaging rather than contrastive divergence.  It is almost always slower than VecProp, but can also capture structural relationships due to the biased randomwalk.

Outputs an embedding, encoded as json, for each node within the network.

[1] Perozzi, Bryan, Rami Al-Rfou, and Steven Skiena. "Deepwalk: Online learning of social representations." Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining. 2014.

### Clustering

Given a graph, attempt to cluster nodes into different groups.  Currently, it only supports disjoint clusters.

#### LPA

Classice Label Propagation Algorithm.  Uses an unweighted, undirected graph to compute clusters of nodes.  Due to randomization, the results are stochastic and can change between runs.

Outputs the cluster number per node.  Nodes with the same cluster number belong to the same cluster.

#### LabelRank

An improved, stable version of LPA [1]. Supports weighted graphs and incorporates elements like momentum in the computation.  Can perform better than LPA, but requires more hyperparameter tuning and double the memory.

Outputs the cluster number per node.  Nodes with the same cluster number belong to the same cluster.

[1] Xie, Jierui, and Boleslaw K. Szymanski. "Labelrank: A stabilized label propagation algorithm for community detection in networks." 2013 IEEE 2nd Network Science Workshop (NSW). IEEE, 2013.

### Utility

Miscellaneous utility methods.

#### Random Walks

Generates random walks on weighted graphs.  This can be used by other algorithms, such as word2vec, to generate other types of embeddings.  This implementation scales much better than other implementations out there.

Installation
---

You'll need the latest version of the Rust compiler [toolchain](http://www.rustup.rs).

    # Will add the `propagon` to ~/.cargo/bin
    cargo install -f --path .

Data Format
---

Data for tournament rankings is expected as the following, line delimited format:

```
    ID_OF_WINNER ID_OF_LOSER [WEIGHT]
    ID_OF_WINNER ID_OF_LOSER [WEIGHT]
    
    ID_OF_WINNER ID_OF_LOSER [WEIGHT]
    ID_OF_WINNER ID_OF_LOSER [WEIGHT]
    ID_OF_WINNER ID_OF_LOSER [WEIGHT]
```

where weight is optional (assumed as 1 if omitted).  Empty lines designate separate batch delimiters: in the case of `glicko2`, each batch will be considered an update against previous batches.  BTM and rate statistics will flatten multiple batches as they don't support updates.

For graph algorithms, the format is interpretted as edges: 

```
    FROM_NODE_ID TO_NODE_ID [WEIGHT]
    FROM_NODE_ID TO_NODE_ID [WEIGHT]
    FROM_NODE_ID TO_NODE_ID [WEIGHT]
```

For graph algorithms operating on undirected graphs, edges will automatically be added in both directions.

For algorithms utilizing a "priors" file, such as the propagation algorithms:

```
    NODE_ID FEAT_1 FEAT_2 [...]
    NODE_ID FEAT_1 FEAT_2 [...]
    NODE_ID FEAT_1 FEAT_2 [...]
```

where FEAT is a token of interest.  Currently, the input format only support boolean features.

Example
---

We've provided the 2018 baseball season as an example dataset.  After installation:
    
    cd example
    bash run.sh

