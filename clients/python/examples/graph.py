"""Graph centrality: PageRank and personalized PageRank.

Run after `maturin develop`:  python examples/graph.py
"""

import propagon


def main() -> None:
    g = propagon.GraphDataset()
    for src, dst in [
        ("home", "about"),
        ("home", "products"),
        ("about", "home"),
        ("products", "home"),
        ("products", "checkout"),
        ("checkout", "home"),
    ]:
        g.push(src, dst)

    print("Global PageRank:")
    for name, score in propagon.PageRank(damping=0.85).fit(g).top(3):
        print(f"  {name:10s} {score:.4f}")

    # Personalize the walk toward the checkout page.
    personalized = propagon.PageRank(
        teleport=propagon.Teleport.seeds([("checkout", 1.0)])
    ).fit(g)
    print("\nPersonalized toward 'checkout':")
    for name, score in personalized.top(3):
        print(f"  {name:10s} {score:.4f}")

    # Split a disconnected graph into independent components.
    components = propagon.extract_components(g, min_size=1)
    print(f"\nConnected components: {len(components)}")


if __name__ == "__main__":
    main()
