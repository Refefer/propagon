// Graph centrality: global and personalized PageRank, plus components.
//   node examples/graph.ts
import { datasets, graph, functions } from "../index.js";

const g = new datasets.GraphDataset();
for (const [src, dst] of [
  ["home", "about"],
  ["home", "products"],
  ["about", "home"],
  ["products", "home"],
  ["products", "checkout"],
  ["checkout", "home"],
]) {
  g.push(src, dst, 1);
}

console.log("Global PageRank:");
for (const [name, score] of graph.fitPageRank({ damping: 0.85 }, g).top(3)) {
  console.log(`  ${name.padEnd(10)} ${score.toFixed(4)}`);
}

console.log("\nPersonalized toward 'checkout':");
const personalized = graph.fitPageRank(
  { teleport: { tag: "seeds", val: [["checkout", 1.0]] } },
  g,
);
for (const [name, score] of personalized.top(3)) {
  console.log(`  ${name.padEnd(10)} ${score.toFixed(4)}`);
}

const components = functions.extractComponents(g, 1);
console.log(`\nConnected components: ${components.length}`);
