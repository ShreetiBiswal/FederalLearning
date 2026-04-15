import { deepScale, deepAdd } from "../utils/tensorUtils.js";

export function aggregateWeightedFedAvg(updates) {
  let averagedWeights = {};
  let layers = Object.keys(updates[0].weights);
  let totalDataSize = updates.reduce((sum, update) => sum + update.size, 0);

  if (totalDataSize === 0)
    throw new Error("Total dataset size is 0. Cannot divide by zero.");

  layers.forEach((layer) => {
    let fraction0 = updates[0].size / totalDataSize;
    let baseLayer = deepScale(updates[0].weights[layer], fraction0);

    for (let k = 1; k < updates.length; k++) {
      let fractionK = updates[k].size / totalDataSize;
      let scaledLayer = deepScale(updates[k].weights[layer], fractionK);
      baseLayer = deepAdd(baseLayer, scaledLayer);
    }
    averagedWeights[layer] = baseLayer;
  });

  return averagedWeights;
}
