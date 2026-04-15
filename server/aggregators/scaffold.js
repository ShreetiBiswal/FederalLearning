import { deepScale, deepAdd } from "../utils/tensorUtils.js";

export function aggregateSCAFFOLD(updates) {
  console.log("\n========== 🛡️ INITIATING SCAFFOLD AGGREGATION ==========");

  let totalDataSize = updates.reduce((sum, update) => sum + update.size, 0);

  // 1. Calculate weight factors (Standard FedAvg ratio for the actual neural net weights)
  let weightFactors = updates.map((update) => update.size / totalDataSize);

  // 2. Aggregate the Neural Network Weights (x)
  let averagedWeights = {};
  let layers = Object.keys(updates[0].weights);

  layers.forEach((layer) => {
    let baseLayer = deepScale(updates[0].weights[layer], weightFactors[0]);
    for (let k = 1; k < updates.length; k++) {
      let scaledLayer = deepScale(updates[k].weights[layer], weightFactors[k]);
      baseLayer = deepAdd(baseLayer, scaledLayer);
    }
    averagedWeights[layer] = baseLayer;
  });

  // 3. Aggregate the Control Variates (c)
  // The paper specifies simple averaging for the control variates: c = (1/N) * sum(c_i)
  let averagedControlVariates = {};

  // Check if the clients actually sent control variates (they might not in Round 1)
  if (updates[0].local_c) {
    console.log("   [📊] Aggregating Client Control Variates...");
    let c_layers = Object.keys(updates[0].local_c);

    // 🔥 CRITICAL FIX: The global control variate tracks the gradient of the objective.
    // Because models use data-proportional weights (weightFactors), the control variates
    // MUST use the exact same distribution to point to the correct global minimum.
    // let c_weight = 1.0 / updates.length; // (Removed flawed arithmetic mean)

    c_layers.forEach((layer) => {
      let baseC = deepScale(updates[0].local_c[layer], weightFactors[0]);
      for (let k = 1; k < updates.length; k++) {
        let scaledC = deepScale(updates[k].local_c[layer], weightFactors[k]);
        baseC = deepAdd(baseC, scaledC);
      }
      averagedControlVariates[layer] = baseC;
    });
  } else {
    console.log("   [ℹ️] No control variates received yet (likely Round 1).");
    averagedControlVariates = null;
  }

  console.log("=========================================================\n");

  // Return BOTH the new global weights and the new global control variate
  return {
    globalWeights: averagedWeights,
    globalControlVariates: averagedControlVariates,
  };
}
