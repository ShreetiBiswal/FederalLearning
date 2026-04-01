import { deepScale, deepAdd } from '../utils/tensorUtils.js';

export function aggregateWSMClassWeighted(updates) {
    console.log("\n========== ⚖️ INITIATING CLASS-WEIGHTED AGGREGATION ==========");
    const C = 9; // Total medical classes
    let global_class_counts = new Array(C).fill(0);

    // 1. Calculate the Global Denominator (Total world data for each disease)
    for (let update of updates) {
        if (!update.beta) {
            throw new Error(`[🚨] ERROR: Client ${update.id} did not send a beta array!`);
        }
        for (let x = 0; x < C; x++) {
            let exact_rows = update.size * update.beta[x];
            global_class_counts[x] += exact_rows;
        }
    }

    // 2. Calculate the Normalized Class Score (Voting Power) for each hospital
    let weightFactors = [];
    for (let update of updates) {
        let total_score_s = 0;
        
        for (let x = 0; x < C; x++) {
            let exact_rows = update.size * update.beta[x];
            
            // Cx = (ni * pix) / Global Sum
            if (global_class_counts[x] > 0) {
                let Cx = exact_rows / global_class_counts[x];
                total_score_s += Cx;
            }
        }
        
        // Final Normalized Aggregation Weight
        let weightFactor = total_score_s / C; 
        weightFactors.push(weightFactor);
        
        console.log(`   [📊] Hosp ID: ${update.id} | Raw Data Size: ${update.size} | Voting Power: ${(weightFactor * 100).toFixed(2)}%`);
    }

    // 3. Aggregate Weights using your existing Tensor Utilities
    let averagedWeights = {};
    let layers = Object.keys(updates[0].weights);

    layers.forEach(layer => {
        // Start by scaling the very first client's layer
        let baseLayer = deepScale(updates[0].weights[layer], weightFactors[0]);

        // Loop through the remaining clients and add their scaled layers
        for (let k = 1; k < updates.length; k++) {
            let scaledLayer = deepScale(updates[k].weights[layer], weightFactors[k]);
            baseLayer = deepAdd(baseLayer, scaledLayer);
        }
        averagedWeights[layer] = baseLayer;
    });

    console.log("=================================================================\n");
    return averagedWeights;
}