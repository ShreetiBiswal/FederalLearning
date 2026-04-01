import { deepScale, deepAdd } from '../utils/tensorUtils.js';

export function aggregateWSMHarmonicClassWeighted(updates) {
    console.log("\n========== ⚖️ INITIATING HARMONIC HYBRID AGGREGATION ==========");
    const C = 9; 
    let global_class_counts = new Array(C).fill(0);

    let totalDataSize = updates.reduce((sum, update) => sum + update.size, 0);

    for (let update of updates) {
        if (!update.beta) throw new Error(`[🚨] ERROR: Client ${update.id} missing beta array!`);
        for (let x = 0; x < C; x++) {
            global_class_counts[x] += update.size * update.beta[x];
        }
    }

    let rawWeightFactors = [];
    let sumHarmonicMeans = 0;

    for (let update of updates) {
        let total_score_s = 0;
        for (let x = 0; x < C; x++) {
            let exact_rows = update.size * update.beta[x];
            if (global_class_counts[x] > 0) {
                total_score_s += exact_rows / global_class_counts[x];
            }
        }
        
        let w_fedavg = update.size / totalDataSize;
        let w_class = total_score_s / C; 
        
        let hm = 0;
        if ((w_fedavg + w_class) > 0) {
            hm = (2 * w_fedavg * w_class) / (w_fedavg + w_class);
        }
        
        rawWeightFactors.push({ id: update.id, size: update.size, w_fedavg, w_class, hm });
        sumHarmonicMeans += hm;
    }

    // Normalize so they sum to 1.0
    let weightFactors = [];
    for (let rwf of rawWeightFactors) {
        let normalizedWeight = sumHarmonicMeans > 0 ? (rwf.hm / sumHarmonicMeans) : 0;
        weightFactors.push(normalizedWeight);
        console.log(`   [📊] Hosp ID: ${rwf.id.substring(0,4)} | FedAvg: ${(rwf.w_fedavg*100).toFixed(1)}% | Class: ${(rwf.w_class*100).toFixed(1)}% | 🏆 NORMALIZED HM: ${(normalizedWeight * 100).toFixed(2)}%`);
    }

    let averagedWeights = {};
    let layers = Object.keys(updates[0].weights);

    layers.forEach(layer => {
        let baseLayer = deepScale(updates[0].weights[layer], weightFactors[0]);
        for (let k = 1; k < updates.length; k++) {
            let scaledLayer = deepScale(updates[k].weights[layer], weightFactors[k]);
            baseLayer = deepAdd(baseLayer, scaledLayer);
        }
        averagedWeights[layer] = baseLayer;
    });

    console.log("=================================================================\n");
    return averagedWeights;
}