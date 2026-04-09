import { deepScale, deepAdd } from '../utils/tensorUtils.js';

export function aggregateErrorAwareHarmonic(updates) {
    console.log("\n========== ⚖️ INITIATING ERROR-AWARE HARMONIC AGGREGATION ==========");
    const C = 9; 
    let global_class_counts = new Array(C).fill(0);

    // 1. Calculate absolute global class counts
    for (let update of updates) {
        if (!update.beta) throw new Error(`[🚨] ERROR: Client ${update.id} missing beta array!`);
        for (let x = 0; x < C; x++) {
            global_class_counts[x] += update.size * update.beta[x];
        }
    }

    // Dynamic class count to prevent unfair deflation if a class is entirely missing
    let active_global_classes = global_class_counts.filter(count => count > 0).length || C; 

    let rawScores = [];
    let sum_fedavg = 0, sum_class = 0, sum_error = 0;

    // 2. Extract Raw Scores for Volume, Quality, and Error Signal
    for (let update of updates) {
        let total_score_s = 0;
        for (let x = 0; x < C; x++) {
            let exact_rows = update.size * update.beta[x];
            if (global_class_counts[x] > 0) {
                total_score_s += exact_rows / global_class_counts[x];
            }
        }
        
        let w_fedavg = update.size;
        let w_class = total_score_s / active_global_classes;
        
        // 🚨 NEW: The Error Signal (Inverted Accuracy). 
        // We use Math.max(..., 0.01) so a perfect 100% accuracy doesn't cause a divide-by-zero crash.
        let error_signal = Math.max(1.0 - update.metrics.accuracy, 0.01); 

        sum_fedavg += w_fedavg;
        sum_class += w_class;
        sum_error += error_signal;

        rawScores.push({ 
            id: update.id, 
            size: update.size, 
            w_fedavg, 
            w_class, 
            error_signal, 
            weights: update.weights 
        });
    }

    let sumHarmonicMeans = 0;

    // 3. Normalize all 3 pillars and calculate the 3-Way Harmonic Mean
    for (let s of rawScores) {
        s.n_fedavg = sum_fedavg > 0 ? (s.w_fedavg / sum_fedavg) : 0;
        s.n_class = sum_class > 0 ? (s.w_class / sum_class) : 0;
        s.n_error = sum_error > 0 ? (s.error_signal / sum_error) : 0;
        
        if (s.n_fedavg > 0 && s.n_class > 0 && s.n_error > 0) {
            // Formula for 3-Way Harmonic Mean: 3 / ((1/a) + (1/b) + (1/c))
            s.hm = 3 / ((1 / s.n_fedavg) + (1 / s.n_class) + (1 / s.n_error));
        } else {
            s.hm = 0;
        }
        
        sumHarmonicMeans += s.hm;
    }

    // 4. Final Normalization and Layer-wise Tensor Math
    let averagedWeights = {};
    let layers = Object.keys(updates[0].weights);

    for (let i = 0; i < rawScores.length; i++) {
        let s = rawScores[i];
        let finalWeight = sumHarmonicMeans > 0 ? (s.hm / sumHarmonicMeans) : 0;
        
        console.log(`   [📊] Hosp: ${s.id.substring(0,4)} | Vol: ${(s.n_fedavg*100).toFixed(1)}% | Class: ${(s.n_class*100).toFixed(1)}% | Error: ${(s.n_error*100).toFixed(1)}% | 🏆 FINAL 3-WAY HM: ${(finalWeight * 100).toFixed(2)}%`);

        if (i === 0) {
            averagedWeights = layers.reduce((acc, layer) => {
                acc[layer] = deepScale(s.weights[layer], finalWeight);
                return acc;
            }, {});
        } else {
            layers.forEach(layer => {
                let scaledLayer = deepScale(s.weights[layer], finalWeight);
                averagedWeights[layer] = deepAdd(averagedWeights[layer], scaledLayer);
            });
        }
    }

    console.log("=================================================================\n");
    return averagedWeights;
}