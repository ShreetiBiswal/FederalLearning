import { deepScale, deepAdd } from "../utils/tensorUtils.js";
import fs from "fs";
import { fileURLToPath } from "url";
import path from "path";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const configPath = path.join(__dirname, "../../shared/config.json");
const config = JSON.parse(fs.readFileSync(configPath, "utf-8"));

export function aggregateErrorAwareHarmonic(updates) {
  console.log(
    "\n========== ⚖️ INITIATING STABILIZED ERROR-AWARE AGGREGATION ==========",
  );
  const C = config.NUM_CLASSES;
  let global_class_counts = new Array(C).fill(0);

  // 1. Calculate absolute global class counts
  for (let update of updates) {
    if (!update.beta)
      throw new Error(`[🚨] ERROR: Client ${update.id} missing beta array!`);
    for (let x = 0; x < C; x++) {
      global_class_counts[x] += update.size * update.beta[x];
    }
  }

  let active_global_classes =
    global_class_counts.filter((count) => count > 0).length || C;

  let rawScores = [];
  let sum_fedavg = 0,
    sum_class = 0;

  // 2. Extract Raw Scores for Volume and Quality (No Error yet)
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

    // Raw error signal (0.0 to 1.0). No need for Math.max(..., 0.01) hack anymore!
    let error_signal = Math.max(1.0 - update.metrics.accuracy, 0.0);

    sum_fedavg += w_fedavg;
    sum_class += w_class;

    rawScores.push({
      id: update.id,
      size: update.size,
      w_fedavg,
      w_class,
      error_signal,
      weights: update.weights,
    });
  }

  let sumFinalScores = 0;

  // 3. Calculate 2-Way HM and apply Error as a Multiplier
  for (let s of rawScores) {
    s.n_fedavg = sum_fedavg > 0 ? s.w_fedavg / sum_fedavg : 0;
    s.n_class = sum_class > 0 ? s.w_class / sum_class : 0;

    // A. Structural Data Quality (2-Way Harmonic Mean: Size & Class Diversity)
    if (s.n_fedavg > 0 && s.n_class > 0) {
      s.data_quality = 2 / (1 / s.n_fedavg + 1 / s.n_class);
    } else {
      s.data_quality = 0;
    }

    // B. Error Modulation (The Stabilizer)
    // Treat structural quality as the baseline (1.0x).
    // Add a proportional "attention boost" (up to +1.0x) based on how much the client is struggling.
    let attention_multiplier = 1.0 + s.error_signal;

    s.final_score = s.data_quality * attention_multiplier;
    sumFinalScores += s.final_score;
  }

  // 4. Final Normalization and Layer-wise Tensor Math
  let averagedWeights = {};
  let layers = Object.keys(updates[0].weights);

  for (let i = 0; i < rawScores.length; i++) {
    let s = rawScores[i];
    let finalWeight = sumFinalScores > 0 ? s.final_score / sumFinalScores : 0;

    console.log(
      `   [📊] Hosp: ${s.id.substring(0, 4)} | Vol: ${(s.n_fedavg * 100).toFixed(1)}% | Class: ${(s.n_class * 100).toFixed(1)}% | Err: ${(s.error_signal * 100).toFixed(1)}% | 🏆 FINAL WSM: ${(finalWeight * 100).toFixed(2)}%`,
    );

    if (i === 0) {
      averagedWeights = layers.reduce((acc, layer) => {
        acc[layer] = deepScale(s.weights[layer], finalWeight);
        return acc;
      }, {});
    } else {
      layers.forEach((layer) => {
        let scaledLayer = deepScale(s.weights[layer], finalWeight);
        averagedWeights[layer] = deepAdd(averagedWeights[layer], scaledLayer);
      });
    }
  }

  console.log(
    "=================================================================\n",
  );
  return averagedWeights;
}
