import express from 'express';
import { createServer } from 'http';
import { Server } from 'socket.io';
import fs from 'fs';       
import zlib from 'zlib';   

import { aggregateWeightedFedAvg } from './aggregators/fedavg.js';

const app = express();
const server = createServer(app);

const io = new Server(server, {
    maxHttpBufferSize: 5e7, // 50MB
    cors: { origin: "*" }
});

const TARGET_HOSPITALS = 4;
const MAX_ROUNDS = 50;            
let connectedHospitals = [];
let evaluatorId = null;
let receivedUpdates = [];
let currentRound = 1;
let isTrainingActive = false;

// --- Initialize CSV File ---
const CSV_FILE = 'server_avg_metrics.csv';
if (!fs.existsSync(CSV_FILE)) {
    fs.writeFileSync(CSV_FILE, 'Algorithm,Round,Global_Accuracy,Global_Loss\n');
}

io.on('connection', (socket) => {
    console.log(`[+] New connection: ${socket.id}`);

    socket.on('register_evaluator', () => {
        evaluatorId = socket.id;
        console.log(`[⚖️] Evaluator Registered: ${socket.id}`);
    });

    socket.on('join_as_hospital', () => {
        if (!connectedHospitals.includes(socket.id)) {
            connectedHospitals.push(socket.id);
            console.log(`[🏥] Hospital Registered: ${socket.id} (${connectedHospitals.length}/${TARGET_HOSPITALS})`);
        }

        if (connectedHospitals.length === TARGET_HOSPITALS && !isTrainingActive) {
            console.log(`\n🚀 All ${TARGET_HOSPITALS} hospitals ready. Starting Round ${currentRound}...`);
            isTrainingActive = true;
            io.emit('start_training_round', { round: currentRound });
        }
    });

    if (connectedHospitals.length === TARGET_HOSPITALS && !isTrainingActive) {
        console.log(`\n🚀 All ${TARGET_HOSPITALS} hospitals connected. Starting FL Round ${currentRound}...`);
        isTrainingActive = true;
        io.emit('start_training_round', { round: currentRound });
    }

    socket.on('upload_weights', (payload) => {
        if (!connectedHospitals.includes(socket.id)) {
            console.log(`[⚠️] Ignoring weights from unauthorized/evaluator socket: ${socket.id}`);
            return;
        }

        if (!payload || !payload.algo || typeof payload.dataset_size !== 'number') {
            console.error(`[!] ERROR: Malformed payload received from ${socket.id}. Rejecting.`);
            socket.emit('server_error', { message: 'Invalid payload structure.' });
            return;
        }

        let weightsToStore = payload.delta_weights;

        if (payload.is_compressed) {
            console.log(`   [🗜️] Decompressing payload from ${socket.id}...`);
            try {
                const unzippedBuffer = zlib.inflateSync(payload.delta_weights);
                weightsToStore = JSON.parse(unzippedBuffer.toString('utf-8'));
            } catch (error) {
                console.error(`[!] Decompression failed: ${error.message}`);
                socket.emit('server_error', { message: 'Failed to decompress weights.' });
                return;
            }
        }

        // --- NEW: Generate the Dynamic Logging Name ---
        const isSmoteDisabled = payload.smote_disabled === true;
        const logAlgoName = isSmoteDisabled ? `${payload.algo}_nosmote` : payload.algo;

        console.log(`[⬇️] Received ${logAlgoName.toUpperCase()} weights from ${socket.id} (Size: ${payload.dataset_size})`);
        
        receivedUpdates.push({
            id: socket.id,
            algo: payload.algo,        // Base algo for math routing (e.g., 'wsm_ce_fedavg')
            logAlgo: logAlgoName,      // Custom algo for file saving (e.g., 'wsm_ce_fedavg_nosmote')
            weights: weightsToStore,
            size: payload.dataset_size,
            metrics: payload.metrics 
        });

        if (receivedUpdates.length === TARGET_HOSPITALS) {
            console.log(`\n[⚙️] All updates received. Processing...`);
            
            let currentAlgo = receivedUpdates[0].algo;
            let currentLogAlgo = receivedUpdates[0].logAlgo; // Use this for files!
            
            let totalDataSize = receivedUpdates.reduce((sum, update) => sum + update.size, 0);

            // --- Calculate and Log Average Accuracy ---
            let globalAcc = 0;
            let globalLoss = 0;
            
            receivedUpdates.forEach(update => {
                let weightFraction = update.size / totalDataSize;
                if (update.metrics) {
                    globalAcc += (update.metrics.accuracy * weightFraction);
                    globalLoss += (update.metrics.loss * weightFraction);
                }
            });

            console.log(`   📈 Local Average Accuracy: ${(globalAcc * 100).toFixed(2)}% | Loss: ${globalLoss.toFixed(4)}`);
            
            // --- Log to CSV using the dynamic logAlgo ---
            fs.appendFileSync(CSV_FILE, `${currentLogAlgo},${currentRound},${globalAcc},${globalLoss}\n`);

            try {
                let globalModelUpdates;

                // --- The Algorithm Router (Still uses the base 'currentAlgo') ---
                if (currentAlgo === 'fedavg' || currentAlgo === 'ce_fedavg' || currentAlgo === 'wsm_ce_fedavg') {
                    globalModelUpdates = aggregateWeightedFedAvg(receivedUpdates);
                } 
                else if (currentAlgo === 'scaffold') {
                    console.log(`   [⚠️] Routing to SCAFFOLD Aggregator...`);
                    globalModelUpdates = aggregateWeightedFedAvg(receivedUpdates); 
                }

                receivedUpdates = [];
                currentRound++;

                if (currentRound <= MAX_ROUNDS) {
                    console.log(`[⬆️] Broadcasting Round ${currentRound} master weights...`);
                    io.emit('apply_global_update', { global_weights: globalModelUpdates, round: currentRound });
                } else {
                    console.log('\n✅ Federated Learning Training Complete!');
                    
                    // --- NEW: Save the JSON file dynamically ---
                    const FINAL_MODEL_PATH = `./final_master_model_${currentLogAlgo}.json`;
                    
                    fs.writeFileSync(FINAL_MODEL_PATH, JSON.stringify(globalModelUpdates));
                    console.log(`💾 Final Master Model saved to: ${FINAL_MODEL_PATH}`);
                    
                    isTrainingActive = false;
                    io.emit('training_finished');
                }
            } catch (error) {
                console.error(`\n[!!!] CRITICAL MATH ERROR during Aggregation: ${error.message}`);
                receivedUpdates = [];
                io.emit('server_error', { message: 'Aggregation failed. Round aborted.' });
            }
        }
    });

    socket.on('disconnect', () => {
        console.log(`[-] Disconnected: ${socket.id}`);
        connectedHospitals = connectedHospitals.filter(id => id !== socket.id);
        
        if (socket.id === evaluatorId) {
            evaluatorId = null;
        }
        
        if (isTrainingActive) {
            console.log(`[!] WARNING: Hospital dropped during active training. Waiting for reconnection to resume Round ${currentRound}.`);
            receivedUpdates = receivedUpdates.filter(update => update.id !== socket.id);
        }
    });
});

const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
    console.log( `FedAvg Aggregator running on http://localhost:${PORT}`);
    console.log(`Waiting for ${TARGET_HOSPITALS} skewed hospital nodes to connect...`);
});