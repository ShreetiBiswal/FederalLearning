import express from 'express';
import { createServer } from 'http';
import { Server } from 'socket.io';
import fs from 'fs';       
import zlib from 'zlib';   

// --- NEW: Import our segregated math module ---
import { aggregateWeightedFedAvg } from './aggregators/fedavg.js';

const app = express();
const server = createServer(app);

const io = new Server(server, {
    maxHttpBufferSize: 5e7, // 50MB
    cors: { origin: "*" }
});

const TARGET_HOSPITALS = 4;
const MAX_ROUNDS = 50;            
let connectedClients = [];
let receivedUpdates = [];
let currentRound = 1;
let isTrainingActive = false;

// --- Initialize CSV File ---
const CSV_FILE = 'server_avg_metrics.csv';
if (!fs.existsSync(CSV_FILE)) {
    fs.writeFileSync(CSV_FILE, 'Algorithm,Round,Global_Accuracy,Global_Loss\n');
}

io.on('connection', (socket) => {
    console.log(`[+] Hospital Connected: ${socket.id}`);
    connectedClients.push(socket.id);

    if (connectedClients.length === TARGET_HOSPITALS && !isTrainingActive) {
        console.log(`\n🚀 All ${TARGET_HOSPITALS} hospitals connected. Starting FL Round ${currentRound}...`);
        isTrainingActive = true;
        io.emit('start_training_round', { round: currentRound });
    }

    socket.on('upload_weights', (payload) => {
        // Validate payload structure
        if (!payload || !payload.algo || typeof payload.dataset_size !== 'number') {
            console.error(`[!] ERROR: Malformed payload received from ${socket.id}. Rejecting.`);
            socket.emit('server_error', { message: 'Invalid payload structure.' });
            return;
        }

        let weightsToStore = payload.delta_weights;

        // --- NEW: Dynamic Decompression for CE-FedAvg ---
        if (payload.is_compressed) {
            console.log(`   [🗜️] Decompressing CE-FedAvg payload from ${socket.id}...`);
            try {
                // Socket.io sends binary data as a Buffer natively
                const unzippedBuffer = zlib.inflateSync(payload.delta_weights);
                weightsToStore = JSON.parse(unzippedBuffer.toString('utf-8'));
            } catch (error) {
                console.error(`[!] Decompression failed: ${error.message}`);
                socket.emit('server_error', { message: 'Failed to decompress weights.' });
                return;
            }
        }

        console.log(`[⬇️] Received ${payload.algo.toUpperCase()} weights from ${socket.id} (Size: ${payload.dataset_size})`);
        
        receivedUpdates.push({
            id: socket.id,
            algo: payload.algo,
            weights: weightsToStore,
            size: payload.dataset_size,
            metrics: payload.metrics // Storing the local accuracy!
        });

        if (receivedUpdates.length === TARGET_HOSPITALS) {
            console.log(`\n[⚙️] All updates received. Processing...`);
            
            let currentAlgo = receivedUpdates[0].algo;
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
            fs.appendFileSync(CSV_FILE, `${currentAlgo},${currentRound},${globalAcc},${globalLoss}\n`);

            try {
                let globalModelUpdates;

                // --- The Algorithm Router ---
                if (currentAlgo === 'fedavg' || currentAlgo === 'ce_fedavg') {
                    // Both use the exact same standard weighted averaging on the server side
                    globalModelUpdates = aggregateWeightedFedAvg(receivedUpdates);
                } 
                else if (currentAlgo === 'scaffold') {
                    console.log(`   [⚠️] Routing to SCAFFOLD Aggregator...`);
                    // globalModelUpdates = aggregateScaffold(receivedUpdates); // We will build this later!
                    globalModelUpdates = aggregateWeightedFedAvg(receivedUpdates); // Fallback for now
                }

                receivedUpdates = [];
                currentRound++;

                if (currentRound <= MAX_ROUNDS) {
                    console.log(`[⬆️] Broadcasting Round ${currentRound} master weights...`);
                    io.emit('apply_global_update', { global_weights: globalModelUpdates, round: currentRound });
                } else {
                    console.log('\n✅ Federated Learning Training Complete!');
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
        console.log(`[-] Hospital Disconnected: ${socket.id}`);
        connectedClients = connectedClients.filter(id => id !== socket.id);
        
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