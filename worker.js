// Lars Oliver Niehsen, Masterarbeit WebAR, Matrikelnummer: 10031818
// =====================================================================
// WEB WORKER für Deep-Learning in eigenem Thread
// =====================================================================

// 1. Lade TensorFlow.js Bibliothek für Deep-Learning-Modell
importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs');

// Deep-Learning-Modell
let model;

// 2. Diese Funktion lädt das trainierte YOLO-Modell
async function loadModel() {
    // ACHTUNG: Speicherort model.json prüfen! Lade .bin Dateien mit TensorFlow.
    model = await tf.loadGraphModel('./model/DL_Veritas_Statue/model.json'); //('./model/DL_Veritas_Statue/model.json')
    console.log("✅ YOLO11n Deep-Learning-Modell erfolgreich geladen!");
    
    // erfolgreiches Laden, dann Übergabe an Haupt-Thread
    postMessage({ type: 'ready' }); 
}

// Ladevorgang des Modells starten
loadModel();

// 3. Übergabe Videoframe von Haupt-Thread
self.onmessage = async function(event) {
    // Abbruch, wenn das Modell noch nicht geladen ist oder der Befehl nicht 'predict'
    if (!model || event.data.type !== 'predict') return;

    try {
        // tf.tidy(): räumt Arbeitsspeicher und GPU auf, nachdem das Bild berechnet wurde.
        const boxes = await tf.tidy(() => {
            
            // Das Pixel-Bild aus der Kamera wird in einen "Tensor" (ein mathematisches Gitter) umgewandelt
            let img = tf.browser.fromPixels(event.data.image);
            
            // Anpassung Bildformat 640x640
            img = tf.image.resizeBilinear(img,[640, 640]);
            
            // Die Farbwerte (0-255) werden in Kommazahlen (0.0 bis 1.0) umgerechnet
            img = img.expandDims(0).div(255.0);

            // Prädiktion DL
            const predictions = model.execute(img);
            const transposed = predictions.transpose([0, 2, 1]).squeeze(); 
            
            // DL-Erkennung Zuverlässigkeit (0.6 = 60%)
            const confidenceThreshold = 0.6; 
            
            // Ausgabe formatieren: Zuverlässigkeit (%) & Koordinaten (x,y)
            const confidences = transposed.slice([0, 4], [-1, 1]).squeeze();
            
            const mask = confidences.greater(confidenceThreshold);
            
            // Leere Liste, wenn keine Erkennung
            if (mask.sum().dataSync()[0] === 0) return[];

            // genauen Koordinaten der besten Detektion
            const filteredBoxes = transposed.slice([0, 0], [-1, 4]);
            
            // Nachricht an Haupt-THread mit Koordinaten und die %-Werte
            return {
                boxes: filteredBoxes.arraySync(),
                scores: confidences.arraySync()
            };
        });

        postMessage({ type: 'result', data: boxes });
        
    } catch (err) {
        // Fehlerausgabe
        console.error("Fehler bei der DL-Erkennung:", err);
    }
};

// Dieser Code wurde mit Gemini 3 Flash für eine performante Ausführung stellenweise optimiert (letzter Zugriff: 29.05.2026)