// =====================================================================
// WEB WORKER: Der unsichtbare Helfer für die Objekterkennung im Hintergrund
// =====================================================================

// 1. Wir laden die TensorFlow.js Bibliothek aus dem Internet in diesen Helfer herunter.
// Das ist das "Gehirn", das die KI-Modelle verstehen kann.
importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs');

// Platzhalter für unser trainiertes KI-Modell
let model;

// 2. Diese Funktion lädt dein spezielles YOLO-Modell von der Festplatte/vom Server
async function loadModel() {
    // ACHTUNG: Hier muss der genaue Pfad zu deiner model.json stehen!
    // TensorFlow sucht dann automatisch nach den zugehörigen .bin Dateien.
    model = await tf.loadGraphModel('./model/model.json');
    console.log("✅ YOLO11 KI-Modell im Hintergrund-Helfer erfolgreich geladen!");
    
    // Wir rufen dem Haupt-Programm (HTML) zu: "Ich bin bereit!"
    postMessage({ type: 'ready' }); 
}

// Startet den Ladevorgang sofort, wenn diese Datei vom Browser gelesen wird
loadModel();

// 3. Wir warten darauf, dass das Haupt-Programm uns ein Videobild (Frame) schickt
self.onmessage = async function(event) {
    // Wenn das Modell noch nicht geladen ist oder der Befehl nicht 'predict' (vorhersagen) heißt, brechen wir ab
    if (!model || event.data.type !== 'predict') return;

    try {
        // tf.tidy() ist extrem wichtig! Es räumt den Arbeitsspeicher der Grafikkarte
        // sofort wieder auf, nachdem das Bild berechnet wurde. Ohne dies würde das Handy abstürzen.
        const boxes = await tf.tidy(() => {
            
            // Das Pixel-Bild aus der Kamera wird in einen "Tensor" (ein mathematisches Gitter) umgewandelt
            let img = tf.browser.fromPixels(event.data.image);
            
            // Die KI wurde mit Bildern der Größe 640x640 trainiert. Wir müssen das Kamera-Bild anpassen.
            img = tf.image.resizeBilinear(img,[640, 640]);
            
            // Die Farbwerte (0-255) werden in Kommazahlen (0.0 bis 1.0) umgerechnet, da die KI das besser versteht
            img = img.expandDims(0).div(255.0);

            // JETZT RECHNET DIE KI! Wir werfen das Bild in das Modell und bekommen ein Ergebnis.
            const predictions = model.execute(img);
            
            // Das Ergebnis ist ein riesiger, komplizierter Daten-Block. 
            // Wir "drehen" und "quetschen" ihn, damit wir ihn leichter lesen können.
            const transposed = predictions.transpose([0, 2, 1]).squeeze(); 
            
            // Wie sicher muss sich die KI sein, damit wir eine Box zeichnen? (0.6 = 60%)
            const confidenceThreshold = 0.6; 
            
            // Wir trennen die Sicherheits-Werte (%) von den Koordinaten (x,y)
            const confidences = transposed.slice([0, 4], [-1, 1]).squeeze();
            
            // Wir erstellen eine Maske, die alle unsicheren Ergebnisse (unter 60%) wegwirft
            const mask = confidences.greater(confidenceThreshold);
            
            // Wenn nach dem Filtern keine einzige Box übrig geblieben ist, geben wir eine leere Liste zurück
            if (mask.sum().dataSync()[0] === 0) return[];

            // Ansonsten holen wir uns die genauen Koordinaten der besten Treffer
            const filteredBoxes = transposed.slice([0, 0], [-1, 4]);
            
            // Wir schicken die fertigen Koordinaten und die %-Werte zurück ans Haupt-Programm
            return {
                boxes: filteredBoxes.arraySync(),
                scores: confidences.arraySync()
            };
        });

        // Die fertige Nachricht an die index.html senden
        postMessage({ type: 'result', data: boxes });
        
    } catch (err) {
        // Falls etwas schiefgeht, wird hier der Fehler gemeldet
        console.error("Fehler bei der KI-Erkennung:", err);
    }
};