# Wazemmes Parking from Above (prototype v2)

Group 8 · AI for Business & Society · SKEMA Business School, Lille.

Counts parked cars on aerial photos in the browser and turns the count into free parking spaces:
1. the photo is enlarged to 10 cm per pixel and cut into overlapping 64 m tiles;
2. a YOLO11n model trained on the COWC aerial car dataset marks each car (confidence cut-off 0.10, chosen by 3-fold cross-validation, seed 0);
3. only cars inside the marked parking (OpenStreetMap streets and car parks, or lines you draw) are kept;
4. spaces = kerb length / 5.5 m (parallel), 3.0 m (angled), 2.5 m (side by side); car parks = capacity or area / 25 m²;
5. cars are corrected by precision / recall (0.895 / 0.819, measured on 432 hand-labelled Wazemmes cars) and for tree cover;
6. free spaces = spaces − corrected cars, with a range from the error rates measured on three photo dates.

Everything runs client-side with onnxruntime-web; `models/own-model.wasm` is the ONNX model (renamed for hosting).
Open `index.html` through a web server (GitHub Pages works); opening the file directly from disk blocks the model download.

Data: IGN Géoplateforme orthophotos (Licence Ouverte 2.0); parking © OpenStreetMap contributors (ODbL).
The first prototype (retail car-park counter) is in the folder above.
