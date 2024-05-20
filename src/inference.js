const tfjs = require('@tensorflow/tfjs-node');

function loadModel(){
    const modelUrl = "file://models/model.json";
    return tfjs.loadLayersModel(modelUrl);
}

function predict(model, imageBuffer) {
    const tensor = tfjs.node
    //melakukan decoding dari gambar dengan format JPEG yang disimpan dalam buffer
    .decodeJpeg(imageBuffer)
    //mengubah gambar yang sebelumnya di-decode menjadi 150 x 150 piksel dengan algoritma nearest neighbor.
    .resizeNearestNeighbor([150, 150])
    //menambahkan dimensi ekstra pd tensor serta mengonversi nilai" dlm tensor ke tipe data float
    .expandDims()
    .toFloat();

    // mengembalikan hasil prediksi dari tensor yang telah di-preprocessing.
    return model.predict(tensor).data();
}

module.exports = { loadModel, predict};