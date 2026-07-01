import { useState } from "react";

function App() {

  const [prediction, setPrediction] = useState("Waiting...");
  const [confidence, setConfidence] = useState(0);
  const [selectedFile, setSelectedFile] = useState(null);

  const handlePrediction = async () => {

    if (!selectedFile) {
      alert("Please select a WAV file.");
      return;
    }
    const formData = new FormData();
    formData.append("file", selectedFile);

    const response = await fetch(
      "https://ai-heart-murmur-detection-1.onrender.com/predict",
      {
        method: "POST",
        body: formData,
      }
    );
    const result = await response.json();

    setPrediction(result.prediction);
    setConfidence(result.confidence);

  };

  return (
    <div>

      <h1>Heart Murmur Detection</h1>

      <h2>{prediction}</h2>

      <p>Confidence : {confidence}%</p>

      <input 
        type = "file" 
        onChange={(event) => setSelectedFile(event.target.files[0])}
      />

      <p>{selectedFile?.name}</p>

      <button onClick = {handlePrediction}>
        Predict
      </button>

      <h2>{prediction}</h2>

      <p>Confidence : {confidence}%</p>

    </div>
  );
}

export default App;
