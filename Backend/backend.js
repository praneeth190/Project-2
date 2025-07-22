const express = require('express');
const multer = require('multer');
const cors = require('cors');
const bodyParser = require('body-parser');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const app = express();
const PORT = 5000;
const uploadDir = path.join(__dirname, 'uploads');

// Create uploads directory if it doesn't exist
if (!fs.existsSync(uploadDir)){
    fs.mkdirSync(uploadDir);
}

// Middleware
app.use(cors());
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

// Configure Multer
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    cb(null, `${Date.now()}-${file.originalname}`);
  }
});

const upload = multer({ 
  storage,
  limits: {
    fileSize: 10 * 1024 * 1024 // 10MB limit
  }
});

app.get('/', (req, res) => {
  res.send('Speech Emotion Recognizer Backend');
});

app.post('/predict', upload.single('audio'), async (req, res) => {
  if (!req.file) {
    console.error("Error: No file was uploaded")
    return res.status(400).send({ error: 'No audio file provided' });
  }

  try {
    const audioPath = req.file.path;
    const pythonScriptPath = path.join(__dirname, '.', 'predict.py'); // Corrected path

    console.log("Audio path: ", audioPath);
    console.log("Python Script Path: ", pythonScriptPath);

    const pythonProcess = spawn('python', [pythonScriptPath, audioPath]);

    let result = '';
    let error = '';
    pythonProcess.stdout.on('data', (data) => {
      console.log("Python stdout: ", data.toString());
      result += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
      error += data.toString();
      console.error("Python stderr: ", data.toString());
    });

    pythonProcess.on('close', (code) => {
        fs.unlink(audioPath, (err) => {
            if (err) console.error('Error deleting file:', err);
          });
      console.log("Python script exited with code: ", code);
      if (code === 0) {
        try {
          const prediction = JSON.parse(result);
          res.send({ emotion: prediction.emotion });
        } catch (err) {
          console.error("Error parsing JSON: ", err);
          res.status(500).send({ error: 'Error parsing prediction result' });
        }
      } else {
        console.error("Error processing prediction: ", error);
        res.status(500).send({ error: `Error processing prediction: ${error}` });
      }
    });
  } catch (err) {
    console.error('Error handling prediction:', err);
    res.status(500).send({ error: 'Error processing file' });
  }
});

app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});