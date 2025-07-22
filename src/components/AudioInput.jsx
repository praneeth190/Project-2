import { useState, useRef, useEffect } from 'react';
import { X } from 'lucide-react';

const AudioInput = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [recordedAudio, setRecordedAudio] = useState(null);
  const [audioPreviewUrl, setAudioPreviewUrl] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [predictedEmotion, setPredictedEmotion] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');
  const [audioDuration, setAudioDuration] = useState(0);

  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const timerRef = useRef(null);
  const audioContextRef = useRef(null);
  const sectionRef = useRef(null);

  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
      if (mediaRecorderRef.current && isRecording) {
        mediaRecorderRef.current.stop();
      }
    };
  }, [isRecording]);

  const validateAudioDuration = (fileOrBlob) => {
    return new Promise((resolve, reject) => {
      const audio = new Audio();
      if (fileOrBlob instanceof File) {
        audio.src = URL.createObjectURL(fileOrBlob);
      } else if (fileOrBlob instanceof Blob) {
        audio.src = URL.createObjectURL(fileOrBlob);
      }

      audio.addEventListener('loadedmetadata', () => {
        const duration = audio.duration;
        URL.revokeObjectURL(audio.src);

        if (duration < 3) {
          reject('Audio must be at least 4 seconds long');
        } else if (duration > 13) {
          reject('Audio must not exceed 10 seconds');
        } else {
          resolve(duration);
        }
      });

      audio.addEventListener('error', () => {
        URL.revokeObjectURL(audio.src);
        reject('Error loading audio file');
      });
    });
  };

  const convertToWav = async (audioBlob) => {
    try {
      if (!audioContextRef.current) {
        audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
      }

      const arrayBuffer = await audioBlob.arrayBuffer();
      const audioBuffer = await audioContextRef.current.decodeAudioData(arrayBuffer);

      const offlineContext = new OfflineAudioContext(
        audioBuffer.numberOfChannels,
        audioBuffer.length,
        audioBuffer.sampleRate
      );

      const source = offlineContext.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(offlineContext.destination);
      source.start();

      const renderedBuffer = await offlineContext.startRendering();
      const wavBlob = await convertToWavBlob(renderedBuffer);
      return new File([wavBlob], 'recorded-audio.wav', { type: 'audio/wav' });
    } catch (error) {
      console.error('Error converting audio:', error);
      throw new Error('Failed to convert audio format');
    }
  };

  const convertToWavBlob = (audioBuffer) => {
    const numOfChan = audioBuffer.numberOfChannels;
    const length = audioBuffer.length * numOfChan * 2;
    const buffer = new ArrayBuffer(44 + length);
    const view = new DataView(buffer);

    writeUTFBytes(view, 0, 'RIFF');
    view.setUint32(4, 36 + length, true);
    writeUTFBytes(view, 8, 'WAVE');
    writeUTFBytes(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, numOfChan, true);
    view.setUint32(24, audioBuffer.sampleRate, true);
    view.setUint32(28, audioBuffer.sampleRate * 2 * numOfChan, true);
    view.setUint16(32, numOfChan * 2, true);
    view.setUint16(34, 16, true);
    writeUTFBytes(view, 36, 'data');
    view.setUint32(40, length, true);

    const channelData = [];
    let offset = 44;
    for (let i = 0; i < audioBuffer.numberOfChannels; i++) {
      channelData[i] = audioBuffer.getChannelData(i);
    }

    while (offset < buffer.byteLength) {
      for (let i = 0; i < numOfChan; i++) {
        const sample = Math.max(-1, Math.min(1, channelData[i][(offset - 44) / 2]));
        view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
        offset += 2;
      }
    }

    return new Blob([buffer], { type: 'audio/wav' });
  };

  const writeUTFBytes = (view, offset, string) => {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  };

  const clearOutputs = () => {
    setPredictedEmotion(null);
    setErrorMessage('');
    setAudioDuration(0);
  };

  const startRecording = async () => {
    clearOutputs();
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);

      mediaRecorderRef.current.ondataavailable = (e) => {
        chunksRef.current.push(e.data);
      };

      mediaRecorderRef.current.onstop = async () => {
        try {
          const audioBlob = new Blob(chunksRef.current, { type: 'audio/webm' });
          const wavFile = await convertToWav(audioBlob);
          
          try {
            const duration = await validateAudioDuration(wavFile);
            setAudioDuration(duration);
            setRecordedAudio(wavFile);
            // Clear any existing preview URL
            if (audioPreviewUrl) {
              URL.revokeObjectURL(audioPreviewUrl);
            }
            setAudioPreviewUrl(URL.createObjectURL(wavFile));
            setSelectedFile(null);
            chunksRef.current = [];
            setRecordingTime(0);
            if (timerRef.current) clearInterval(timerRef.current);
          } catch (error) {
            setErrorMessage(error);
            chunksRef.current = [];
            return;
          }
        } catch (error) {
          setErrorMessage('Error processing recorded audio');
          console.error(error);
        }
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);

      timerRef.current = setInterval(() => {
        setRecordingTime(prev => {
          if (prev >= 10) {
            stopRecording();
            return prev;
          }
          return prev + 1;
        });
      }, 1000);

    } catch (err) {
      console.error("Error accessing microphone:", err);
      setErrorMessage('Error accessing microphone. Please ensure microphone permissions are granted.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      const tracks = mediaRecorderRef.current.stream.getTracks();
      tracks.forEach(track => track.stop());
    }
  };

  const handleFileChange = async (e) => {
    const file = e.target.files[0];
    clearOutputs();
    
    if (file) {
      if (file.size > 10 * 1024 * 1024) {
        setErrorMessage('File size must be less than 10MB');
        e.target.value = '';
        return;
      }

      const validTypes = ['audio/wav', 'audio/mpeg', 'audio/mp3', 'audio/webm'];
      if (!validTypes.includes(file.type)) {
        setErrorMessage('Please upload a WAV, MP3, or WebM file');
        e.target.value = '';
        return;
      }

      try {
        const duration = await validateAudioDuration(file);
        
        // If there's an existing preview URL, revoke it
        if (audioPreviewUrl) {
          URL.revokeObjectURL(audioPreviewUrl);
        }

        // Update the states
        setAudioDuration(duration);
        setSelectedFile(file);
        setAudioPreviewUrl(URL.createObjectURL(file));
        setRecordedAudio(null); // Clear any recorded audio
        
      } catch (error) {
        setErrorMessage(error);
        e.target.value = '';
      }
    }
  };

  const handleRemoveFile = () => {
    setSelectedFile(null);
    setRecordedAudio(null);
    setAudioDuration(0);
    if (audioPreviewUrl) {
      URL.revokeObjectURL(audioPreviewUrl);
      setAudioPreviewUrl(null);
    }
    clearOutputs();
    const fileInput = document.getElementById('audio-file');
    if (fileInput) {
      fileInput.value = '';
    }
  };

  const handlePrediction = async () => {
    const audioFile = selectedFile || recordedAudio;
    if (!audioFile) return;
    
    setIsLoading(true);
    setErrorMessage('');

    try {
      const formData = new FormData();
      if (recordedAudio) {
        formData.append('audio', audioFile, 'recorded-audio.wav');
      } else {
        formData.append('audio', audioFile);
      }

      const result = await fetch('https://8a86-2409-40f0-102e-249d-e9e7-e8ff-6739-3020.ngrok-free.app/predict', {
        method: 'POST',
        body: formData
      });

      const data = await result.json();
      if (data.error) {
        setErrorMessage(data.error);
        setPredictedEmotion(null);
      } else {
        setPredictedEmotion(data.emotion);
        sectionRef.current?.scrollIntoView({ behavior: 'smooth' });
      }
    } catch (error) {
      console.error('Error predicting emotion:', error);
      setErrorMessage('Error predicting emotion');
    } finally {
      setIsLoading(false);
    }
  };

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const getEmoji = (emotion) => {
    const emojiMap = {
      'happy': '😄',
      'neutral': '🙂',
      'sad': '😢',
      'disgust': '🤢',
      'angry': '😠',
      'fear': '😨'
    };
    return emojiMap[emotion?.toLowerCase()] || '❓';
  };

  return (
    <section className="audio-section" ref={sectionRef}>
      <div className="container">
        <div className="audio-card">
          {errorMessage && <div className="error-message">{errorMessage}</div>}

          <div className="input-group">
            <label className="input-label">Upload Audio File</label>
            <div className="file-upload">
              <input
                type="file"
                accept="audio/wav,audio/mp3,audio/mpeg"
                onChange={handleFileChange}
                style={{ display: 'none' }}
                id="audio-file"
              />
              <label 
                htmlFor="audio-file" 
                className="file-upload-label"
              >
                <div className="upload-icon">📁</div>
                <div className="upload-text">
                  {selectedFile ? (
                    <div className="file-info">
                      <span className="file-name">{selectedFile.name}</span>
                      <span className="file-duration">({audioDuration.toFixed(1)}s)</span>
                      <button
                        onClick={(e) => {
                          e.preventDefault();
                          handleRemoveFile();
                        }}
                        className="remove-file"
                        aria-label="Remove file"
                      >
                        <X className="w-4 h-4" />
                      </button>
                    </div>
                  ) : (
                    <>
                      Click to upload or drag and drop
                      <span className="file-hint"> WAV, MP3 (3-10 seconds) up to 10MB</span>
                    </>
                  )}
                </div>
              </label>
            </div>
          </div>

          <div className="input-group">
            <label className="input-label">Record Audio</label>
            <div className="record-container">
              <button
                onClick={isRecording ? stopRecording : startRecording}
                className={`record-button ${isRecording ? 'recording' : ''}`}
              >
                {isRecording ? '◉ Stop Recording' : '⬤ Start Recording'}
              </button>
              {isRecording && (
                <div className="recording-time">
                  Recording: {formatTime(recordingTime)}
                  {recordingTime < 4 && <span className="recording-hint"> (Minimum 4 seconds required)</span>}
                </div>
              )}
            </div>
          </div>

          {audioPreviewUrl && (
            <div className="input-group">
              <label className="input-label">Preview Audio</label>
              <audio controls className="audio-preview">
                <source src={audioPreviewUrl} type="audio/wav" />
                Your browser does not support the audio element.
              </audio>
            </div>
          )}

          <button
            onClick={handlePrediction}
            disabled={isLoading || (!selectedFile && !recordedAudio)}
            className={`predict-button ${isLoading ? 'loading' : ''}`}
          >
            {isLoading ? (
              <span className="loading-text">
                <span className="loading-spinner"></span>
                Analyzing...
              </span>
            ) : (
              'Predict Emotion'
            )}
          </button>

          {predictedEmotion && (
            <div className="emotion-result">
              <h3 className="emotion-title">Predicted Emotion</h3>
              <div className="emotion-emoji">{getEmoji(predictedEmotion)}</div>
              <p className="emotion-text">{predictedEmotion.toUpperCase()}</p>
            </div>
          )}
        </div>
      </div>
    </section>
  );
};

export default AudioInput;