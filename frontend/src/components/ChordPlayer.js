import React, { useState, useRef, useEffect } from "react";
import { Play, Pause, SkipBack, Upload } from "lucide-react";

const ChordPlayer = () => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [audioFile, setAudioFile] = useState(null);
  const [chordData, setChordData] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [duration, setDuration] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);
  const audioRef = useRef(null);
  const progressRef = useRef(null);

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setIsProcessing(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://localhost:5000/api/process-audio", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (data.success) {
        setChordData(data.chordData);
        setAudioFile(URL.createObjectURL(file));
      } else {
        alert("Error processing audio: " + data.error);
      }
    } catch (error) {
      alert("Error uploading file: " + error.message);
    } finally {
      setIsProcessing(false);
    }
  };

  const getCurrentChord = (time) => {
    const currentChord = chordData.find(
      (segment) => time >= segment.startTime && time < segment.endTime
    );
    return currentChord ? currentChord.chord : "";
  };

  const formatTime = (timeInSeconds) => {
    const minutes = Math.floor(timeInSeconds / 60);
    const seconds = Math.floor(timeInSeconds % 60);
    return `${minutes.toString().padStart(2, "0")}:${seconds
      .toString()
      .padStart(2, "0")}`;
  };

  const handlePlayPause = () => {
    if (audioRef.current.paused) {
      audioRef.current.play();
      setIsPlaying(true);
    } else {
      audioRef.current.pause();
      setIsPlaying(false);
    }
  };

  const handleTimeUpdate = () => {
    if (audioRef.current) {
      setCurrentTime(audioRef.current.currentTime);
    }
  };

  const handleLoadedMetadata = () => {
    if (audioRef.current) {
      setDuration(audioRef.current.duration);
    }
  };

  const handleProgressClick = (e) => {
    if (progressRef.current && audioRef.current) {
      const rect = progressRef.current.getBoundingClientRect();
      const pos = (e.clientX - rect.left) / rect.width;
      const newTime = pos * audioRef.current.duration;
      audioRef.current.currentTime = newTime;
      setCurrentTime(newTime);
    }
  };

  const handleReset = () => {
    if (audioRef.current) {
      audioRef.current.currentTime = 0;
      setCurrentTime(0);
      setIsPlaying(false);
    }
  };

  useEffect(() => {
    return () => {
      // Cleanup function to revoke object URL when component unmounts
      if (audioFile) {
        URL.revokeObjectURL(audioFile);
      }
    };
  }, [audioFile]);

  return (
    <div className="w-full max-w-2xl mx-auto p-6 bg-gray-900 rounded-lg shadow-xl border border-gray-800">
      {/* File upload */}
      <div className="mb-6">
        <label className="flex flex-col items-center p-4 border-2 border-dashed border-orange-500/50 rounded-lg cursor-pointer hover:bg-gray-800 transition-colors">
          <Upload className="w-8 h-8 text-orange-500 mb-2" />
          <span className="text-sm text-orange-500">Upload audio file</span>
          <input
            type="file"
            accept="audio/*"
            onChange={handleFileUpload}
            className="hidden"
          />
        </label>
      </div>

      {isProcessing ? (
        <div className="text-center py-8">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-t-2 border-orange-500 mb-2"></div>
          <div className="text-orange-500">Processing audio file...</div>
        </div>
      ) : isLoading ? (
        <div className="text-center py-8 text-orange-500">Loading...</div>
      ) : audioFile ? (
        <>
          <audio
            ref={audioRef}
            onTimeUpdate={handleTimeUpdate}
            onLoadedMetadata={handleLoadedMetadata}
            onEnded={() => setIsPlaying(false)}
            src={audioFile}
          />

          {/* Current chord display */}
          <div className="mb-8 text-center">
            <div className="text-6xl font-bold text-white mb-2">
              {getCurrentChord(currentTime)}
            </div>
            <div className="text-orange-500">
              {formatTime(currentTime)} / {formatTime(duration)}
            </div>
          </div>

          {/* Progress bar */}
          <div
            ref={progressRef}
            className="w-full h-2 bg-gray-800 rounded-full mb-4 cursor-pointer"
            onClick={handleProgressClick}
          >
            <div
              className="h-full bg-orange-500 rounded-full transition-all duration-100"
              style={{
                width: `${(currentTime / (duration || 1)) * 100}%`,
              }}
            />
          </div>

          {/* Controls */}
          <div className="flex justify-center items-center gap-4">
            <button
              onClick={handleReset}
              className="p-2 rounded-full hover:bg-gray-800 text-orange-500 border border-orange-500/50"
            >
              <SkipBack className="w-6 h-6" />
            </button>
            <button
              onClick={handlePlayPause}
              className="p-4 rounded-full bg-orange-500 hover:bg-orange-600 text-white transition-colors"
            >
              {isPlaying ? (
                <Pause className="w-8 h-8" />
              ) : (
                <Play className="w-8 h-8" />
              )}
            </button>
          </div>

          {/* Chord timeline */}
          <div className="mt-8 border border-gray-800 rounded-lg overflow-hidden">
            {chordData.map((segment, index) => (
              <div
                key={index}
                className={`flex justify-between p-3 ${
                  currentTime >= segment.startTime &&
                  currentTime < segment.endTime
                    ? "bg-orange-500/10"
                    : index % 2 === 0
                    ? "bg-gray-800/50"
                    : "bg-gray-900"
                } transition-colors`}
              >
                <span className="font-medium text-white">{segment.chord}</span>
                <span className="text-orange-500">
                  {formatTime(segment.startTime)} -{" "}
                  {formatTime(segment.endTime)}
                </span>
              </div>
            ))}
          </div>
        </>
      ) : (
        <div className="text-center py-8 text-orange-500">
          Upload an audio file to begin
        </div>
      )}
    </div>
  );
};

export default ChordPlayer;
