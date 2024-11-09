import React from "react";
import ChordPlayer from "./components/ChordPlayer";

const App = () => {
  return (
    <div className="min-h-screen bg-black">
      {/* Header */}
      <header className="bg-gray-900 border-b border-gray-800">
        <div className="max-w-7xl mx-auto py-6 px-4">
          <h1 className="text-3xl font-bold text-white">
            Chord Detection System
          </h1>
          <p className="mt-2 text-sm text-orange-500">
            Upload an audio file to detect and visualize its chord progression
          </p>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto py-12 px-4">
        <div className="bg-gray-900 rounded-lg shadow-xl border border-gray-800">
          <ChordPlayer />
        </div>

        {/* Instructions Panel */}
        <div className="mt-8 bg-gray-900 rounded-lg shadow-xl border border-gray-800 p-6">
          <h2 className="text-xl font-semibold text-white mb-4">How to Use</h2>
          <ol className="list-decimal list-inside space-y-2 text-gray-400">
            <li>Click the upload button or drag and drop an audio file</li>
            <li>Wait for the chord detection system to process your audio</li>
            <li>Use the play/pause button to control playback</li>
            <li>Watch as the current chord is highlighted in real-time</li>
            <li>
              Click anywhere on the progress bar to jump to different parts
            </li>
          </ol>
        </div>

        {/* Footer */}
        <footer className="mt-8 text-center text-gray-600 text-sm">
          <p>Powered by TensorFlow and React</p>
        </footer>
      </main>
    </div>
  );
};

export default App;
