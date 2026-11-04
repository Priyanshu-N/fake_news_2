import React, { useState } from 'react';

export default function App() {
  const [title, setTitle] = useState('');
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleAnalyze = async (e) => {
    e.preventDefault();
    if (!text.trim()) {
      setError('Please provide the main text of the news article.');
      return;
    }

    setLoading(true);
    setError('');
    setResult(null);

    try {
      const response = await fetch('http://127.0.0.1:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title, text }),
      });

      if (!response.ok) {
        throw new Error('Server returned an error. Ensure FastAPI backend is running.');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message || 'Failed to connect to TruthLens API.');
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setTitle('');
    setText('');
    setResult(null);
    setError('');
  };

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 flex flex-col justify-between p-4 md:p-8">
      {/* Header */}
      <header className="max-w-4xl mx-auto w-full text-center py-6 border-b border-slate-800">
        <h1 className="text-4xl font-extrabold tracking-tight text-white mb-2">
          TruthLens <span className="text-blue-500">AI</span> 🔍
        </h1>
        <p className="text-slate-400 text-sm md:text-base">
          Fake News Detection System Powered by Natural Language Processing & Machine Learning
        </p>
      </header>

      {/* Main Form Section */}
      <main className="max-w-3xl mx-auto w-full my-8 flex-1">
        <form onSubmit={handleAnalyze} className="bg-slate-800 border border-slate-700 rounded-xl p-6 shadow-xl">
          <div className="mb-4">
            <label className="block text-slate-300 font-semibold mb-2 text-sm">
              Headline / Title <span className="text-slate-500 font-normal">(Optional)</span>
            </label>
            <input
              type="text"
              placeholder="e.g., Government Announces New Policy..."
              className="w-full bg-slate-900 border border-slate-700 rounded-lg p-3 text-slate-100 placeholder-slate-500 focus:outline-none focus:border-blue-500 transition"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
            />
          </div>

          <div className="mb-6">
            <label className="block text-slate-300 font-semibold mb-2 text-sm">
              Article Content <span className="text-red-400">*</span>
            </label>
            <textarea
              rows="6"
              placeholder="Paste the full news article text here..."
              className="w-full bg-slate-900 border border-slate-700 rounded-lg p-3 text-slate-100 placeholder-slate-500 focus:outline-none focus:border-blue-500 transition resize-none"
              value={text}
              onChange={(e) => setText(e.target.value)}
              required
            ></textarea>
          </div>

          {error && (
            <div className="mb-4 p-3 bg-red-900/40 border border-red-500 rounded-lg text-red-300 text-sm">
              ⚠️ {error}
            </div>
          )}

          <div className="flex gap-4">
            <button
              type="submit"
              disabled={loading}
              className="flex-1 bg-blue-600 hover:bg-blue-500 text-white font-semibold py-3 px-6 rounded-lg shadow transition disabled:opacity-50"
            >
              {loading ? 'Analyzing Text...' : 'Analyze Article 🚀'}
            </button>
            <button
              type="button"
              onClick={handleClear}
              className="bg-slate-700 hover:bg-slate-600 text-slate-300 font-medium py-3 px-4 rounded-lg transition"
            >
              Clear
            </button>
          </div>
        </form>

        {/* Prediction Results Display */}
        {result && (
          <div className="mt-6 bg-slate-800 border border-slate-700 rounded-xl p-6 shadow-xl">
            <div className="flex items-center justify-between border-b border-slate-700 pb-4 mb-4">
              <div>
                <span className="text-xs uppercase font-bold tracking-wider text-slate-400">
                  Classification Result
                </span>
                <h3
                  className={`text-3xl font-extrabold mt-1 ${
                    result.label === 1 ? 'text-red-400' : 'text-emerald-400'
                  }`}
                >
                  {result.label_name}
                </h3>
              </div>
              <div className="text-right">
                <span className="text-xs uppercase font-bold tracking-wider text-slate-400">
                  Model Confidence
                </span>
                <p className="text-2xl font-bold text-white">{result.confidence}%</p>
              </div>
            </div>

            {/* Confidence Bar - FIX APPLIED HERE */}
            <div className="w-full bg-slate-900 rounded-full h-3 mb-4 overflow-hidden">
              <div
                className={`h-full transition-all duration-500 ${
                  result.label === 1 ? 'bg-red-500' : 'bg-emerald-500'
                }`}
                style={{ width: `${result.confidence}%` }}
              ></div>
            </div>

            <p className="text-xs text-slate-400 italic">
              ℹ️ {result.disclaimer}
            </p>
          </div>
        )}
      </main>

      <footer className="text-center text-xs text-slate-500 border-t border-slate-800 pt-4">
        TruthLens AI Project • Natural Language Processing & Supervised Learning
      </footer>
    </div>
  );
}