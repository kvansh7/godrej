import React, { useState, useEffect } from 'react';
import axios from 'axios';

const VendorMatching = () => {
  const [psOptions, setPsOptions] = useState([]);
  const [selectedPsId, setSelectedPsId] = useState('');
  const [topK, setTopK] = useState(20);
  const [batchSize, setBatchSize] = useState(5);
  const [aiProvider, setAiProvider] = useState('gemini');
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [loadingPs, setLoadingPs] = useState(true);
  const [expandedVendor, setExpandedVendor] = useState(null);
  const [selectedVendors, setSelectedVendors] = useState([]);

  // Web Search Feature
  const [webSearchCount, setWebSearchCount] = useState(5);
  const [webSearchLoading, setWebSearchLoading] = useState(false);
  const [webSearchResults, setWebSearchResults] = useState(null);
  const [expandedWebVendor, setExpandedWebVendor] = useState(null);
  const [selectedWebVendors, setSelectedWebVendors] = useState([]);

  // Dynamic parameters state
  const [evaluationParams, setEvaluationParams] = useState([
    { name: 'Domain Fit', weight: 40 },
    { name: 'Tools Fit', weight: 30 },
    { name: 'Experience', weight: 20 },
    { name: 'Scalability', weight: 10 }
  ]);

  // Parameter management functions
  const addParameter = () => {
    const newParam = { name: '', weight: '' };
    setEvaluationParams([...evaluationParams, newParam]);
  };

  const removeParameter = (index) => {
    if (evaluationParams.length > 1) {
      setEvaluationParams(evaluationParams.filter((_, i) => i !== index));
    }
  };

  const updateParameter = (index, field, value) => {
    const updated = [...evaluationParams];
    if (field === 'weight') {
      // Handle empty string or convert to number
      updated[index][field] = value === '' ? '' : Number(value);
    } else {
      updated[index][field] = value;
    }
    setEvaluationParams(updated);
  };

  const getTotalWeight = () => {
    return evaluationParams.reduce((sum, param) => {
      const weight = param.weight === '' ? 0 : Number(param.weight);
      return sum + weight;
    }, 0);
  };

  const normalizeWeights = () => {
    const total = getTotalWeight();
    if (total !== 100 && total > 0) {
      const normalized = evaluationParams.map(param => ({
        ...param,
        weight: Math.round(((param.weight === '' ? 0 : Number(param.weight)) / total) * 100)
      }));
      setEvaluationParams(normalized);
    }
  };

  useEffect(() => {
    const fetchPsOptions = async () => {
      try {
        const res = await axios.get('/api/problem_statements');
        const options = res.data.map(ps => ({ id: ps.id, title: ps.title }));
        setPsOptions(options);
        setLoadingPs(false);
      } catch (err) {
        setError('Failed to load problem statements');
        setLoadingPs(false);
      }
    };
    fetchPsOptions();
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setResults(null);
    setExpandedVendor(null);
    setSelectedVendors([]);
    setWebSearchResults(null);

    try {
      const res = await axios.post('/api/vendor_matching', {
        ps_id: selectedPsId,
        top_k: topK,
        batch_size: batchSize,
        ai_provider: aiProvider,
        evaluation_params: evaluationParams
      });
      setResults(res.data);
    } catch (err) {
      setError(err.response?.data?.error || 'Error matching vendors');
    } finally {
      setLoading(false);
    }
  };

  const handleWebSearch = async () => {
    if (!selectedPsId) {
      setError('Please select a problem statement first');
      return;
    }

    setWebSearchLoading(true);
    setError('');
    setWebSearchResults(null);

    try {
      const res = await axios.post('/api/web_search_vendors', {
        ps_id: selectedPsId,
        count: webSearchCount,
        evaluation_params: evaluationParams
      });

      // De-duplicate web sources across all vendors
      if (res.data?.vendors) {
        const uniqueSources = new Map();
        
        res.data.vendors.forEach(vendor => {
          if (vendor.web_sources) {
            vendor.web_sources = vendor.web_sources.filter(src => {
              const normalizedUrl = src.url.toLowerCase().trim();
              if (uniqueSources.has(normalizedUrl)) return false;
              uniqueSources.set(normalizedUrl, true);
              return true;
            });
          }
        });
        
        res.data.sources_count = uniqueSources.size;
      }

      setWebSearchResults(res.data);
    } catch (err) {
      setError(err.response?.data?.error || 'Error searching web for vendors');
    } finally {
      setWebSearchLoading(false);
    }
  };

  const toggleSelect = (index) => {
    setSelectedVendors((prev) =>
      prev.includes(index)
        ? prev.filter((i) => i !== index)
        : [...prev, index]
    );
  };

  const toggleWebSelect = (index) => {
    setSelectedWebVendors((prev) =>
      prev.includes(index)
        ? prev.filter((i) => i !== index)
        : [...prev, index]
    );
  };

  const getScoreColor = (score) => {
    if (score >= 80) return 'text-green-400';
    if (score >= 60) return 'text-yellow-400';
    if (score >= 40) return 'text-orange-400';
    return 'text-red-400';
  };

  const getProgressColor = (score) => {
    if (score >= 80) return 'bg-green-500';
    if (score >= 60) return 'bg-yellow-500';
    if (score >= 40) return 'bg-orange-500';
    return 'bg-red-500';
  };

  // Get all selected vendors for comparison
  const getAllSelectedVendors = () => {
    const vendors = [];
    
    // Add repository vendors
    selectedVendors.forEach(index => {
      if (results && results.results[index]) {
        vendors.push({
          ...results.results[index],
          source: 'Repository',
          displayIndex: `#${index + 1}`
        });
      }
    });
    
    // Add web vendors
    selectedWebVendors.forEach(index => {
      if (webSearchResults && webSearchResults.vendors[index]) {
        vendors.push({
          ...webSearchResults.vendors[index],
          source: 'Web',
          displayIndex: `WEB #${index + 1}`
        });
      }
    });
    
    return vendors;
  };

  if (loadingPs) return <p className="text-center text-white">Loading problem statements...</p>;
  if (error && !psOptions.length) return <p className="text-center text-red-400">{error}</p>;

  const allSelectedVendors = getAllSelectedVendors();

  return (
    <div className="max-w-6xl mx-auto py-16 space-y-8">
      <h1 className="text-3xl font-bold text-white text-center mb-6">
        🔍 Vendor Matching
      </h1>

      {psOptions.length === 0 ? (
        <p className="text-center text-yellow-400">
          ⚠️ No problem statements submitted yet. Please submit one first.
        </p>
      ) : (
        <>
          {/* Form */}
          <div className="glass-card p-6 space-y-4 shadow-xl">
            <form onSubmit={handleSubmit} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-white/80 mb-1">
                  Select Problem Statement
                </label>
                <select
                  value={selectedPsId}
                  onChange={(e) => setSelectedPsId(e.target.value)}
                  className="w-full p-3 rounded-md bg-white/5 text-white border border-white/10 focus:outline-none focus:ring-2 focus:ring-blue-400"
                  required
                >
                  <option value="" className="bg-gray-800 text-white">Select a problem statement</option>
                  {psOptions.map((ps) => (
                    <option key={ps.id} value={ps.id} className="bg-gray-800 text-white">
                      {ps.title} (ID: {ps.id})
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-white/80">
                  Number of vendors to shortlist ({topK})
                </label>
                <input
                  type="range"
                  min="5"
                  max="50"
                  value={topK}
                  onChange={(e) => setTopK(Number(e.target.value))}
                  className="w-full mt-1"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-white/80">
                  Batch size for LLM evaluation ({batchSize})
                </label>
                <input
                  type="range"
                  min="1"
                  max="10"
                  value={batchSize}
                  onChange={(e) => setBatchSize(Number(e.target.value))}
                  className="w-full mt-1"
                />
              </div>

              {/* AI Provider Selection */}
              <div>
                <label className="block text-sm font-medium text-white/80 mb-1">
                  AI Provider
                </label>
                <select
                  value={aiProvider}
                  onChange={(e) => setAiProvider(e.target.value)}
                  className="w-full p-3 rounded-md bg-white/5 text-white border border-white/10 focus:outline-none focus:ring-2 focus:ring-blue-400"
                >
                  <option value="gemini" className="bg-gray-800 text-white">Google Gemini</option>
                  <option value="openai" className="bg-gray-800 text-white">OpenAI GPT</option>
                </select>
              </div>

              {/* Dynamic Evaluation Parameters - Enhanced UI */}
              <div className="space-y-4 p-5 rounded-lg bg-gradient-to-br from-blue-500/10 to-purple-500/10 border border-blue-400/20">
                <div className="flex items-center justify-between">
                  <div>
                    <label className="block text-base font-semibold text-white">
                      Evaluation Parameters
                    </label>
                    <p className="text-xs text-white/60 mt-1">Define criteria and their weights for vendor evaluation</p>
                  </div>
                  <button
                    type="button"
                    onClick={addParameter}
                    className="flex items-center space-x-2 px-4 py-2 bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white text-sm rounded-lg transition-all duration-300 shadow-lg hover:shadow-blue-500/50"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                    </svg>
                    <span>Add Parameter</span>
                  </button>
                </div>
                
                <div className="space-y-3">
                  {evaluationParams.map((param, index) => (
                    <div key={index} className="flex items-center space-x-3 p-4 bg-white/5 backdrop-blur-sm rounded-lg border border-white/10 hover:border-white/20 transition-all">
                      <div className="flex-1">
                        <input
                          type="text"
                          placeholder="e.g., Technical Expertise"
                          value={param.name}
                          onChange={(e) => updateParameter(index, 'name', e.target.value)}
                          className="w-full p-2.5 rounded-md bg-white/10 text-white placeholder-white/40 border border-white/20 focus:outline-none focus:ring-2 focus:ring-blue-400 focus:border-transparent"
                        />
                      </div>
                      <div className="flex items-center space-x-2 bg-white/5 px-3 py-2 rounded-md border border-white/10">
                        <input
                          type="number"
                          min="0"
                          max="100"
                          placeholder="0"
                          value={param.weight}
                          onChange={(e) => updateParameter(index, 'weight', e.target.value)}
                          className="w-16 p-1.5 rounded bg-white/10 text-white text-center placeholder-white/30 border border-white/20 focus:outline-none focus:ring-2 focus:ring-blue-400 focus:border-transparent"
                        />
                        <span className="text-white/70 font-medium">%</span>
                      </div>
                      {evaluationParams.length > 1 && (
                        <button
                          type="button"
                          onClick={() => removeParameter(index)}
                          className="p-2 text-red-400 hover:text-red-300 hover:bg-red-500/10 rounded-md transition-all"
                          title="Remove parameter"
                        >
                          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                          </svg>
                        </button>
                      )}
                    </div>
                  ))}
                </div>
                
                <div className="flex items-center justify-between p-3 bg-white/5 rounded-lg border border-white/10">
                  <div className="flex items-center space-x-3">
                    <span className="text-sm text-white/60">Total Weight:</span>
                    <span className={`text-lg font-bold ${getTotalWeight() === 100 ? 'text-green-400' : 'text-orange-400'}`}>
                      {getTotalWeight()}%
                    </span>
                    {getTotalWeight() === 100 && (
                      <svg className="w-5 h-5 text-green-400" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                      </svg>
                    )}
                  </div>
                  {getTotalWeight() !== 100 && getTotalWeight() > 0 && (
                    <button
                      type="button"
                      onClick={normalizeWeights}
                      className="px-4 py-2 bg-gradient-to-r from-orange-600 to-orange-700 hover:from-orange-700 hover:to-orange-800 text-white text-sm rounded-lg transition-all duration-300 shadow-lg hover:shadow-orange-500/50"
                    >
                      Normalize to 100%
                    </button>
                  )}
                </div>
              </div>

              <button
                type="submit"
                disabled={loading || !selectedPsId || getTotalWeight() !== 100}
                className={`w-full py-3 rounded-lg font-semibold text-white transition-all duration-300 ${
                  loading || !selectedPsId || getTotalWeight() !== 100
                    ? 'bg-white/20 cursor-not-allowed'
                    : 'bg-blue-600 hover:bg-blue-700'
                }`}
              >
                {loading ? 'Matching...' : 'Match Vendors from Repository'}
              </button>
              {getTotalWeight() !== 100 && (
                <p className="text-orange-400 text-sm text-center">
                  Please ensure parameter weights total 100%
                </p>
              )}
            </form>
            {error && <p className="text-red-400 text-center mt-2">{error}</p>}
          </div>

          {/* Repository Results */}
          {results && (
            <div className="space-y-6">
              <h2 className="text-2xl font-bold text-white text-center">🎯 Repository Vendors Results</h2>

              <div className="grid grid-cols-3 gap-4">
                <div className="glass-card p-4 text-center shadow">
                  <p className="text-white/80">Total Vendors Analyzed</p>
                  <p className="text-3xl font-bold text-white">{results.total_vendors_analyzed}</p>
                </div>
                <div className="glass-card p-4 text-center shadow">
                  <p className="text-white/80">Shortlisted Vendors</p>
                  <p className="text-3xl font-bold text-white">{results.shortlisted_vendors}</p>
                </div>
                <div className="glass-card p-4 text-center shadow">
                  <p className="text-white/80">Top Composite Score</p>
                  <p className={`text-3xl font-bold ${getScoreColor(results.top_composite_score)}`}>
                    {results.top_composite_score.toFixed(1)}%
                  </p>
                </div>
              </div>

              <div className="space-y-4">
                {results.results.map((result, index) => (
                  <div key={index} className="glass-card p-4 shadow rounded">
                    <div 
                      className="flex items-center justify-between cursor-pointer"
                      onClick={() => setExpandedVendor(expandedVendor === index ? null : index)}
                    >
                      <div className="flex items-center space-x-4 flex-1">
                        <input
                          type="checkbox"
                          checked={selectedVendors.includes(index)}
                          onChange={(e) => {
                            e.stopPropagation();
                            toggleSelect(index);
                          }}
                          className="form-checkbox h-5 w-5 text-blue-600 bg-white/5 border-white/10 rounded"
                        />
                        <span className="text-white/60 font-bold text-lg">#{index + 1}</span>
                        <div className="flex-1">
                          <h3 className="font-semibold text-white text-lg">{result.name}</h3>
                          <p className={`text-2xl font-bold ${getScoreColor(result.composite_score)}`}>
                            {result.composite_score.toFixed(1)}%
                          </p>
                        </div>
                      </div>
                      <svg 
                        className={`w-6 h-6 text-white/60 transition-transform ${expandedVendor === index ? 'rotate-180' : ''}`}
                        fill="none" 
                        stroke="currentColor" 
                        viewBox="0 0 24 24"
                      >
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                      </svg>
                    </div>

                    {expandedVendor === index && (
                      <div className="mt-4 space-y-4 pt-4 border-t border-white/10">
                        <div className="grid md:grid-cols-2 gap-4">
                          {evaluationParams.map((param, paramIndex) => {
                            const scoreKey = param.name.toLowerCase().replace(/[^a-z0-9]/g, '_') + '_score';
                            const score = result[scoreKey] || 0;
                            return (
                              <div key={paramIndex}>
                                <div className="flex justify-between mb-1">
                                  <span className="text-white/80 font-medium">{param.name}</span>
                                  <span className={`font-semibold ${getScoreColor(score)}`}>
                                    {score.toFixed(1)}%
                                  </span>
                                </div>
                                <div className="w-full bg-white/10 rounded-full h-2.5">
                                  <div
                                    className={`h-2.5 rounded-full ${getProgressColor(score)}`}
                                    style={{ width: `${score}%` }}
                                  ></div>
                                </div>
                              </div>
                            );
                          })}
                        </div>

                        <div>
                          <p className="font-medium text-white mb-2">📝 Justification</p>
                          <p className="text-white/80 text-sm">{result.justification}</p>
                        </div>

                        {result.strengths && result.strengths.length > 0 && (
                          <div>
                            <p className="font-medium text-green-400 mb-2">✅ Key Strengths</p>
                            <ul className="list-disc list-inside space-y-1">
                              {result.strengths.map((strength, i) => (
                                <li key={i} className="text-white/80 text-sm">{strength}</li>
                              ))}
                            </ul>
                          </div>
                        )}

                        {result.concerns && result.concerns.length > 0 && (
                          <div>
                            <p className="font-medium text-orange-400 mb-2">⚠️ Potential Concerns</p>
                            <ul className="list-disc list-inside space-y-1">
                              {result.concerns.map((concern, i) => (
                                <li key={i} className="text-white/80 text-sm">{concern}</li>
                              ))}
                            </ul>
                          </div>
                        )}

                        {result.similarity_percentage && (
                          <div>
                            <div className="flex justify-between mb-1">
                              <span className="text-white/60 text-sm">Semantic Similarity</span>
                              <span className="text-white/60 text-sm">{result.similarity_percentage.toFixed(1)}%</span>
                            </div>
                            <div className="w-full bg-white/5 rounded-full h-1.5">
                              <div
                                className="bg-blue-400 h-1.5 rounded-full"
                                style={{ width: `${result.similarity_percentage}%` }}
                              ></div>
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* 🌐 Web Search Section */}
          <div className="glass-card p-6 space-y-4 shadow-xl border-2 border-purple-500/30">
            <div className="flex items-center justify-between">
              <h2 className="text-2xl font-bold text-white">🌐 Discover Vendors from Web</h2>
              <span className="text-purple-400 text-sm font-semibold">Powered by OpenAI</span>
            </div>

            <p className="text-white/70 text-sm">
              Search the web to discover and evaluate vendors that match your problem statement.
            </p>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-white/80 mb-2">
                  Number of vendors to discover ({webSearchCount})
                </label>
                <input
                  type="range"
                  min="3"
                  max="10"
                  value={webSearchCount}
                  onChange={(e) => setWebSearchCount(Number(e.target.value))}
                  className="w-full"
                />
              </div>

              <button
                onClick={handleWebSearch}
                disabled={webSearchLoading || !selectedPsId || getTotalWeight() !== 100}
                className={`w-full py-3 rounded-lg font-semibold text-white transition-all duration-300 ${
                  webSearchLoading || !selectedPsId || getTotalWeight() !== 100
                    ? 'bg-white/20 cursor-not-allowed'
                    : 'bg-purple-600 hover:bg-purple-700'
                }`}
              >
                {webSearchLoading ? (
                  <span className="flex items-center justify-center">
                    <svg className="animate-spin h-5 w-5 mr-3" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                    </svg>
                    Searching Web...
                  </span>
                ) : (
                  '🔎 Search Web for Vendors'
                )}
              </button>
            </div>
          </div>

          {/* 🌐 Web Search Results */}
          {webSearchResults && (
            <div className="space-y-6">
              <h2 className="text-2xl font-bold text-white text-center">🌐 Web-Discovered Vendors</h2>

              <div className="grid grid-cols-3 gap-4">
                <div className="glass-card p-4 text-center shadow border border-purple-500/30">
                  <p className="text-white/80">Vendors Found</p>
                  <p className="text-3xl font-bold text-purple-400">{webSearchResults.total_found}</p>
                </div>
                <div className="glass-card p-4 text-center shadow border border-purple-500/30">
                  <p className="text-white/80">Unique Sources</p>
                  <p className="text-3xl font-bold text-purple-400">{webSearchResults.sources_count}</p>
                </div>
                <div className="glass-card p-4 text-center shadow border border-purple-500/30">
                  <p className="text-white/80">Top Score</p>
                  <p className={`text-3xl font-bold ${getScoreColor(webSearchResults.top_score)}`}>
                    {webSearchResults.top_score.toFixed(1)}%
                  </p>
                </div>
              </div>

              <div className="space-y-4">
                {webSearchResults.vendors.map((vendor, index) => (
                  <div key={index} className="glass-card p-4 shadow rounded border border-purple-500/20">
                    <div
                      className="flex items-center justify-between cursor-pointer"
                      onClick={() => setExpandedWebVendor(expandedWebVendor === index ? null : index)}
                    >
                      <div className="flex items-center space-x-4 flex-1">
                        <input
                          type="checkbox"
                          checked={selectedWebVendors.includes(index)}
                          onChange={(e) => {
                            e.stopPropagation();
                            toggleWebSelect(index);
                          }}
                          className="form-checkbox h-5 w-5 text-purple-500 bg-white/5 border-white/10 rounded"
                        />
                        <span className="bg-purple-500/20 text-purple-300 px-2 py-1 rounded text-xs font-bold">
                          WEB #{index + 1}
                        </span>
                        <div className="flex-1">
                          <h3 className="font-semibold text-white text-lg">{vendor.name}</h3>
                          <p className={`text-2xl font-bold ${getScoreColor(vendor.composite_score)}`}>
                            {vendor.composite_score.toFixed(1)}%
                          </p>
                        </div>
                      </div>
                      <svg
                        className={`w-6 h-6 text-white/60 transition-transform ${expandedWebVendor === index ? 'rotate-180' : ''}`}
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                      </svg>
                    </div>

                    {expandedWebVendor === index && (
                      <div className="mt-4 space-y-4 pt-4 border-t border-white/10">
                        {/* Scores */}
                        <div className="grid md:grid-cols-2 gap-4">
                          {evaluationParams.map((param, i) => {
                            const key = param.name.toLowerCase().replace(/[^a-z0-9]/g, '_') + '_score';
                            const score = vendor[key] || 0;
                            return (
                              <div key={i}>
                                <div className="flex justify-between mb-1">
                                  <span className="text-white/80 font-medium text-sm">{param.name}</span>
                                  <span className={`font-semibold text-sm ${getScoreColor(score)}`}>
                                    {score.toFixed(1)}%
                                  </span>
                                </div>
                                <div className="w-full bg-white/10 rounded-full h-2">
                                  <div
                                    className={`h-2 rounded-full ${getProgressColor(score)}`}
                                    style={{ width: `${score}%` }}
                                  ></div>
                                </div>
                              </div>
                            );
                          })}
                        </div>

                        {vendor.description && (
                          <div>
                            <p className="font-medium text-white mb-1 text-sm">📋 Description</p>
                            <p className="text-white/70 text-sm">{vendor.description}</p>
                          </div>
                        )}

                        <div>
                          <p className="font-medium text-white mb-1 text-sm">📝 Evaluation</p>
                          <p className="text-white/70 text-sm">{vendor.justification}</p>
                        </div>

                        {vendor.strengths?.length > 0 && (
                          <div>
                            <p className="font-medium text-green-400 mb-1 text-sm">✅ Strengths</p>
                            <ul className="list-disc list-inside space-y-1">
                              {vendor.strengths.map((s, i) => (
                                <li key={i} className="text-white/70 text-sm">{s}</li>
                              ))}
                            </ul>
                          </div>
                        )}

                        {vendor.concerns?.length > 0 && (
                          <div>
                            <p className="font-medium text-orange-400 mb-1 text-sm">⚠️ Concerns</p>
                            <ul className="list-disc list-inside space-y-1">
                              {vendor.concerns.map((c, i) => (
                                <li key={i} className="text-white/70 text-sm">{c}</li>
                              ))}
                            </ul>
                          </div>
                        )}

                        {vendor.web_sources?.length > 0 && (
                          <div>
                            <p className="font-medium text-blue-400 mb-2 text-sm">🔗 Sources</p>
                            <div className="space-y-1">
                              {vendor.web_sources.map((src, i) => (
                                <a
                                  key={i}
                                  href={src.url}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="block text-blue-300 hover:text-blue-200 text-xs hover:underline"
                                >
                                  {src.title || src.url}
                                </a>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Comparison Table - Shown at the bottom when 2+ vendors selected */}
          {allSelectedVendors.length >= 2 && (
            <div className="space-y-4 mt-8">
              <h2 className="text-2xl font-bold text-white text-center">📊 Vendor Comparison</h2>
              <div className="glass-card p-4 shadow rounded overflow-x-auto">
                <table className="min-w-full divide-y divide-white/10">
                  <thead>
                    <tr>
                      <th className="px-6 py-3 text-left text-sm font-medium text-white/80">Vendor</th>
                      <th className="px-6 py-3 text-left text-sm font-medium text-white/80">Source</th>
                      <th className="px-6 py-3 text-left text-sm font-medium text-white/80">Composite Score</th>
                      {evaluationParams.map((param, index) => (
                        <th key={index} className="px-6 py-3 text-left text-sm font-medium text-white/80">
                          {param.name}
                        </th>
                      ))}
                      <th className="px-6 py-3 text-left text-sm font-medium text-white/80">Semantic Similarity</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-white/10">
                    {allSelectedVendors.map((vendor, idx) => (
                      <tr key={idx}>
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-white">
                          {vendor.name} ({vendor.displayIndex})
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm">
                          <span className={`px-2 py-1 rounded text-xs font-semibold ${
                            vendor.source === 'Web' 
                              ? 'bg-purple-500/20 text-purple-300' 
                              : 'bg-blue-500/20 text-blue-300'
                          }`}>
                            {vendor.source}
                          </span>
                        </td>
                        <td className={`px-6 py-4 whitespace-nowrap text-sm font-semibold ${getScoreColor(vendor.composite_score)}`}>
                          {vendor.composite_score.toFixed(1)}%
                        </td>
                        {evaluationParams.map((param, paramIndex) => {
                          const scoreKey = param.name.toLowerCase().replace(/[^a-z0-9]/g, '_') + '_score';
                          const score = vendor[scoreKey] || 0;
                          return (
                            <td key={paramIndex} className={`px-6 py-4 whitespace-nowrap text-sm ${getScoreColor(score)}`}>
                              {score.toFixed(1)}%
                            </td>
                          );
                        })}
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-white/80">
                          {vendor.similarity_percentage ? `${vendor.similarity_percentage.toFixed(1)}%` : 'N/A'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default VendorMatching;