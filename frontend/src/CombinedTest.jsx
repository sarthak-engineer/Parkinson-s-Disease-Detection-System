import React, { useState } from 'react';
import Navbar from './components/Navbar';
import FileUpload from './components/FileUpload';
import ResultDisplay from './components/ResultDisplay';
import { Mic, Eye, Loader2 } from 'lucide-react';
import axios from 'axios';

const CombinedTest = () => {
    const [audioFile, setAudioFile] = useState(null);
    const [imageFile, setImageFile] = useState(null);
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);

    const handleSubmit = async () => {
        if (!audioFile || !imageFile) {
            setError("Please upload both audio and image files.");
            return;
        }

        setLoading(true);
        setError(null);
        setResult(null);

        const formData = new FormData();
        formData.append('audio', audioFile);
        formData.append('image', imageFile);

        try {
            // Assuming backend runs on port 8000
            const response = await axios.post('http://127.0.0.1:8000/pro', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });
            setResult(response.data);
        } catch (err) {
            console.error(err);
            setError("An error occurred during analysis. Please try again. Ensure the backend is running.");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen bg-slate-50">
            <Navbar />

            <div className="max-w-4xl mx-auto px-4 py-12">
                <div className="text-center mb-10">
                    <h1 className="text-3xl font-bold text-slate-900 mb-3">Parkinson's Disease Analysis</h1>
                    <p className="text-slate-600 max-w-2xl mx-auto">
                        Upload a voice recording and a drawing image for a comprehensive multi-modal analysis.
                    </p>
                </div>

                <div className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden">
                    <div className="p-8">
                        <div className="grid md:grid-cols-2 gap-8 mb-8">
                            <FileUpload
                                id="audio-upload"
                                label="Voice Recording"
                                accept="audio/*"
                                icon={Mic}
                                selectedFile={audioFile}
                                onFileSelect={setAudioFile}
                            />
                            <FileUpload
                                id="image-upload"
                                label="Drawing Image"
                                accept="image/*"
                                icon={Eye}
                                selectedFile={imageFile}
                                onFileSelect={setImageFile}
                            />
                        </div>

                        <button
                            onClick={handleSubmit}
                            disabled={loading || !audioFile || !imageFile}
                            className={`w-full py-4 rounded-xl font-bold text-lg text-white transition-all transform ${loading || !audioFile || !imageFile
                                ? 'bg-slate-300 cursor-not-allowed'
                                : 'bg-primary-600 hover:bg-primary-700 shadow-lg hover:shadow-xl hover:-translate-y-0.5'
                                }`}
                        >
                            {loading ? (
                                <span className="flex items-center justify-center">
                                    <Loader2 className="animate-spin mr-2 h-6 w-6" />
                                    Analyzing Data...
                                </span>
                            ) : (
                                "Start Analysis"
                            )}
                        </button>

                        <ResultDisplay result={result} error={error} />
                    </div>
                </div>
            </div>
        </div>
    );
};

export default CombinedTest;
