import React from 'react';
import { AlertCircle, CheckCircle } from 'lucide-react';

const ResultDisplay = ({ result, error }) => {
    if (error) {
        return (
            <div className="mt-8 p-4 bg-red-50 border border-red-200 rounded-xl flex items-start">
                <AlertCircle className="h-5 w-5 text-red-500 mt-0.5 mr-3 flex-shrink-0" />
                <p className="text-red-700">{error}</p>
            </div>
        );
    }

    if (!result) return null;

    const isPositive = result.prediction === "Positive";

    return (
        <div className="mt-8 animate-fade-in">
            <div className={`p-6 rounded-xl border-2 ${isPositive ? 'border-red-100 bg-red-50' : 'border-green-100 bg-green-50'
                }`}>
                <div className="flex items-center mb-4">
                    {isPositive ? (
                        <AlertCircle className="h-8 w-8 text-red-500 mr-3" />
                    ) : (
                        <CheckCircle className="h-8 w-8 text-green-500 mr-3" />
                    )}
                    <h3 className={`text-2xl font-bold ${isPositive ? 'text-red-800' : 'text-green-800'
                        }`}>
                        Result: {result.prediction}
                    </h3>
                </div>

                <div className="space-y-4">
                    <div>
                        <div className="flex justify-between text-sm font-medium mb-1">
                            <span className="text-slate-600">Confidence Score</span>
                            <span className="text-slate-900">{(result.confidence * 100).toFixed(1)}%</span>
                        </div>
                        <div className="w-full bg-white rounded-full h-2.5">
                            <div
                                className={`h-2.5 rounded-full ${isPositive ? 'bg-red-500' : 'bg-green-500'}`}
                                style={{ width: `${result.confidence * 100}%` }}
                            ></div>
                        </div>
                    </div>

                    <p className="text-sm text-slate-600 mt-4 leading-relaxed">
                        {isPositive
                            ? "The analysis has detected patterns consistent with Parkinson's disease. This is a screening result, not a medical diagnosis. Please consult with a healthcare professional for a comprehensive evaluation."
                            : "The analysis did not detect patterns indicating Parkinson's disease. However, if you are experiencing symptoms, please consult with a healthcare professional."
                        }
                    </p>
                </div>
            </div>

            <div className="mt-4 text-center">
                <p className="text-xs text-slate-400">
                    Disclaimer: This AI tool is for assistance only and does not replace professional medical advice.
                </p>
            </div>
        </div>
    );
};

export default ResultDisplay;
