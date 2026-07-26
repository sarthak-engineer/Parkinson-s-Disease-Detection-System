
import React from 'react';
import Navbar from './components/Navbar';
import Hero from './components/Hero';

const Home = () => {
    return (
        <div className="min-h-screen bg-white">
            <Navbar />
            <Hero />

            <section className="py-16 bg-slate-50">
                <div className="max-w-4xl mx-auto px-4 sm:px-6">
                    <h2 className="text-3xl font-bold text-slate-900 mb-6">About This System</h2>
                    <div className="space-y-4 text-slate-600 leading-relaxed">
                        <p>
                            This AI-powered diagnostic tool leverages advanced machine learning algorithms to analyze vocal patterns and drawing analysis data. Early detection of Parkinson's disease can significantly improve treatment outcomes and quality of life.
                        </p>
                        <p>
                            Our system uses state-of-the-art neural networks trained on extensive medical datasets to provide reliable, non-invasive screening. Results are presented with confidence scores to support clinical decision-making.
                        </p>
                        <div className="bg-white p-6 rounded-lg border border-slate-200 mt-8 shadow-sm">
                            <p className="font-semibold text-slate-800">
                                Important: This tool is designed to assist healthcare professionals and should not replace professional medical diagnosis. Always consult with a qualified neurologist for comprehensive evaluation.
                            </p>
                        </div>
                    </div>
                </div>
            </section>

            <footer className="bg-white py-8 border-t border-slate-200">
                <div className="max-w-7xl mx-auto px-4 text-center text-slate-500 text-sm">
                    &copy; {new Date().getFullYear()} Parkinson's Disease Detection System. All rights reserved.
                </div>
            </footer>
        </div>
    );
};

export default Home;
