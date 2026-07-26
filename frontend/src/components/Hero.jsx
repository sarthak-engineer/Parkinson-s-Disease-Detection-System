import React from 'react';
import { Activity, Brain, Stethoscope, Mic, Eye, Layers } from 'lucide-react';
import FeatureCard from './FeatureCard';

const Hero = () => {
    return (
        <div className="py-12 md:py-20">
            <div className="text-center max-w-3xl mx-auto mb-16">
                <div className="flex justify-center mb-4">
                    {/* Logo or specialized icon could go here */}
                </div>
                <h1 className="text-4xl md:text-5xl font-bold text-slate-900 mb-6 font-sans">
                    Parkinson's Disease Detection System
                </h1>
                <p className="text-lg text-slate-600 mb-8 max-w-2xl mx-auto">
                    Advanced AI-powered diagnostic tool using voice and drawing analysis to support early detection of Parkinson's disease.
                </p>

                <div className="flex flex-wrap justify-center gap-4 md:gap-8 mt-12">
                    <div className="bg-white px-6 py-4 rounded-xl shadow-sm border border-slate-100 flex items-center space-x-3">
                        <Activity className="h-5 w-5 text-primary-500" />
                        <div className="text-left">
                            <div className="text-lg font-bold text-slate-900">94.2%</div>
                            <div className="text-xs text-slate-500">Accuracy</div>
                        </div>
                    </div>
                    <div className="bg-white px-6 py-4 rounded-xl shadow-sm border border-slate-100 flex items-center space-x-3">
                        <Brain className="h-5 w-5 text-primary-500" />
                        <div className="text-left">
                            <div className="text-lg font-bold text-slate-900">3</div>
                            <div className="text-xs text-slate-500">AI Models</div>
                        </div>
                    </div>
                    <div className="bg-white px-6 py-4 rounded-xl shadow-sm border border-slate-100 flex items-center space-x-3">
                        <Stethoscope className="h-5 w-5 text-primary-500" />
                        <div className="text-left">
                            <div className="text-lg font-bold text-slate-900">10K+</div>
                            <div className="text-xs text-slate-500">Tests Conducted</div>
                        </div>
                    </div>
                </div>
            </div>

            <div className="flex justify-center max-w-7xl mx-auto">
                <div className="w-full md:w-1/3">
                    <FeatureCard
                        icon={Layers}
                        title="Start Analysis"
                        description="Comprehensive multi-modal assessment combining voice and drawing data for enhanced accuracy."
                        link="/combined"
                    />
                </div>
            </div>
        </div>
    );
};

export default Hero;
