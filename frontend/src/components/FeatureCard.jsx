import React from 'react';
import { Link } from 'react-router-dom';

const FeatureCard = ({ icon: Icon, title, description, link, buttonText = "Start Test" }) => {
    return (
        <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm hover:shadow-md transition-shadow">
            <div className="h-12 w-12 bg-primary-50 rounded-lg flex items-center justify-center mb-4">
                <Icon className="h-6 w-6 text-primary-600" />
            </div>
            <h3 className="text-xl font-semibold text-slate-900 mb-2">{title}</h3>
            <p className="text-slate-600 mb-6 text-sm leading-relaxed">{description}</p>
            <Link
                to={link}
                className="block w-full text-center bg-primary-500 hover:bg-primary-600 text-white font-medium py-2.5 px-4 rounded-lg transition-colors"
            >
                {buttonText}
            </Link>
        </div>
    );
};

export default FeatureCard;
