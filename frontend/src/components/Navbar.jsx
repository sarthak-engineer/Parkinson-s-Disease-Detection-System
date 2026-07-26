import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Activity, Mic, Eye, Layers, Home } from 'lucide-react';

const Navbar = () => {
    const location = useLocation();

    const isActive = (path) => {
        return location.pathname === path ? 'text-primary-600 bg-primary-50' : 'text-slate-600 hover:text-primary-600 hover:bg-slate-50';
    };

    return (
        <nav className="border-b border-slate-200 bg-white sticky top-0 z-50">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="flex justify-between h-16">
                    <div className="flex items-center">
                        <Link to="/" className="flex items-center">
                            <Activity className="h-8 w-8 text-primary-500" />
                            <span className="ml-2 text-xl font-bold text-slate-900">PD Detector</span>
                        </Link>
                    </div>
                    <div className="flex items-center space-x-1">
                        <Link to="/" className={`px-4 py-2 rounded-md text-sm font-medium transition-colors flex items-center ${isActive('/')}`}>
                            <Home className="h-4 w-4 mr-2" />
                            Home
                        </Link>
                        <Link to="/combined" className={`px-4 py-2 rounded-md text-sm font-medium transition-colors flex items-center ${isActive('/combined')}`}>
                            <Layers className="h-4 w-4 mr-2" />
                            Start Analysis
                        </Link>
                    </div>
                </div>
            </div>
        </nav>
    );
};

export default Navbar;
