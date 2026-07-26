import React, { useCallback, useState } from 'react';
import { Upload, X, Check, FileAudio, ImageIcon } from 'lucide-react';

const FileUpload = ({ label, accept, icon: Icon, onFileSelect, selectedFile, id }) => {
    const [dragActive, setDragActive] = useState(false);

    const handleDrag = (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === "dragenter" || e.type === "dragover") {
            setDragActive(true);
        } else if (e.type === "dragleave") {
            setDragActive(false);
        }
    };

    const handleDrop = (e) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);

        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            onFileSelect(e.dataTransfer.files[0]);
        }
    };

    const handleChange = (e) => {
        if (e.target.files && e.target.files[0]) {
            onFileSelect(e.target.files[0]);
        }
    };

    return (
        <div className="w-full">
            <label className="block text-sm font-medium text-slate-700 mb-2">{label}</label>
            <div
                className={`relative border-2 border-dashed rounded-xl p-8 text-center transition-all duration-200 ${dragActive ? 'border-primary-500 bg-primary-50' : 'border-slate-300 hover:border-primary-400 bg-slate-50'
                    } ${selectedFile ? 'border-primary-500 bg-primary-50' : ''}`}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
            >
                <input
                    type="file"
                    id={id}
                    className="hidden"
                    accept={accept}
                    onChange={handleChange}
                />

                {selectedFile ? (
                    <div className="flex flex-col items-center">
                        <div className="h-12 w-12 bg-white rounded-full flex items-center justify-center shadow-sm mb-3">
                            <Check className="h-6 w-6 text-green-500" />
                        </div>
                        <p className="text-sm font-medium text-slate-900 mb-1">{selectedFile.name}</p>
                        <p className="text-xs text-slate-500 mb-4">{(selectedFile.size / 1024 / 1024).toFixed(2)} MB</p>
                        <button
                            onClick={(e) => {
                                e.preventDefault();
                                onFileSelect(null);
                            }}
                            className="text-xs text-red-500 hover:text-red-700 font-medium"
                        >
                            Remove file
                        </button>
                    </div>
                ) : (
                    <label htmlFor={id} className="cursor-pointer flex flex-col items-center">
                        <div className="h-12 w-12 bg-white rounded-full flex items-center justify-center shadow-sm mb-3">
                            <Icon className="h-6 w-6 text-slate-400" />
                        </div>
                        <p className="text-sm font-medium text-slate-900 mb-1">
                            <span className="text-primary-600 hover:text-primary-700">Upload a file</span> or drag and drop
                        </p>
                        <p className="text-xs text-slate-500">
                            {accept === "audio/*" ? "MP3, WAV up to 10MB" : "PNG, JPG up to 10MB"}
                        </p>
                    </label>
                )}
            </div>
        </div>
    );
};

export default FileUpload;
