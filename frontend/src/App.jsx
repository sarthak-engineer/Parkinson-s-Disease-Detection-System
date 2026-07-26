import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Home from './Home';
import CombinedTest from './CombinedTest';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/combined" element={<CombinedTest />} />
      </Routes>
    </Router>
  );
}

export default App;
