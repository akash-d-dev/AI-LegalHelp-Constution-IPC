import React from 'react';
import ChatContainer from './components/ChatContainer';
import './App.css';

function App() {
  return (
    <div className="App">
      <main className="app-main">
        <ChatContainer />
      </main>
      
      <footer className="app-footer">
        <p>
          © 2024 Constitutional AI - Educational tool for Indian Constitution & IPC
        </p>
      </footer>
    </div>
  );
}

export default App;
