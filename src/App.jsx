import Navbar from './components/Navbar.jsx';
import Hero from './components/Hero.jsx';
import AudioInput from './components/AudioInput.jsx';
import './App.css';

const App = () => {
  return (
    <div className="app">
      <Navbar />
      <Hero />
      <AudioInput />
    </div>
  );
};

export default App;
