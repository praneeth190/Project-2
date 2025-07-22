import React from 'react';
import AudioImage from '../assets/AudioImage.svg';

const Hero = () => {
  return (
    <section className="hero">
      <div className="container">
        <div className="hero-content">
          <div className="hero-text">
            <h1 className="hero-title">
              Speech Emotion <span>Recognizer</span>
            </h1>
            <p className="hero-description">
              Discover the power of emotional intelligence through voice analysis. Our
              advanced deep learning model leverages MFCC features to decode the
              emotional nuances in human speech with remarkable accuracy.
            </p>
            <div className="tags">
              <span className="tag">Deep Learning</span>
              <span className="tag">MFCC Analysis</span>
              <span className="tag">Real-time Processing</span>
            </div>
          </div>
          <div className="hero-illustration">
            <img src={AudioImage} alt="Speech Emotion Illustration" />
          </div>
        </div>
      </div>
    </section>
  );
};

export default Hero;
