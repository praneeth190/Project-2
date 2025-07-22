import { Github } from 'lucide-react';

const Navbar = () => {
  return (
    <nav className="navbar">
      <div className="container">
        <div className="navbar-content">
          <a href="/" className="navbar-brand">Voice Vista</a>
          <a 
            href="https://github.com/praneeth190" 
            target="_blank" 
            rel="noopener noreferrer"
            className="github-link"
          >
            <Github size={24} color="#4b5563" />
          </a>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
