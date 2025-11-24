import { useState } from 'react';
import { useWebSocket } from './hooks/useWebSocket';
import { AppHeader } from './components/AppHeader';
import { Terminal } from './components/Terminal';
import { MainContent } from './components/MainContent';
import { ArtifactViewer } from './components/ArtifactViewer';

function App() {
  const { isConnected, messages, artifacts, sendMessage, sendContinue, cancelProcessing } = useWebSocket();
  const [activeTab, setActiveTab] = useState('templates');
  const [terminalCollapsed, setTerminalCollapsed] = useState(true);
  const [terminalWidth, setTerminalWidth] = useState(500);

  // Calculate the left margin for content based on terminal state
  const contentMarginLeft = terminalCollapsed ? 48 : terminalWidth;

  return (
    <div className="h-screen w-screen flex flex-col overflow-hidden bg-slate-50">
      {/* Header */}
      <AppHeader activeTab={activeTab} onTabChange={setActiveTab} />

      {/* Main content area below header */}
      <div className="flex-1 flex overflow-hidden">
        {/* Terminal Sidebar (fixed position, manages its own width) */}
        <Terminal
          messages={messages}
          isConnected={isConnected}
          onSendMessage={sendMessage}
          onSendContinue={sendContinue}
          onCancelProcessing={cancelProcessing}
          onCollapseChange={setTerminalCollapsed}
          onWidthChange={setTerminalWidth}
        />

        {/* Main Content Area - shows different content based on active tab */}
        <div
          className="flex-1 h-full transition-[margin-left] duration-200 ease-out"
          style={{ marginLeft: `${contentMarginLeft}px` }}
        >
          {activeTab === 'preview' && (
            <div className="h-full flex flex-col overflow-hidden bg-white">
              <ArtifactViewer artifacts={artifacts} />
            </div>
          )}

          {activeTab === 'artifacts' && (
            <div className="h-full overflow-auto bg-white">
              <div className="max-w-7xl mx-auto p-8">
                <h2 className="text-2xl font-semibold font-mono mb-4 text-slate-900">Your Artifacts</h2>
                <p className="text-sm text-slate-600 font-mono">
                  Coming soon: Browse and manage all your created artifacts
                </p>
              </div>
            </div>
          )}

          {activeTab === 'templates' && <MainContent />}
        </div>
      </div>
    </div>
  );
}

export default App;
