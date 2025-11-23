import { useWebSocket } from './hooks/useWebSocket';
import { Terminal } from './components/Terminal';
import { ArtifactViewer } from './components/ArtifactViewer';

function App() {
  const { isConnected, messages, artifacts, sendMessage, sendContinue } = useWebSocket();

  return (
    <div className="h-screen w-screen flex overflow-hidden bg-background text-foreground">
      {/* Terminal Column - 60% */}
      <div className="w-[60%] border-r border-border flex flex-col">
        <Terminal
          messages={messages}
          isConnected={isConnected}
          onSendMessage={sendMessage}
          onSendContinue={sendContinue}
        />
      </div>

      {/* Artifact Column - 40% */}
      <div className="w-[40%] flex flex-col">
        <ArtifactViewer artifacts={artifacts} />
      </div>
    </div>
  );
}

export default App;
