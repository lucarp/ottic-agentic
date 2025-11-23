import { useEffect, useRef, useState } from 'react';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import type { WebSocketMessage } from '@/types';
import { Send } from 'lucide-react';

interface TerminalProps {
  messages: WebSocketMessage[];
  isConnected: boolean;
  onSendMessage: (content: string) => void;
  onSendContinue: (originalInput: string, responseId: string) => void;
}

export const Terminal = ({ messages, isConnected, onSendMessage, onSendContinue }: TerminalProps) => {
  const [input, setInput] = useState('');
  const [lastUserInput, setLastUserInput] = useState('');
  const [pendingResponseId, setPendingResponseId] = useState<string | null>(null);
  const [currentStreamingText, setCurrentStreamingText] = useState('');
  const [currentStreamingReasoning, setCurrentStreamingReasoning] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, currentStreamingText, currentStreamingReasoning]);

  // Process streaming messages
  useEffect(() => {
    const lastMessage = messages[messages.length - 1];

    if (lastMessage?.type === 'text_delta') {
      setCurrentStreamingText(prev => prev + lastMessage.delta);
    } else if (lastMessage?.type === 'text_done') {
      setCurrentStreamingText('');
    } else if (lastMessage?.type === 'reasoning_delta') {
      setCurrentStreamingReasoning(prev => prev + lastMessage.delta);
    } else if (lastMessage?.type === 'reasoning') {
      setCurrentStreamingReasoning('');
    } else if (lastMessage?.type === 'max_turns_exceeded') {
      setPendingResponseId(lastMessage.response_id);
    }
  }, [messages]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || !isConnected) return;

    setLastUserInput(input);
    onSendMessage(input);
    setInput('');
    setCurrentStreamingText('');
    setCurrentStreamingReasoning('');
  };

  const handleContinue = () => {
    if (pendingResponseId && lastUserInput) {
      onSendContinue(lastUserInput, pendingResponseId);
      setPendingResponseId(null);
    }
  };

  const renderMessage = (message: WebSocketMessage, index: number) => {
    switch (message.type) {
      case 'user_message':
        return (
          <div key={index} className="flex justify-end mb-4">
            <div className="bg-primary text-primary-foreground rounded-lg px-4 py-2 max-w-[70%]">
              {message.content}
            </div>
          </div>
        );

      case 'assistant_message':
        return (
          <div key={index} className="flex justify-start mb-4">
            <div className="bg-secondary text-secondary-foreground rounded-lg px-4 py-2 max-w-[85%]">
              {message.content}
            </div>
          </div>
        );

      case 'reasoning':
        return (
          <div key={index} className="flex justify-start mb-4">
            <div className="bg-muted/50 text-muted-foreground rounded-lg px-4 py-2 max-w-[85%] border-l-4 border-purple-500 italic">
              <span className="text-purple-400">💭 Thinking:</span> {message.content}
            </div>
          </div>
        );

      case 'tool_execution':
        return (
          <div key={index} className="flex justify-start mb-4">
            <div className={`bg-muted/30 rounded-lg px-4 py-2 max-w-[85%] border-l-4 font-mono text-sm ${
              message.status === 'completed' ? 'border-green-500' :
              message.status === 'failed' ? 'border-red-500' :
              'border-blue-500'
            }`}>
              <div className="flex items-center gap-2 mb-1">
                <span>🔧</span>
                <span className="font-semibold">{message.tool_name}</span>
                <Badge variant={message.status === 'completed' ? 'default' : 'destructive'}>
                  {message.status}
                </Badge>
              </div>
              {message.output && (
                <pre className="text-xs mt-2 overflow-x-auto">
                  {JSON.stringify(message.output, null, 2)}
                </pre>
              )}
            </div>
          </div>
        );

      case 'error':
        return (
          <div key={index} className="flex justify-start mb-4">
            <div className="bg-destructive/20 text-destructive rounded-lg px-4 py-2 max-w-[85%] border-l-4 border-destructive">
              <span className="font-semibold">❌ Error:</span> {message.error}
            </div>
          </div>
        );

      case 'max_turns_exceeded':
        return (
          <div key={index} className="flex justify-start mb-4">
            <div className="bg-yellow-950/50 text-yellow-200 rounded-lg px-4 py-3 max-w-[85%] border-l-4 border-yellow-500">
              <p className="mb-3">{message.message}</p>
              <Button
                onClick={handleContinue}
                size="sm"
                className="bg-green-600 hover:bg-green-700"
              >
                Continue
              </Button>
            </div>
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="border-b border-border p-4 flex items-center justify-between">
        <h2 className="text-lg font-semibold">Terminal</h2>
        <div className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`} />
          <span className="text-sm text-muted-foreground">
            {isConnected ? 'Connected' : 'Disconnected'}
          </span>
        </div>
      </div>

      {/* Messages */}
      <ScrollArea className="flex-1 p-4">
        <div className="space-y-2">
          {messages.map((message, index) => renderMessage(message, index))}

          {/* Streaming reasoning */}
          {currentStreamingReasoning && (
            <div className="flex justify-start mb-4">
              <div className="bg-muted/50 text-muted-foreground rounded-lg px-4 py-2 max-w-[85%] border-l-4 border-purple-500 italic">
                <span className="text-purple-400">💭 Thinking:</span> {currentStreamingReasoning}
              </div>
            </div>
          )}

          {/* Streaming text */}
          {currentStreamingText && (
            <div className="flex justify-start mb-4">
              <div className="bg-secondary text-secondary-foreground rounded-lg px-4 py-2 max-w-[85%]">
                {currentStreamingText}
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </ScrollArea>

      {/* Input */}
      <div className="border-t border-border p-4">
        <form onSubmit={handleSubmit} className="flex gap-2">
          <Input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type your message..."
            disabled={!isConnected}
            className="flex-1"
          />
          <Button type="submit" disabled={!isConnected || !input.trim()}>
            <Send className="w-4 h-4" />
          </Button>
        </form>
      </div>
    </div>
  );
};
