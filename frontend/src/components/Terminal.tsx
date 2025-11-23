import { useEffect, useRef, useState } from 'react';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Textarea } from '@/components/ui/textarea';
import { Button } from '@/components/ui/button';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import type { WebSocketMessage } from '@/types';
import {
  Terminal as TerminalIcon,
  Copy,
  ThumbsUp,
  ThumbsDown,
  Check,
  ChevronLeft,
  ChevronRight,
  ChevronDown,
  Loader2,
} from 'lucide-react';

interface TerminalProps {
  messages: WebSocketMessage[];
  isConnected: boolean;
  onSendMessage: (content: string) => void;
  onSendContinue: (originalInput: string, responseId: string) => void;
  onCollapseChange?: (collapsed: boolean) => void;
  onWidthChange?: (width: number) => void;
}

interface MessageFeedback {
  [messageId: string]: 'up' | 'down' | null;
}

interface ToolCollapsedState {
  [messageId: string]: boolean;
}

// Welcome quotes
const WELCOME_QUOTES = [
  { text: "Simplicity is the ultimate sophistication.", author: "Leonardo da Vinci" },
  { text: "In the middle of difficulty lies opportunity.", author: "Albert Einstein" },
  { text: "The unexamined life is not worth living.", author: "Socrates" },
  { text: "We are what we repeatedly do. Excellence, then, is not an act, but a habit.", author: "Aristotle" },
  { text: "Imagination is more important than knowledge.", author: "Albert Einstein" },
  { text: "Be curious, not judgmental.", author: "Walt Whitman" },
  { text: "The best way to predict the future is to create it.", author: "Peter Drucker" },
  { text: "Do not go where the path may lead, go instead where there is no path and leave a trail.", author: "Ralph Waldo Emerson" },
];

// Thinking animation phrases
const THINKING_VERBS = [
  'Crystallizing', 'Mapping', 'Orchestrating', 'Triangulating', 'Excavating',
  'Synthesizing', 'Architecting', 'Curating', 'Harmonizing', 'Choreographing',
  'Weaving', 'Sculpting', 'Distilling', 'Navigating', 'Illuminating',
];

const THINKING_NOUNS = [
  'insights', 'territories', 'campaigns', 'data points', 'patterns',
  'narratives', 'journeys', 'moments', 'channels', 'touchpoints',
  'stories', 'positioning', 'essence', 'markets', 'landscapes',
];

export const Terminal = ({ messages, isConnected, onSendMessage, onSendContinue, onCollapseChange, onWidthChange }: TerminalProps) => {
  const [input, setInput] = useState('');
  const [lastUserInput, setLastUserInput] = useState('');
  const [pendingResponseId, setPendingResponseId] = useState<string | null>(null);
  const [currentStreamingText, setCurrentStreamingText] = useState('');
  const [currentStreamingReasoning, setCurrentStreamingReasoning] = useState('');
  const [commandHistory, setCommandHistory] = useState<string[]>([]);
  const [historyIndex, setHistoryIndex] = useState(-1);
  const [feedback, setFeedback] = useState<MessageFeedback>({});
  const [collapsedTools, setCollapsedTools] = useState<ToolCollapsedState>({});
  const [copiedMessageId, setCopiedMessageId] = useState<string | null>(null);
  const [isCollapsed, setIsCollapsed] = useState(true);
  const [sidebarWidth, setSidebarWidth] = useState(500);

  // Notify parent of collapse/width changes
  useEffect(() => {
    onCollapseChange?.(isCollapsed);
  }, [isCollapsed, onCollapseChange]);

  useEffect(() => {
    onWidthChange?.(sidebarWidth);
  }, [sidebarWidth, onWidthChange]);
  const [isResizing, setIsResizing] = useState(false);
  const [thinkingMessage, setThinkingMessage] = useState('');
  const [animationFrame, setAnimationFrame] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const resizeStartRef = useRef({ x: 0, width: 0 });

  // Auto-scroll to bottom
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, currentStreamingText, currentStreamingReasoning]);

  // Auto-focus textarea on mount
  useEffect(() => {
    setTimeout(() => textareaRef.current?.focus(), 100);
  }, []);

  // Thinking animation
  useEffect(() => {
    if (!isProcessing) {
      setThinkingMessage('');
      setAnimationFrame(0);
      return;
    }

    const randomVerb = THINKING_VERBS[Math.floor(Math.random() * THINKING_VERBS.length)];
    const randomNoun = THINKING_NOUNS[Math.floor(Math.random() * THINKING_NOUNS.length)];
    setThinkingMessage(`${randomVerb} ${randomNoun}`);

    const messageInterval = setInterval(() => {
      const verb = THINKING_VERBS[Math.floor(Math.random() * THINKING_VERBS.length)];
      const noun = THINKING_NOUNS[Math.floor(Math.random() * THINKING_NOUNS.length)];
      setThinkingMessage(`${verb} ${noun}`);
    }, Math.random() * 3000 + 2500);

    const frameInterval = setInterval(() => {
      setAnimationFrame(prev => (prev + 1) % 16);
    }, 120);

    return () => {
      clearInterval(messageInterval);
      clearInterval(frameInterval);
    };
  }, [isProcessing]);

  // Process streaming messages
  useEffect(() => {
    const lastMessage = messages[messages.length - 1];

    if (lastMessage?.type === 'text_delta') {
      setCurrentStreamingText(prev => prev + lastMessage.delta);
      setIsProcessing(true);
    } else if (lastMessage?.type === 'text_done') {
      setCurrentStreamingText('');
      setIsProcessing(false);
    } else if (lastMessage?.type === 'reasoning_delta') {
      setCurrentStreamingReasoning(prev => prev + lastMessage.delta);
      setIsProcessing(true);
    } else if (lastMessage?.type === 'reasoning') {
      setCurrentStreamingReasoning('');
      setIsProcessing(false);
    } else if (lastMessage?.type === 'max_turns_exceeded') {
      setPendingResponseId(lastMessage.response_id);
      setIsProcessing(false);
    } else if (lastMessage?.type === 'tool_execution' && lastMessage.status === 'started') {
      setIsProcessing(true);
    } else if (lastMessage?.type === 'tool_execution' && lastMessage.status === 'completed') {
      setIsProcessing(false);
    }
  }, [messages]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Cmd/Ctrl + K to toggle sidebar
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        setIsCollapsed(prev => !prev);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  // Resize handlers
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (isResizing) {
        const deltaX = e.clientX - resizeStartRef.current.x;
        const newWidth = Math.max(300, Math.min(resizeStartRef.current.width + deltaX, window.innerWidth * 0.8));
        setSidebarWidth(newWidth);
      }
    };

    const handleMouseUp = () => {
      setIsResizing(false);
    };

    if (isResizing) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
      return () => {
        document.removeEventListener('mousemove', handleMouseMove);
        document.removeEventListener('mouseup', handleMouseUp);
      };
    }
  }, [isResizing]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || !isConnected) return;

    const userInput = input.trim();
    setLastUserInput(userInput);
    setCommandHistory(prev => [...prev, userInput]);
    setHistoryIndex(-1);
    onSendMessage(userInput);
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

  const handleCopy = async (content: string, messageId: string) => {
    try {
      await navigator.clipboard.writeText(content);
      setCopiedMessageId(messageId);
      setTimeout(() => setCopiedMessageId(null), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  const handleFeedback = (messageId: string, type: 'up' | 'down') => {
    setFeedback(prev => ({
      ...prev,
      [messageId]: prev[messageId] === type ? null : type,
    }));
  };

  const toggleToolCollapse = (messageId: string) => {
    setCollapsedTools(prev => ({
      ...prev,
      [messageId]: !prev[messageId],
    }));
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!isProcessing && input.trim()) {
        handleSubmit(e);
      }
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      if (commandHistory.length > 0 && historyIndex < commandHistory.length - 1) {
        const newIndex = historyIndex + 1;
        setHistoryIndex(newIndex);
        setInput(commandHistory[commandHistory.length - 1 - newIndex]);
      }
    } else if (e.key === 'ArrowDown') {
      e.preventDefault();
      if (historyIndex > 0) {
        const newIndex = historyIndex - 1;
        setHistoryIndex(newIndex);
        setInput(commandHistory[commandHistory.length - 1 - newIndex]);
      } else if (historyIndex === 0) {
        setHistoryIndex(-1);
        setInput('');
      }
    }
  };

  const renderThinkingAnimation = () => {
    const shapes = ['◐', '◓', '◑', '◒', '◈', '◇', '◆', '●', '◉', '○', '◎', '▣', '■', '▪', '◘', '○'];
    return shapes[animationFrame % shapes.length];
  };

  const renderMessage = (message: WebSocketMessage, index: number) => {
    const messageId = `msg-${index}`;

    switch (message.type) {
      case 'user_message':
        return (
          <div key={index} className="flex gap-2 mb-3">
            <span className="text-primary font-mono">{'>'}</span>
            <div className="bg-primary text-primary-foreground px-3 py-1 inline-block font-mono text-sm">
              {message.content}
            </div>
          </div>
        );

      case 'assistant_message':
        return (
          <div key={index} className="pl-6 mb-4">
            <div className="text-accent whitespace-pre-wrap font-mono text-sm leading-relaxed">
              {message.content}
            </div>

            <TooltipProvider>
              <div className="flex gap-2 mt-3">
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-7 text-secondary hover:text-primary hover:bg-muted"
                      onClick={() => handleCopy(message.content, messageId)}
                    >
                      {copiedMessageId === messageId ? (
                        <Check className="h-3 w-3" />
                      ) : (
                        <Copy className="h-3 w-3" />
                      )}
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>
                    {copiedMessageId === messageId ? 'Copied!' : 'Copy response'}
                  </TooltipContent>
                </Tooltip>

                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-7 text-secondary hover:text-primary hover:bg-muted"
                      onClick={() => handleFeedback(messageId, 'up')}
                    >
                      <ThumbsUp className={`h-3 w-3 ${feedback[messageId] === 'up' ? 'fill-primary' : ''}`} />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>Good response</TooltipContent>
                </Tooltip>

                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-7 text-secondary hover:text-destructive hover:bg-muted"
                      onClick={() => handleFeedback(messageId, 'down')}
                    >
                      <ThumbsDown className={`h-3 w-3 ${feedback[messageId] === 'down' ? 'fill-destructive' : ''}`} />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>Poor response</TooltipContent>
                </Tooltip>
              </div>
            </TooltipProvider>
          </div>
        );

      case 'reasoning':
        return (
          <div key={index} className="pl-6 mb-3">
            <div className="bg-muted/50 text-muted-foreground rounded px-3 py-2 border-l-4 border-purple-500 italic font-mono text-xs">
              <span className="text-purple-400">💭 {message.content}</span>
            </div>
          </div>
        );

      case 'tool_execution':
        return (
          <div key={index} className="pl-6 mb-3">
            <Collapsible
              open={!collapsedTools[messageId]}
              onOpenChange={() => toggleToolCollapse(messageId)}
              className="bg-card border-2 border-border border-l-4 border-l-primary rounded overflow-hidden"
            >
              <CollapsibleTrigger className="w-full p-3 hover:bg-muted/50 transition-colors flex items-center gap-2">
                {message.status === 'completed' && <Check className="h-4 w-4 text-primary" />}
                {message.status === 'started' && <Loader2 className="h-4 w-4 text-primary animate-spin" />}
                {message.status === 'failed' && <span className="text-destructive">✗</span>}

                <span className="font-mono text-sm text-primary font-semibold flex-1 text-left">
                  {message.tool_name}
                </span>

                <Badge variant={message.status === 'completed' ? 'default' : 'secondary'} className="text-xs">
                  {message.status}
                </Badge>

                {message.status === 'completed' && (
                  <ChevronDown className={`h-4 w-4 text-primary transition-transform ${collapsedTools[messageId] ? 'rotate-180' : ''}`} />
                )}
              </CollapsibleTrigger>

              {message.status === 'completed' && (
                <CollapsibleContent className="border-t border-border/50 p-3 bg-muted/20">
                  {message.output && (
                    <pre className="text-xs font-mono text-accent overflow-x-auto whitespace-pre-wrap">
                      {JSON.stringify(message.output, null, 2)}
                    </pre>
                  )}
                </CollapsibleContent>
              )}
            </Collapsible>
          </div>
        );

      case 'error':
        return (
          <div key={index} className="pl-6 mb-3">
            <div className="bg-destructive/20 text-destructive rounded px-3 py-2 border-l-4 border-destructive font-mono text-sm">
              <span className="font-semibold">❌ Error:</span> {message.error}
            </div>
          </div>
        );

      case 'max_turns_exceeded':
        return (
          <div key={index} className="pl-6 mb-3">
            <div className="bg-yellow-950/50 text-yellow-200 rounded px-3 py-3 border-l-4 border-yellow-500 font-mono text-sm">
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

  // Welcome message
  const welcomeQuote = WELCOME_QUOTES[Math.floor(Math.random() * WELCOME_QUOTES.length)];

  return (
    <>
      {/* Collapsed Sidebar Bar */}
      {isCollapsed && (
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <div
                onClick={() => setIsCollapsed(false)}
                className="fixed left-0 top-14 h-[calc(100vh-3.5rem)] w-12 bg-[#0d1b13] border-r-2 border-[#06ff83]/30 flex flex-col items-center justify-center cursor-pointer hover:bg-[#122319] hover:border-[#06ff83]/50 transition-all z-50 group gap-2"
              >
                <ChevronRight className="h-5 w-5 text-[#06ff83] group-hover:translate-x-0.5 transition-transform" />
                <div className="writing-mode-vertical text-[#06ff83] font-mono text-xs font-semibold tracking-wider transition-colors">
                  gtm_terminal:~$
                </div>
              </div>
            </TooltipTrigger>
            <TooltipContent side="right" className="font-mono bg-[#0d1b13] border-[#06ff83]/30">
              <p className="font-semibold text-[#06ff83]">Expand Terminal</p>
              <p className="text-xs opacity-90 text-[#06ff83]/70">Cmd+K</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      )}

      {/* Main Terminal Sidebar */}
      {!isCollapsed && (
        <div
          className="fixed left-0 top-14 h-[calc(100vh-3.5rem)] bg-background border-r-4 border-primary flex flex-col z-40 transition-all shadow-xl"
          style={{ width: sidebarWidth }}
        >
          {/* Resize Handle */}
          <div
            onMouseDown={(e) => {
              setIsResizing(true);
              resizeStartRef.current = { x: e.clientX, width: sidebarWidth };
            }}
            className="absolute top-0 right-0 bottom-0 w-1.5 cursor-ew-resize hover:bg-primary/50 active:bg-primary transition-colors"
          />

          {/* Header */}
          <div className="p-4 bg-muted border-b-2 border-primary flex items-center gap-2">
            <TerminalIcon className="h-5 w-5 text-primary" />
            <span className="font-mono text-sm text-primary font-semibold tracking-wide flex-1">
              gtm_terminal:~$
            </span>

            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-7 w-7 p-0 text-secondary hover:text-primary"
                    onClick={() => setIsCollapsed(true)}
                  >
                    <ChevronLeft className="h-4 w-4" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  <p>Collapse sidebar</p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </div>

          {/* Messages Area */}
          <ScrollArea className="flex-1 p-4">
            {/* Welcome Message */}
            {messages.length === 0 && (
              <div className="text-secondary font-mono text-xs leading-relaxed whitespace-pre-wrap mb-6">
{`★  . . WELCOME BACK 🏆  ˙    . .★  . .
. .★  . . ✰   ֹ  . .★  . . ⋆ ֹ   .    ͘ ͘
✨ "${welcomeQuote.text}" — ${welcomeQuote.author} ✨
.    .     .    .   .  ★  . ∙  ݀.  ݀  ★      ◌ .

┌─ Ottic Agentic Terminal ────────────────────┐
│ Type your message to interact with AI
│ Press ↑↓ to navigate command history
└──────────────────────────────────────────────┘`}
              </div>
            )}

            {messages.map((message, index) => renderMessage(message, index))}

            {/* Streaming reasoning */}
            {currentStreamingReasoning && (
              <div className="pl-6 mb-3">
                <div className="bg-muted/50 text-muted-foreground rounded px-3 py-2 border-l-4 border-purple-500 italic font-mono text-xs">
                  <span className="text-purple-400">💭 {currentStreamingReasoning}</span>
                </div>
              </div>
            )}

            {/* Streaming text */}
            {currentStreamingText && (
              <div className="pl-6 mb-4">
                <div className="text-accent whitespace-pre-wrap font-mono text-sm leading-relaxed">
                  {currentStreamingText}
                  <span className="inline-block w-2 h-4 bg-primary ml-1 animate-pulse" />
                </div>
              </div>
            )}

            {/* Thinking indicator */}
            {isProcessing && thinkingMessage && (
              <div className="pl-6 mb-3 flex items-center gap-2">
                <span className="text-primary font-mono text-lg font-bold">
                  {renderThinkingAnimation()}
                </span>
                <span className="text-secondary font-mono text-sm">
                  {thinkingMessage}…
                  <span className="text-muted-foreground text-xs ml-2">(esc to interrupt)</span>
                </span>
              </div>
            )}

            <div ref={messagesEndRef} />
          </ScrollArea>

          {/* Input Area */}
          <div className="p-4 bg-muted border-t-2 border-primary">
            <form onSubmit={handleSubmit} className="flex gap-2 items-start">
              <span className="text-primary font-mono text-sm pt-2">{'>'}</span>
              <Textarea
                ref={textareaRef}
                value={input}
                onChange={(e) => {
                  setInput(e.target.value);
                  // Auto-resize
                  const target = e.target as HTMLTextAreaElement;
                  target.style.height = 'auto';
                  target.style.height = Math.min(target.scrollHeight, 150) + 'px';
                }}
                onKeyDown={handleKeyDown}
                placeholder="Type a command or ask naturally..."
                disabled={!isConnected}
                className="flex-1 min-h-[38px] max-h-[150px] bg-background border-2 border-primary text-accent font-mono text-sm resize-none focus-visible:ring-primary"
                rows={1}
                style={{
                  height: 'auto',
                  minHeight: '38px',
                }}
              />
            </form>

            <div className="text-muted-foreground font-mono text-xs mt-2 pl-6">
              {isProcessing
                ? 'Press ESC to interrupt execution'
                : 'Press ↑↓ for history • Shift+Enter for new line'}
            </div>
          </div>
        </div>
      )}
    </>
  );
};
