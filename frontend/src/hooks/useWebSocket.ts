import { useState, useEffect, useCallback, useRef } from 'react';
import type { WebSocketMessage, Artifact } from '@/types';

interface WebSocketState {
  isConnected: boolean;
  messages: WebSocketMessage[];
  artifacts: Artifact[];
  sendMessage: (content: string) => void;
  sendContinue: (originalInput: string, responseId: string) => void;
  cancelProcessing: () => void;
}

const WS_URL = 'ws://localhost:8000/ws';
const RECONNECT_DELAY = 3000;
const MAX_RECONNECT_ATTEMPTS = 10;

export const useWebSocket = (): WebSocketState => {
  const [isConnected, setIsConnected] = useState(false);
  const [messages, setMessages] = useState<WebSocketMessage[]>([]);
  const [artifacts, setArtifacts] = useState<Artifact[]>([]);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const reconnectAttemptsRef = useRef(0);
  const isConnectingRef = useRef(false);

  const connect = useCallback(() => {
    // Prevent multiple simultaneous connection attempts
    if (isConnectingRef.current || wsRef.current?.readyState === WebSocket.OPEN) {
      console.log('⏸️ Connection attempt skipped (already connecting or connected)');
      return;
    }

    try {
      isConnectingRef.current = true;
      console.log('🔌 Attempting to connect to WebSocket:', WS_URL);
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log('✅ WebSocket connected successfully');
        setIsConnected(true);
        reconnectAttemptsRef.current = 0; // Reset counter on successful connection
        isConnectingRef.current = false;
      };

      ws.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data);
          console.log('📨 Received message:', message.type);

          // Add message to history
          setMessages((prev) => [...prev, message]);

          // Handle artifact creation
          if (message.type === 'artifact_created') {
            setArtifacts((prev) => [...prev, message.artifact]);
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      ws.onerror = (error) => {
        console.error('❌ WebSocket error:', error);
        console.error('❌ Connection failed to:', WS_URL);
      };

      ws.onclose = (event) => {
        console.log('🔌 WebSocket disconnected. Code:', event.code, 'Reason:', event.reason);
        setIsConnected(false);
        wsRef.current = null;
        isConnectingRef.current = false;

        // Attempt to reconnect with limit
        reconnectAttemptsRef.current += 1;
        if (reconnectAttemptsRef.current <= MAX_RECONNECT_ATTEMPTS) {
          console.log(`⏳ Reconnect attempt ${reconnectAttemptsRef.current}/${MAX_RECONNECT_ATTEMPTS} in ${RECONNECT_DELAY/1000}s...`);
          reconnectTimeoutRef.current = setTimeout(() => {
            connect();
          }, RECONNECT_DELAY);
        } else {
          console.error(`❌ Max reconnection attempts (${MAX_RECONNECT_ATTEMPTS}) reached. Giving up.`);
        }
      };
    } catch (error) {
      console.error('Error creating WebSocket connection:', error);
      isConnectingRef.current = false;
    }
  }, []);

  const sendMessage = useCallback((content: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      const message: WebSocketMessage = {
        type: 'user_message',
        content,
      };
      wsRef.current.send(JSON.stringify(message));
    } else {
      console.error('WebSocket is not connected');
    }
  }, []);

  const sendContinue = useCallback((originalInput: string, responseId: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      const message: WebSocketMessage = {
        type: 'continue_execution',
        content: originalInput,
        previous_response_id: responseId,
      };
      wsRef.current.send(JSON.stringify(message));
    } else {
      console.error('WebSocket is not connected');
    }
  }, []);

  const cancelProcessing = useCallback(() => {
    console.log('⚠️ Canceling processing - closing and reconnecting WebSocket');
    if (wsRef.current) {
      // Close the current connection (will trigger reconnect)
      wsRef.current.close();
      wsRef.current = null;
    }
    // Clear messages to stop showing processing state
    setMessages((prev) => {
      // Keep only non-streaming messages
      const filtered = prev.filter(
        (msg) => !['text_delta', 'reasoning_delta', 'tool_execution'].includes(msg.type)
      );
      return filtered;
    });
    // Force reconnect
    setTimeout(() => {
      connect();
    }, 100);
  }, [connect]);

  useEffect(() => {
    // Only connect once on mount
    connect();

    return () => {
      console.log('🧹 Cleaning up WebSocket connection');
      isConnectingRef.current = false;
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current?.readyState === WebSocket.OPEN || wsRef.current?.readyState === WebSocket.CONNECTING) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, []); // Empty deps - only run once on mount

  return {
    isConnected,
    messages,
    artifacts,
    sendMessage,
    sendContinue,
    cancelProcessing,
  };
};
