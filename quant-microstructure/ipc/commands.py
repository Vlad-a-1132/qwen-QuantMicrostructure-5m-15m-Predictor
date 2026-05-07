"""
ZeroMQ IPC Module - Commands.
REQ/REP protocol for UI control messages.
"""

import zmq
import zmq.asyncio
import msgpack
import asyncio
from typing import Dict, Any, Optional, Callable
import logging

logger = logging.getLogger(__name__)


class CommandClient:
    """
    ZeroMQ REQ client for sending commands to Python core.
    Used by Tauri UI to control workers.
    """
    
    def __init__(self, addr: str = "tcp://127.0.0.1:5556", timeout: float = 5.0):
        """
        Args:
            addr: ZeroMQ connection address
            timeout: Request timeout in seconds
        """
        self.addr = addr
        self.timeout = timeout
        
        self._context: Optional[zmq.asyncio.Context] = None
        self._socket: Optional[zmq.asyncio.Socket] = None
        self._connected = False
        
    async def connect(self):
        """Establish connection to server."""
        self._context = zmq.asyncio.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, int(self.timeout * 1000))
        self._socket.connect(self.addr)
        
        self._connected = True
        logger.info(f"CommandClient connected to {self.addr}")
        
    async def disconnect(self):
        """Close connection."""
        self._connected = False
        
        if self._socket:
            self._socket.close()
        if self._context:
            self._context.term()
            
        logger.info("CommandClient disconnected")
        
    async def send_command(
        self,
        action: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send command and wait for response.
        
        Args:
            action: Command action name
            **kwargs: Additional command parameters
            
        Returns:
            Response dictionary
        """
        if not self._connected:
            raise RuntimeError("Not connected")
            
        command = {
            'action': action,
            **kwargs
        }
        
        # Pack and send
        packed = msgpack.packb(command, use_bin_type=True)
        await self._socket.send(packed)
        
        # Wait for response
        try:
            response_raw = await self._socket.recv()
            response = msgpack.unpackb(response_raw, raw=False)
            return response
        except zmq.error.Again:
            logger.warning(f"Command timeout: {action}")
            return {
                'status': 'error',
                'message': f'Timeout waiting for response to {action}'
            }
            
    # Convenience methods for common commands
    
    async def start_timeframe(self, timeframe: str) -> Dict:
        """Start a specific timeframe worker."""
        return await self.send_command('start', timeframe=timeframe)
        
    async def stop_timeframe(self, timeframe: str) -> Dict:
        """Stop a specific timeframe worker."""
        return await self.send_command('stop', timeframe=timeframe)
        
    async def set_theta(self, timeframe: str, value: float) -> Dict:
        """Set theta threshold for a timeframe."""
        return await self.send_command('set_theta', timeframe=timeframe, value=value)
        
    async def set_risk_params(
        self,
        max_exposure: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit_mult: Optional[float] = None
    ) -> Dict:
        """Update risk parameters."""
        params = {}
        if max_exposure is not None:
            params['max_exposure'] = max_exposure
        if stop_loss is not None:
            params['stop_loss'] = stop_loss
        if take_profit_mult is not None:
            params['take_profit_mult'] = take_profit_mult
            
        return await self.send_command('set_risk', **params)
        
    async def get_status(self) -> Dict:
        """Get current system status."""
        return await self.send_command('get_status')
        
    async def export_history(
        self,
        format: str = 'csv',
        timeframe: Optional[str] = None
    ) -> Dict:
        """Export trade history."""
        return await self.send_command(
            'export_history',
            format=format,
            timeframe=timeframe
        )
        
    async def get_metrics(self, timeframe: Optional[str] = None) -> Dict:
        """Get current metrics."""
        return await self.send_command('get_metrics', timeframe=timeframe)


class SignalSubscriber:
    """
    ZeroMQ SUB client for receiving signals from Python core.
    Used by Tauri UI to display real-time data.
    """
    
    def __init__(
        self,
        addr: str = "tcp://127.0.0.1:5555",
        topics: Optional[list] = None
    ):
        """
        Args:
            addr: ZeroMQ connection address
            topics: List of topics to subscribe to
        """
        self.addr = addr
        self.topics = topics or ['signals_5m', 'signals_15m', 'metrics_5m', 'metrics_15m', 'history']
        
        self._context: Optional[zmq.asyncio.Context] = None
        self._socket: Optional[zmq.asyncio.Socket] = None
        self._running = False
        
        # Callbacks
        self._on_signal: Optional[Callable] = None
        self._on_metrics: Optional[Callable] = None
        self._on_history: Optional[Callable] = None
        
    async def connect(self):
        """Establish connection and subscribe to topics."""
        self._context = zmq.asyncio.Context()
        self._socket = self._context.socket(zmq.SUB)
        
        # Connect
        self._socket.connect(self.addr)
        
        # Subscribe to all topics
        for topic in self.topics:
            self._socket.setsockopt_string(zmq.SUBSCRIBE, topic)
            
        self._running = True
        logger.info(f"SignalSubscriber connected to {addr}, subscribed to {self.topics}")
        
        # Start receive loop
        asyncio.create_task(self._receive_loop())
        
    async def disconnect(self):
        """Close connection."""
        self._running = False
        
        if self._socket:
            self._socket.close()
        if self._context:
            self._context.term()
            
        logger.info("SignalSubscriber disconnected")
        
    def on_signal(self, callback: Callable):
        """Set callback for signal messages."""
        self._on_signal = callback
        
    def on_metrics(self, callback: Callable):
        """Set callback for metrics messages."""
        self._on_metrics = callback
        
    def on_history(self, callback: Callable):
        """Set callback for history messages."""
        self._on_history = callback
        
    async def _receive_loop(self):
        """Receive and dispatch messages."""
        while self._running:
            try:
                # Receive multipart message
                topic, data = await self._socket.recv_multipart()
                topic_str = topic.decode('utf-8')
                
                # Unpack data
                message = msgpack.unpackb(data, raw=False)
                
                # Dispatch based on topic
                if topic_str.startswith('signals_'):
                    if self._on_signal:
                        await self._dispatch(self._on_signal, topic_str, message)
                elif topic_str.startswith('metrics_'):
                    if self._on_metrics:
                        await self._dispatch(self._on_metrics, topic_str, message)
                elif topic_str == 'history':
                    if self._on_history:
                        await self._dispatch(self._on_history, topic_str, message)
                        
            except Exception as e:
                if self._running:
                    logger.error(f"Receive error: {e}")
                await asyncio.sleep(0.1)
                
    async def _dispatch(self, callback: Callable, topic: str, message: Dict):
        """Dispatch message to callback."""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(topic, message)
            else:
                callback(topic, message)
        except Exception as e:
            logger.error(f"Callback error for {topic}: {e}")


async def create_ipc_client(
    pub_addr: str = "tcp://127.0.0.1:5555",
    rep_addr: str = "tcp://127.0.0.1:5556"
) -> tuple:
    """
    Factory function to create command client and signal subscriber.
    
    Returns:
        (CommandClient, SignalSubscriber)
    """
    cmd_client = CommandClient(addr=rep_addr)
    sig_sub = SignalSubscriber(addr=pub_addr)
    
    await cmd_client.connect()
    await sig_sub.connect()
    
    return cmd_client, sig_sub


if __name__ == "__main__":
    async def main():
        # Test client
        cmd, sub = await create_ipc_client()
        
        # Set up callbacks
        def on_sig(topic, msg):
            print(f"SIGNAL [{topic}]: {msg}")
            
        def on_met(topic, msg):
            print(f"METRICS [{topic}]: {msg}")
            
        sub.on_signal(on_sig)
        sub.on_metrics(on_met)
        
        # Send test command
        status = await cmd.get_status()
        print(f"Status: {status}")
        
        # Run for a few seconds
        await asyncio.sleep(5)
        
        await cmd.disconnect()
        await sub.disconnect()
        
    asyncio.run(main())
