"""
ZeroMQ IPC Module - Publishers.
Publishes signals and metrics to Tauri UI via ZeroMQ PUB/SUB pattern.
Uses msgpack for efficient binary serialization.
"""

import zmq
import zmq.asyncio
import msgpack
import asyncio
from typing import Dict, Any, Optional
import logging
import time

logger = logging.getLogger(__name__)


class SignalPublisher:
    """
    ZeroMQ publisher for signals and metrics.
    Publishes to multiple topics for UI subscription.
    """
    
    def __init__(
        self,
        addr: str = "tcp://127.0.0.1:5555",
        hwm: int = 1000
    ):
        """
        Args:
            addr: ZeroMQ bind address
            hwm: High water mark for message buffering
        """
        self.addr = addr
        self.hwm = hwm
        
        self._context: Optional[zmq.asyncio.Context] = None
        self._socket: Optional[zmq.asyncio.Socket] = None
        self._running = False
        
    async def start(self):
        """Start the publisher."""
        self._context = zmq.asyncio.Context()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.setsockopt(zmq.SNDHWM, self.hwm)
        self._socket.bind(self.addr)
        
        self._running = True
        logger.info(f"SignalPublisher bound to {self.addr}")
        
    async def stop(self):
        """Stop the publisher."""
        self._running = False
        
        if self._socket:
            self._socket.close()
        if self._context:
            self._context.term()
            
        logger.info("SignalPublisher stopped")
        
    async def publish_signal(
        self,
        timeframe: str,
        signal_data: Dict[str, Any]
    ):
        """
        Publish signal to timeframe-specific topic.
        
        Args:
            timeframe: "5m" or "15m"
            signal_data: Signal dictionary with ts, dir, S_t, p_t, θ*, status
        """
        if not self._running or not self._socket:
            return
            
        topic = f"signals_{timeframe}"
        
        # Add timestamp if not present
        if 'ts' not in signal_data:
            signal_data['ts'] = time.time()
            
        # Pack with msgpack
        packed = msgpack.packb(signal_data, use_bin_type=True)
        
        # Send as multipart: topic + data
        await self._socket.send_multipart([
            topic.encode('utf-8'),
            packed
        ])
        
    async def publish_metrics(
        self,
        timeframe: str,
        metrics: Dict[str, Any]
    ):
        """
        Publish metrics to timeframe-specific topic.
        
        Args:
            timeframe: "5m" or "15m"
            metrics: Metrics dictionary with win_rate, RR, E_R, drawdown, sharpe_rolling
        """
        if not self._running or not self._socket:
            return
            
        topic = f"metrics_{timeframe}"
        
        # Add timestamp
        metrics['ts'] = time.time()
        
        # Pack with msgpack
        packed = msgpack.packb(metrics, use_bin_type=True)
        
        await self._socket.send_multipart([
            topic.encode('utf-8'),
            packed
        ])
        
    async def publish_history(
        self,
        trade_data: Dict[str, Any]
    ):
        """
        Publish trade to history topic.
        
        Args:
            trade_data: Trade dictionary with ts, tf, dir, entry, exit, result, pnl
        """
        if not self._running or not self._socket:
            return
            
        topic = "history"
        
        # Add timestamp if not present
        if 'ts' not in trade_data:
            trade_data['ts'] = time.time()
            
        # Pack with msgpack
        packed = msgpack.packb(trade_data, use_bin_type=True)
        
        await self._socket.send_multipart([
            topic.encode('utf-8'),
            packed
        ])
        
    async def publish_status(
        self,
        status_data: Dict[str, Any]
    ):
        """
        Publish system status.
        
        Args:
            status_data: Status dictionary
        """
        if not self._running or not self._socket:
            return
            
        topic = "status"
        
        status_data['ts'] = time.time()
        packed = msgpack.packb(status_data, use_bin_type=True)
        
        await self._socket.send_multipart([
            topic.encode('utf-8'),
            packed
        ])


class CommandResponder:
    """
    ZeroMQ REP socket for receiving commands from UI.
    Implements REQ/REP pattern for control messages.
    """
    
    def __init__(
        self,
        addr: str = "tcp://127.0.0.1:5556",
        on_command: Optional[callable] = None
    ):
        """
        Args:
            addr: ZeroMQ bind address
            on_command: Callback function for handling commands
        """
        self.addr = addr
        self.on_command = on_command
        
        self._context: Optional[zmq.asyncio.Context] = None
        self._socket: Optional[zmq.asyncio.Socket] = None
        self._running = False
        
    async def start(self):
        """Start the command responder."""
        self._context = zmq.asyncio.Context()
        self._socket = self._context.socket(zmq.REP)
        self._socket.bind(self.addr)
        
        self._running = True
        logger.info(f"CommandResponder bound to {self.addr}")
        
        # Start listening loop
        asyncio.create_task(self._listen_loop())
        
    async def stop(self):
        """Stop the command responder."""
        self._running = False
        
        if self._socket:
            self._socket.close()
        if self._context:
            self._context.term()
            
        logger.info("CommandResponder stopped")
        
    async def _listen_loop(self):
        """Listen for incoming commands."""
        while self._running:
            try:
                # Receive command
                message = await self._socket.recv()
                command = msgpack.unpackb(message, raw=False)
                
                logger.debug(f"Received command: {command}")
                
                # Process command
                response = await self._process_command(command)
                
                # Send response
                response_packed = msgpack.packb(response, use_bin_type=True)
                await self._socket.send(response_packed)
                
            except Exception as e:
                logger.error(f"Command processing error: {e}")
                
                # Send error response
                error_response = {
                    'status': 'error',
                    'message': str(e)
                }
                await self._socket.send(
                    msgpack.packb(error_response, use_bin_type=True)
                )
                
    async def _process_command(self, command: Dict) -> Dict:
        """
        Process incoming command.
        
        Expected commands:
        - {"action": "start", "timeframe": "5m"}
        - {"action": "stop", "timeframe": "5m"}
        - {"action": "set_theta", "timeframe": "5m", "value": 0.3}
        - {"action": "get_status"}
        - {"action": "export_history", "format": "csv"}
        """
        action = command.get('action')
        
        if self.on_command:
            try:
                result = await self.on_command(command)
                return {
                    'status': 'ok',
                    'action': action,
                    'result': result
                }
            except Exception as e:
                return {
                    'status': 'error',
                    'action': action,
                    'message': str(e)
                }
        else:
            # Default handler - just acknowledge
            return {
                'status': 'ok',
                'action': action,
                'message': 'Command received (no handler)'
            }


async def create_pubsub_system(
    pub_addr: str = "tcp://127.0.0.1:5555",
    rep_addr: str = "tcp://127.0.0.1:5556",
    on_command: Optional[callable] = None
) -> tuple:
    """
    Factory function to create publisher and responder.
    
    Returns:
        (SignalPublisher, CommandResponder)
    """
    publisher = SignalPublisher(addr=pub_addr)
    responder = CommandResponder(addr=rep_addr, on_command=on_command)
    
    await publisher.start()
    await responder.start()
    
    return publisher, responder


if __name__ == "__main__":
    async def test_handler(command):
        print(f"Handling command: {command}")
        return {'handled': True}
    
    async def main():
        pub, resp = await create_pubsub_system(on_command=test_handler)
        
        # Test publishing
        await pub.publish_signal("5m", {
            'dir': 1,
            'S_t': 0.85,
            'p_t': 0.72,
            'θ*': 0.3,
            'status': 'SIGNAL'
        })
        
        await pub.publish_metrics("5m", {
            'win_rate': 0.32,
            'RR': 2.8,
            'E_R': 0.0011,
            'drawdown': -0.02,
            'sharpe_rolling': 1.2
        })
        
        print("Test messages sent. Running for 5 seconds...")
        await asyncio.sleep(5)
        
        await pub.stop()
        await resp.stop()
        
    asyncio.run(main())
