"""
Central Actor - aggregates data and coordinates federated learning
"""
from datetime import timedelta, datetime
from thespian.actors import Actor, ActorExitRequest, ChildActorExited
from federated_learning.fl_coordinator import FederatedLearningCoordinator
from crdt.lww_register import LWWRegister
from visualization.plotter import ConsumptionPlotter
import logging
import os
import socket
import threading
import pickle
import struct
from enum import Enum

logger = logging.getLogger(__name__)


class CentralState(Enum):
    """State machine states for central actor"""
    INITIALIZING = "initializing"  # Starting up
    MONITORING = "monitoring"      # Normal monitoring mode
    COORDINATING = "coordinating"  # Coordinating FL training
    EMERGENCY = "emergency"        # Emergency mode (critical consumption detected)
    SHUTDOWN = "shutdown"          # Shutting down


class CentralActor(Actor):
    """Central actor that coordinates households and manages federated learning"""
    
    def __init__(self):
        super().__init__()
        self.actor_id = os.getenv('CENTRAL_ID', f'central_{id(self)}')
        self.households = {}
        self.household_actors = {}  # Store household ActorAddresses
        self.aggregated_data = []
        self.crdt_register = LWWRegister()
        self.fl_coordinator = FederatedLearningCoordinator()
        self.training_rounds = 0
        self.data_count = 0
        self.training_threshold = 20  # Start training after 20 data points
        
        # Peer CentralActors for distributed coordination
        self.peer_centrals = {}  # {central_id: socket_connection}
        self.peer_models = {}    # Cache of models from peer centrals
        self.discovery_interval = 15  # seconds
        
        # Raw TCP P2P communication
        self.p2p_port = int(os.getenv('P2P_PORT', '0'))
        self.p2p_server_socket = None
        self.p2p_server_thread = None
        if self.p2p_port > 0:
            self._start_p2p_server()
        
        # Visualization
        self.plotter = ConsumptionPlotter(output_dir=f"plots/{self.actor_id}")
        self.training_history = []
        self.last_plot_time = datetime.now()
        self.plot_interval = 60  # Generate plots every 60 seconds
        
        # Evaluation metrics tracking
        self.evaluation_metrics = []  # History of aggregated metrics
        self.current_round_metrics = []  # Metrics for current training round
        
        # Consumption control thresholds
        self.max_household_consumption = 0.6  # kWh - max consumption per household
        self.critical_household_consumption = 0.8  # kWh - critical level
        self.household_consumption_tracking = {}  # Track current consumption per household
        
        # State machine
        self.state = CentralState.INITIALIZING
        self.state_history = []
        self.shutdown_requested = False
        self.child_actors = {}  # Track child actors
        
    def receiveMessage(self, message, sender):
        """Handle incoming messages"""
        # Handle Thespian system messages
        from thespian.actors import WakeupMessage
        
        # Lifecycle events
        if isinstance(message, ActorExitRequest):
            self._handle_exit_request()
            return
        
        if isinstance(message, ChildActorExited):
            self._handle_child_exit(message)
            return
        
        # Handle application messages (must be dict)
        if not isinstance(message, dict):
            logger.warning(f"[{self.actor_id}] Received unknown message type: {type(message)}")
            return
        
        msg_type = message.get("type")
        
        # Initialize actor_id from message if provided
        if msg_type == "init_central":
            self.actor_id = message.get("central_id", self.actor_id)
            self.plotter = ConsumptionPlotter(output_dir=f"plots/{self.actor_id}")
            logger.info(f"[{self.actor_id}] Initialized with ID from env")
            return
        
        if msg_type == "register_household":
            self._register_household(message, sender)
        elif msg_type == "consumption_data":
            self._handle_consumption_data(message, sender)
        elif msg_type == "historical_data":
            self._handle_historical_data(message)
        elif msg_type == "request_training":
            self._coordinate_training()
        elif msg_type == "model_update":
            self._handle_model_update(message, sender)
        elif msg_type == "discover_peer":
            self._discover_peer(message)
        elif msg_type == "sync_model":
            self._handle_peer_model_sync(message, sender)
        elif msg_type == "sync_crdt":
            self._handle_peer_crdt_sync(message)
        elif msg_type == "peer_data_share":
            self._handle_peer_data_share(message)
        elif msg_type == "peer_ack":
            # Handle peer acknowledgment
            peer_id = message.get("peer_id")
            peer_address = message.get("peer_address")
            if peer_id and peer_address:
                self.peer_centrals[peer_id] = peer_address
                logger.info(f"[{self.actor_id}] ✓ Peer {peer_id} confirmed at {peer_address}")
        elif msg_type == "discover_households":
            self._discover_households()
        elif msg_type == "household_shutdown":
            self._handle_household_shutdown(message)
        elif msg_type == "household_shutdown":
            self._handle_household_shutdown(message)
            
    def _check_and_control_consumption(self, household_id, consumption, room_details, household_addr):
        """Check consumption and send control commands if needed"""
        if consumption >= self.critical_household_consumption:
            # Transition to EMERGENCY state
            self._transition_state(CentralState.EMERGENCY)
            
            # Critical level - turn off highest consuming room
            logger.warning(
                f"[{self.actor_id}] ⚠️ CRITICAL consumption at {household_id}: {consumption:.2f} kWh! "
                f"Shutting down highest consuming room..."
            )
            self._shutdown_highest_consuming_room(household_id, room_details, household_addr)
            
            # Return to MONITORING state
            self._transition_state(CentralState.MONITORING)
            
        elif consumption >= self.max_household_consumption:
            # High level - request consumption reduction
            logger.warning(
                f"[{self.actor_id}] ⚠️ High consumption at {household_id}: {consumption:.2f} kWh. "
                f"Requesting reduction..."
            )
            self._request_consumption_reduction(household_id, household_addr)
    
    def _shutdown_highest_consuming_room(self, household_id, room_details, household_addr):
        """Shut down the room with highest consumption"""
        if not room_details:
            logger.warning(f"[{self.actor_id}] No room details available for {household_id}")
            return
        
        # Find room with highest consumption
        max_room = None
        max_consumption = 0
        for room_name, consumption in room_details.items():
            if consumption > max_consumption:
                max_consumption = consumption
                max_room = room_name
        
        if max_room:
            self.send(household_addr, {
                "type": "shutdown_room",
                "room_name": max_room,
                "reason": "critical_consumption"
            })
            logger.info(f"[{self.actor_id}] 🔴 Commanded {household_id} to shutdown {max_room} ({max_consumption:.2f} kWh)")
    
    def _request_consumption_reduction(self, household_id, household_addr):
        """Request household to reduce consumption"""
        self.send(household_addr, {
            "type": "reduce_consumption",
            "target_reduction": 0.2,  # Request 20% reduction
            "reason": "high_consumption"
        })
        logger.info(f"[{self.actor_id}] 📉 Requested {household_id} to reduce consumption by 20%")
    
    def _discover_households(self):
        """Proactively find household actors by globalName - only OUR households"""
        # Determine which households belong to this central
        # central_1 -> households 1,2 | central_2 -> households 3,4
        if self.actor_id == "central_1":
            household_ids = [1, 2]
        elif self.actor_id == "central_2":
            household_ids = [3, 4]
        else:
            household_ids = [1, 2, 3, 4]  # Fallback
            
        for i in household_ids:
            household_globalname = f"household_{i}"
            try:
                # Lookup existing household actor
                household_addr = self.createActor('actors.household_actor.HouseholdActor',
                                                  globalName=household_globalname)
                # Contact household with our info AND pass household_id
                self.send(household_addr, {
                    "type": "central_discovered",
                    "central_id": self.actor_id,
                    "household_id": f"household_{i}"  # Pass the ID!
                })
                logger.info(f"[{self.actor_id}] → Contacted {household_globalname}")
            except Exception as e:
                logger.debug(f"[{self.actor_id}] Could not find {household_globalname}: {e}")
    
    def _register_household(self, message, sender):
        """Register a new household"""
        household_id = message.get("household_id")
        self.household_actors[household_id] = sender  # Store household's address
        self.households[household_id] = sender
        self.fl_coordinator.num_households = len(self.households)
        
        logger.info(f"[{self.actor_id}] Registered household: {household_id} (Total: {len(self.households)})")
        
        # Track as child actor
        self.child_actors[household_id] = sender
        
        # Start periodic tasks if first household
        if len(self.households) == 1:
            # Transition to MONITORING state
            self._transition_state(CentralState.MONITORING)
    
    def _start_p2p_server(self):
        """Start TCP server for P2P communication"""
        try:
            self.p2p_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.p2p_server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.p2p_server_socket.bind(('0.0.0.0', self.p2p_port))
            self.p2p_server_socket.listen(5)
            
            self.p2p_server_thread = threading.Thread(target=self._accept_p2p_connections, daemon=True)
            self.p2p_server_thread.start()
            logger.info(f"[{self.actor_id}] P2P TCP server listening on port {self.p2p_port}")
        except Exception as e:
            logger.error(f"[{self.actor_id}] Failed to start P2P server: {e}")
    
    def _accept_p2p_connections(self):
        """Accept incoming P2P connections from peer centrals"""
        while True:
            try:
                client_socket, client_address = self.p2p_server_socket.accept()
                logger.info(f"[{self.actor_id}] Accepted P2P connection from {client_address}")
                threading.Thread(target=self._handle_p2p_client, args=(client_socket,), daemon=True).start()
            except Exception as e:
                logger.error(f"[{self.actor_id}] Error accepting P2P connection: {e}")
                break
    
    def _handle_p2p_client(self, client_socket):
        """Handle incoming messages from P2P client"""
        try:
            while True:
                # Read message length (4 bytes)
                raw_msglen = self._recvall(client_socket, 4)
                if not raw_msglen:
                    break
                msglen = struct.unpack('>I', raw_msglen)[0]
                
                # Read message data
                raw_msg = self._recvall(client_socket, msglen)
                if not raw_msg:
                    break
                
                # Deserialize and process message
                message = pickle.loads(raw_msg)
                self._process_p2p_message(message)
        except Exception as e:
            logger.error(f"[{self.actor_id}] Error handling P2P client: {e}")
        finally:
            client_socket.close()
    
    def _recvall(self, sock, n):
        """Helper to receive exactly n bytes"""
        data = bytearray()
        while len(data) < n:
            packet = sock.recv(n - len(data))
            if not packet:
                return None
            data.extend(packet)
        return bytes(data)
    
    def _process_p2p_message(self, message):
        """Process message received via P2P TCP"""
        msg_type = message.get("type")
        if msg_type == "consumption_data":
            peer_id = message.get("peer_id")
            data = message.get("data")
            logger.info(f"[{self.actor_id}] ← Received consumption data from peer {peer_id}: {data['household_id']} - {data['consumption']:.2f} kWh")
        elif msg_type == "model_update":
            peer_id = message.get("peer_id")
            peer_model = message.get("model")
            peer_round = message.get("training_round", 0)
            
            self.peer_models[peer_id] = {
                "model": peer_model,
                "round": peer_round
            }
            
            logger.info(f"[{self.actor_id}] ← Received model update from peer {peer_id} (round {peer_round})")
            
            # Merge peer models if we have completed our local aggregation
            if peer_round >= self.training_rounds and self.fl_coordinator.ready_to_aggregate() == False:
                self._merge_peer_models()
        elif msg_type == "sync_crdt":
            peer_id = message.get("peer_id")
            peer_crdt_state = message.get("crdt_state")
            self.crdt_register.merge(peer_crdt_state)
            logger.info(f"[{self.actor_id}] ← Received CRDT sync from peer {peer_id}")
    
    def _discover_peer(self, message):
        """Connect to peer central via raw TCP socket"""
        peer_id = message.get("peer_id")
        peer_host = message.get("peer_host")
        peer_port = message.get("peer_port")
        
        logger.info(f"[{self.actor_id}] Connecting to peer {peer_id} at {peer_host}:{peer_port}")
        
        try:
            # Parse P2P port from PEER_P2P_PORTS environment variable
            peer_p2p_ports = os.getenv('PEER_P2P_PORTS', '')
            p2p_port = None
            for peer_spec in peer_p2p_ports.split(','):
                if peer_spec.strip().startswith(peer_id + ':'):
                    p2p_port = int(peer_spec.split(':')[1])
                    break
            
            if not p2p_port:
                logger.error(f"[{self.actor_id}] No P2P port found for {peer_id}")
                return
            
            # Create TCP connection to peer
            peer_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            peer_socket.connect((peer_host, p2p_port))
            
            self.peer_centrals[peer_id] = peer_socket
            logger.info(f"[{self.actor_id}] ✓ Connected to peer {peer_id} at {peer_host}:{p2p_port} (Total: {len(self.peer_centrals)})")
            
        except Exception as e:
            logger.error(f"[{self.actor_id}] ✗ Failed to connect to peer {peer_id}: {e}")
        
    def _handle_consumption_data(self, message, sender):
        """Process incoming consumption data"""
        data = message.get("data")
        crdt_state = message.get("crdt_state")
        
        # Convert timestamp from ISO string to datetime object for plotting
        if 'timestamp' in data and isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        elif 'timestamp' not in data:
            data['timestamp'] = datetime.now()
        
        # Store data with proper timestamp and total_consumption key for plotting
        plot_data = {
            'household_id': data['household_id'],
            'timestamp': data['timestamp'],
            'total_consumption': data['consumption'],  # Key name for plotter
            'room_details': data.get('room_details', [])
        }
        
        self.aggregated_data.append(plot_data)
        self.data_count += 1
        self.crdt_register.merge(crdt_state)
        
        # Track current consumption
        household_id = data['household_id']
        consumption = data['consumption']
        room_details = data.get('room_details', {})
        
        self.household_consumption_tracking[household_id] = {
            "consumption": consumption,
            "timestamp": data['timestamp'],
            "room_details": room_details
        }
        
        logger.info(
            f"Central received data: {data['household_id']} - "
            f"{data['consumption']:.2f} kWh (Total data points: {self.data_count})"
        )
        
        # Check if consumption is too high and take action
        self._check_and_control_consumption(household_id, consumption, room_details, sender)
        
        # Keep only recent data for plotting (last 100 points to avoid overlapping chaos)
        if len(self.aggregated_data) > 100:
            self.aggregated_data = self.aggregated_data[-100:]
        
        # Share aggregated data with peer centrals for distributed monitoring
        self._share_data_with_peers(plot_data)
        
        # Trigger federated learning when enough data is collected
        if self.data_count >= self.training_threshold:
            self._coordinate_training()
            self.data_count = 0
            
    def _handle_historical_data(self, message):
        """Store historical data for analysis"""
        household_id = message.get("household_id")
        data = message.get("data")
        logger.info(f"Central received {len(data)} historical records from {household_id}")
        
    def _coordinate_training(self):
        """Coordinate federated learning round"""
        if len(self.households) == 0:
            logger.warning("No households registered, skipping training")
            return
        
        # Transition to COORDINATING state
        self._transition_state(CentralState.COORDINATING)
            
        self.training_rounds += 1
        logger.info(f"Starting federated learning round {self.training_rounds}")
        
        # Track training history for visualization
        self.training_history.append({
            'round': self.training_rounds,
            'num_participants': len(self.households),
            'loss': 0.0  # Will be updated after aggregation
        })
        
        # Request model updates from all households
        for household_id, household_actor in self.households.items():
            self.send(household_actor, {
                "type": "train_local_model",
                "global_model": self.fl_coordinator.get_global_model()
            })
        
    def _handle_model_update(self, message, sender):
        """Aggregate model updates from households"""
        household_id = message.get("household_id")
        local_weights = message.get("model_weights")
        evaluation_metrics = message.get("evaluation_metrics")
        
        self.fl_coordinator.add_local_update(local_weights)
        logger.info(f"[{self.actor_id}] Received model update from {household_id}")
        
        # Collect evaluation metrics
        if evaluation_metrics:
            self.current_round_metrics.append(evaluation_metrics)
            logger.info(
                f"📊 [{self.actor_id}] Metrics from {household_id}: "
                f"MSE={evaluation_metrics['MSE']:.4f}, "
                f"MAE={evaluation_metrics['MAE']:.4f}, "
                f"RMSE={evaluation_metrics['RMSE']:.4f}"
            )
        else:
            logger.warning(f"[{self.actor_id}] No metrics received from {household_id}")
        
        # When all households have sent updates, aggregate
        if self.fl_coordinator.ready_to_aggregate():
            global_model = self.fl_coordinator.aggregate_models()
            
            logger.info(f"[{self.actor_id}] Global model aggregated from {len(self.households)} households")
            
            # Return to MONITORING state after coordination
            self._transition_state(CentralState.MONITORING)
            
            # Aggregate evaluation metrics from all households
            if self.current_round_metrics:
                import numpy as np
                avg_mse = np.mean([m['MSE'] for m in self.current_round_metrics])
                avg_mae = np.mean([m['MAE'] for m in self.current_round_metrics])
                avg_rmse = np.mean([m['RMSE'] for m in self.current_round_metrics])
                
                aggregated_metrics = {
                    'round': self.training_rounds,
                    'MSE': float(avg_mse),
                    'MAE': float(avg_mae),
                    'RMSE': float(avg_rmse),
                    'num_participants': len(self.current_round_metrics)
                }
                
                self.evaluation_metrics.append(aggregated_metrics)
                
                logger.info(
                    f"📊 [{self.actor_id}] Round {self.training_rounds} Aggregated Metrics: "
                    f"MSE={avg_mse:.4f}, MAE={avg_mae:.4f}, RMSE={avg_rmse:.4f} "
                    f"(from {len(self.current_round_metrics)} households)"
                )
                
                # Clear for next round
                self.current_round_metrics = []
            
            # Update training history with aggregation loss
            if self.training_history:
                # Use RMSE as loss metric if available
                if self.evaluation_metrics:
                    self.training_history[-1]['loss'] = self.evaluation_metrics[-1]['RMSE']
                else:
                    # Fallback: Simple loss estimation based on model variance
                    import numpy as np
                    weights_variance = np.var(global_model['weights'])
                    self.training_history[-1]['loss'] = float(weights_variance)
            
            # Sync with peer CentralActors (Gossip protocol)
            self._sync_model_to_all_peers()
            
            # Generate plots periodically
            time_since_last_plot = (datetime.now() - self.last_plot_time).total_seconds()
            if time_since_last_plot >= self.plot_interval:
                self._generate_plots()
                self.last_plot_time = datetime.now()
            
            # Sync CRDT state via TCP
            for peer_id, peer_socket in self.peer_centrals.items():
                message = {
                    "type": "sync_crdt",
                    "peer_id": self.actor_id,
                    "crdt_state": self.crdt_register.get_state()
                }
                self._send_p2p_message(peer_socket, message)
    
    def _handle_peer_model_sync(self, message, sender):
        """Handle model synchronization from peer CentralActor"""
        peer_id = message.get("central_id")
        peer_model = message.get("model_weights")
        peer_round = message.get("training_round", 0)
        
        self.peer_models[peer_id] = {
            "model": peer_model,
            "round": peer_round
        }
        
        logger.info(f"[{self.actor_id}] Received model from peer {peer_id} (round {peer_round})")
        
        # If peer has newer model, consider merging
        if peer_round > self.training_rounds:
            self._merge_peer_models()
    
    def _handle_peer_crdt_sync(self, message):
        """Synchronize CRDT state with peer CentralActor"""
        peer_id = message.get("central_id")
        peer_crdt_state = message.get("crdt_state")
        
        self.crdt_register.merge(peer_crdt_state)
        logger.info(f"[{self.actor_id}] ← Received CRDT sync from peer {peer_id}")
    
    def _handle_peer_data_share(self, message):
        """Handle consumption data shared by peer central"""
        peer_id = message.get("central_id")
        data_point = message.get("data")
        
        # Log that we received peer's data (for monitoring/aggregation visibility)
        logger.info(f"[{self.actor_id}] ← Received consumption data from peer {peer_id}: "
                   f"{data_point.get('household_id')} - {data_point.get('total_consumption', 0):.2f} kWh")
    
    def _send_p2p_message(self, peer_socket, message):
        """Send message over TCP socket to peer"""
        try:
            msg_data = pickle.dumps(message)
            msg_len = struct.pack('>I', len(msg_data))
            peer_socket.sendall(msg_len + msg_data)
        except Exception as e:
            logger.error(f"[{self.actor_id}] Failed to send P2P message: {e}")
    
    def _share_data_with_peers(self, data_point):
        """Share consumption data with peer centrals for distributed aggregation"""
        if not self.peer_centrals:
            return
            
        for peer_id, peer_socket in self.peer_centrals.items():
            message = {
                "type": "consumption_data",
                "peer_id": self.actor_id,
                "data": {
                    "household_id": data_point['household_id'],
                    "consumption": data_point['total_consumption'],
                    "timestamp": data_point['timestamp'].isoformat()
                }
            }
            self._send_p2p_message(peer_socket, message)
        
        if len(self.peer_centrals) > 0:
            logger.info(f"[{self.actor_id}] → Shared consumption data with {len(self.peer_centrals)} peer(s)")
    
    def _sync_model_to_all_peers(self):
        """Broadcast current model to all peer CentralActors (Gossip protocol)"""
        global_model = self.fl_coordinator.get_global_model()
        
        for peer_id, peer_socket in self.peer_centrals.items():
            message = {
                "type": "model_update",
                "peer_id": self.actor_id,
                "model": global_model,
                "training_round": self.training_rounds
            }
            self._send_p2p_message(peer_socket, message)
            
        if self.peer_centrals:
            logger.info(f"[{self.actor_id}] Broadcasted model to {len(self.peer_centrals)} peers")
        
        # Generate plots periodically
        time_since_last_plot = (datetime.now() - self.last_plot_time).total_seconds()
        if time_since_last_plot >= self.plot_interval:
            self._generate_plots()
            self.last_plot_time = datetime.now()
    
    def _merge_peer_models(self):
        """Merge models from peer CentralActors for distributed aggregation"""
        if not self.peer_models:
            return
        
        # Collect all peer models
        all_models = [self.fl_coordinator.get_global_model()]
        all_models.extend([pm["model"] for pm in self.peer_models.values()])
        
        # Average all models (distributed FedAvg)
        import numpy as np
        avg_weights = np.mean([m["weights"] for m in all_models], axis=0)
        avg_bias = np.mean([m["bias"] for m in all_models])
        
        merged_model = {
            "weights": avg_weights,
            "bias": avg_bias
        }
        
        self.fl_coordinator.global_model = merged_model
        logger.info(f"[{self.actor_id}] Merged {len(all_models)} models from peers")
    
    def _generate_plots(self):
        """Generate visualization plots for consumption and FL training"""
        try:
            if not self.aggregated_data or len(self.aggregated_data) < 5:
                logger.debug(f"[{self.actor_id}] Not enough data for plotting (need 5+, have {len(self.aggregated_data)})") 
                return
            
            # Data is already in correct format from _handle_consumption_data
            logger.info(f"[{self.actor_id}] Generating visualization plots ({len(self.aggregated_data)} data points)...")
            
            self.plotter.plot_household_consumption(
                self.aggregated_data,  # Use directly
                filename=f"{self.actor_id}_consumption.png"
            )
            
            self.plotter.plot_room_consumption(
                self.aggregated_data,
                filename=f"{self.actor_id}_rooms.png"
            )
            
            if self.training_history:
                self.plotter.plot_fl_training(
                    self.training_history,
                    filename=f"{self.actor_id}_fl_training.png"
                )
            
            # Plot evaluation metrics if available
            if self.evaluation_metrics:
                logger.info(f"[{self.actor_id}] Generating evaluation metrics plot ({len(self.evaluation_metrics)} rounds)...")
                self.plotter.plot_evaluation_metrics(
                    self.evaluation_metrics,
                    filename=f"{self.actor_id}_evaluation_metrics.png"
                )
            else:
                logger.warning(f"[{self.actor_id}] No evaluation metrics to plot yet")
            
            # Generate comprehensive dashboard
            if len(self.aggregated_data) >= 10 and self.training_history:
                self.plotter.plot_summary_dashboard(
                    self.aggregated_data,
                    self.training_history,
                    filename=f"{self.actor_id}_dashboard.png"
                )
            
            # Log evaluation metrics summary
            if self.evaluation_metrics:
                latest_metrics = self.evaluation_metrics[-1]
                logger.info(
                    f"[{self.actor_id}] Latest Evaluation Metrics (Round {latest_metrics['round']}): "
                    f"MSE={latest_metrics['MSE']:.4f}, "
                    f"MAE={latest_metrics['MAE']:.4f}, "
                    f"RMSE={latest_metrics['RMSE']:.4f}"
                )
            
            logger.info(f"[{self.actor_id}] ✓ Plots generated successfully")
            
        except Exception as e:
            logger.error(f"[{self.actor_id}] Error generating plots: {e}", exc_info=True)
    
    def _transition_state(self, new_state: CentralState):
        """Transition to a new state with logging and history"""
        old_state = self.state
        self.state = new_state
        self.state_history.append({
            "from": old_state.value,
            "to": new_state.value,
            "timestamp": datetime.now()
        })
        logger.info(f"🔄 [{self.actor_id}] State transition: {old_state.value} → {new_state.value}")
    
    def _handle_exit_request(self):
        """Handle actor shutdown request"""
        logger.warning(f"⚠️ [{self.actor_id}] Received exit request, shutting down gracefully...")
        self._transition_state(CentralState.SHUTDOWN)
        self.shutdown_requested = True
        
        # Request shutdown of all household actors
        for household_id, household_addr in self.households.items():
            self.send(household_addr, ActorExitRequest())
            logger.info(f"📤 [{self.actor_id}] Sent shutdown request to {household_id}")
        
        # Close P2P connections
        for peer_id, peer_socket in self.peer_centrals.items():
            try:
                peer_socket.close()
                logger.info(f"🔌 [{self.actor_id}] Closed P2P connection to {peer_id}")
            except Exception as e:
                logger.error(f"Error closing P2P connection to {peer_id}: {e}")
        
        if self.p2p_server_socket:
            self.p2p_server_socket.close()
        
        logger.info(f"✓ [{self.actor_id}] Shutdown complete")
    
    def _handle_child_exit(self, message):
        """Handle child actor exit"""
        child_addr = message.childAddress
        
        # Find which household exited
        exited_household = None
        for household_id, addr in self.child_actors.items():
            if addr == child_addr:
                exited_household = household_id
                break
        
        if exited_household:
            logger.warning(f"👶 [{self.actor_id}] Child household {exited_household} exited unexpectedly")
            
            # Remove from tracking
            if exited_household in self.households:
                del self.households[exited_household]
            if exited_household in self.child_actors:
                del self.child_actors[exited_household]
            if exited_household in self.household_consumption_tracking:
                del self.household_consumption_tracking[exited_household]
            
            logger.info(f"🧹 [{self.actor_id}] Cleaned up {exited_household} (Remaining: {len(self.households)})")
        else:
            logger.info(f"👶 [{self.actor_id}] Unknown child actor exited: {child_addr}")
    
    def _handle_household_shutdown(self, message):
        """Handle graceful household shutdown notification"""
        household_id = message.get("household_id")
        final_state = message.get("final_state")
        state_history = message.get("state_history", [])
        
        logger.info(
            f"📥 [{self.actor_id}] {household_id} shutdown gracefully "
            f"(Final state: {final_state}, Transitions: {len(state_history)})"
        )
        
        # Remove from active households
        if household_id in self.households:
            del self.households[household_id]
        if household_id in self.household_consumption_tracking:
            del self.household_consumption_tracking[household_id]
        
        logger.info(f"🧹 [{self.actor_id}] Removed {household_id} (Remaining: {len(self.households)})")
