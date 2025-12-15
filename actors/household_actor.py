"""
Household Actor - monitors electricity consumption in a household
"""
from datetime import datetime, timedelta
from thespian.actors import Actor, ActorExitRequest, ChildActorExited
from models.consumption_data import ConsumptionData
from crdt.lww_register import LWWRegister
from simulation.consumption_simulator import ConsumptionSimulator
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class HouseholdState(Enum):
    """State machine states for household actor"""
    INITIALIZING = "initializing"  # Starting up
    IDLE = "idle"                  # Normal operation, low consumption
    ACTIVE = "active"              # Normal operation, active consumption
    REDUCING = "reducing"          # Actively reducing consumption
    CRITICAL = "critical"          # Critical mode, minimal consumption
    SHUTDOWN = "shutdown"          # Shutting down


class HouseholdActor(Actor):
    """Actor representing a single household that monitors electricity consumption"""
    
    def __init__(self):
        super().__init__()
        self.household_id = None
        self.central_actor = None
        self.consumption_history = []
        self.crdt_register = LWWRegister()
        self.monitoring_interval = 5  # seconds
        self.is_monitoring = False
        self.simulator = None  # ConsumptionSimulator instance
        
        # State machine
        self.state = HouseholdState.INITIALIZING
        self.state_history = []
        self.shutdown_requested = False
        
        
    def receiveMessage(self, message, sender):
        """Handle incoming messages"""
        # Handle Thespian system messages
        from thespian.actors import WakeupMessage
        if isinstance(message, WakeupMessage):
            if self.is_monitoring:
                self._collect_consumption_data()
            return
        
        # Handle application messages (must be dict)
        if not isinstance(message, dict):
            logger.warning(f"{self.household_id} - Received unknown message type: {type(message)}")
            return
        
        msg_type = message.get("type")
        
        if msg_type == "register":
            self._handle_registration(message, sender)
        elif msg_type == "central_discovered":
            self._handle_central_discovered(message, sender)
        elif msg_type == "stop_monitoring":
            self._stop_monitoring()
        elif msg_type == "get_data":
            self._send_data_to_central()
        elif msg_type == "train_local_model":
            self._train_local_model(message)
        elif msg_type == "update_model":
            self._update_prediction_model(message)
        elif msg_type == "sync_crdt":
            self._sync_crdt(message)
        elif msg_type == "initial_model":
            self._receive_initial_model(message)
        elif msg_type == "toggle_room":
            self._toggle_room(message)
        elif msg_type == "get_rooms":
            self._send_room_status(sender)
        elif msg_type == "shutdown_room":
            self._shutdown_room(message)
        elif msg_type == "reduce_consumption":
            self._reduce_consumption(message)
            
    def _handle_registration(self, message, sender):
        """Register with central actor - just initialize and WAIT for central to contact us"""
        self.household_id = message.get("household_id", f"household_{id(self)}")
        
        # Initialize simulator for this household
        self.simulator = ConsumptionSimulator(self.household_id)
        rooms = self.simulator.initialize_rooms()
        
        logger.info(f"{self.household_id} initialized with {len(rooms)} rooms")
        logger.info(f"→ Waiting for central to discover us...")
        
    def _handle_central_discovered(self, message, sender):
        """Handle when central actor contacts us - sender IS the central"""
        central_id = message.get("central_id")
        self.household_id = message.get("household_id")  # Get ID from central!
        self.central_actor = sender  # Central's address is the sender!
        
        # Initialize simulator NOW that we have household_id
        self.simulator = ConsumptionSimulator(self.household_id)
        rooms = self.simulator.initialize_rooms()
        
        # Transition from INITIALIZING to IDLE
        self._transition_state(HouseholdState.IDLE)
        
        logger.info(f"[{self.household_id}] ✓ Contacted by central: {central_id}")
        logger.info(f"{self.household_id} initialized with {len(rooms)} rooms")
        
        # Confirm registration
        self.send(self.central_actor, {
            "type": "register_household",
            "household_id": self.household_id
        })
        
        # Start monitoring
        self.is_monitoring = True
        self.wakeupAfter(timedelta(seconds=self.monitoring_interval))
        
    def _collect_consumption_data(self):
        """Collect and report consumption data"""
        # Get consumption from simulator
        total_consumption, room_consumptions = self.simulator.get_total_consumption()
        timestamp = datetime.now()
        
        data = ConsumptionData(
            household_id=self.household_id,
            consumption=total_consumption,
            timestamp=timestamp,
            room_details=room_consumptions
        )
        
        self.consumption_history.append(data)
        self.crdt_register.set(self.household_id, data.to_dict(), timestamp)
        
        # Transition to ACTIVE when sending data (if not already)
        if self.state == HouseholdState.IDLE:
            self._transition_state(HouseholdState.ACTIVE)
        
        active_count = self.simulator.get_active_room_count()
        total_rooms = len(self.simulator.rooms)
        logger.info(
            f"{self.household_id} - Total: {total_consumption:.2f} kWh "
            f"({active_count}/{total_rooms} rooms active)"
        )
        
        # Send to central actor
        if self.central_actor:
            self.send(self.central_actor, {
                "type": "consumption_data",
                "data": data.to_dict(),
                "crdt_state": self.crdt_register.get_state()
            })
        
        # Schedule next monitoring
        self.wakeupAfter(timedelta(seconds=self.monitoring_interval))
        

    
    def _send_data_to_central(self):
        """Send historical data to central actor"""
        if self.central_actor:
            self.send(self.central_actor, {
                "type": "historical_data",
                "household_id": self.household_id,
                "data": [d.to_dict() for d in self.consumption_history]
            })
            logger.info(f"{self.household_id} sent {len(self.consumption_history)} records to central")
    
    def _train_local_model(self, message):
        """Train local model on household data"""
        global_model = message.get("global_model")
        
        logger.info(f"[{self.household_id}] Training requested - have {len(self.consumption_history)} data points")
        
        if len(self.consumption_history) < 6:
            logger.warning(f"{self.household_id} has insufficient data for training (need 6+, have {len(self.consumption_history)})")
            return
        
        # REAL local training with gradient descent
        import numpy as np
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        
        local_weights = global_model["weights"].copy()
        local_bias = global_model["bias"]
        
        # train/test split (80% train, 20% test) 
        split_idx = int(len(self.consumption_history) * 0.8)
        train_data = self.consumption_history[:split_idx]  # Older data for training
        test_data = self.consumption_history[split_idx:]   # Newer data for testing
        
        # Training hyperparameters
        learning_rate = 0.01
        epochs = 10
        
        logger.info(
            f"[{self.household_id}] Starting training on {len(train_data)} data points "
            f"(test set: {len(test_data)} points) for {epochs} epochs..."
        )
        
        # Gradient descent training - ONLY on train_data
        for epoch in range(epochs):
            total_loss = 0.0
            
            for data in train_data:
                hour = data.timestamp.hour
                actual_consumption = data.consumption
                
                # Forward pass: prediction
                if 0 <= hour < 24:
                    predicted_consumption = local_weights[hour] + local_bias
                    
                    # Calculate error
                    error = actual_consumption - predicted_consumption
                    total_loss += error ** 2
                    # Backward pass: update weights and bias
                    local_weights[hour] += learning_rate * error
                    local_bias += learning_rate * error * 0.1  # Smaller update for bias
            
            avg_loss = total_loss / len(train_data)
            if epoch % 3 == 0:
                logger.info(f"[{self.household_id}] Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.4f}")
        
        logger.info(f"{self.household_id} completed local training")
        
        # Evaluate model on TEST data
        y_true = []
        y_pred = []
        
        for data in test_data:  # ← ONLY TEST DATA!
            hour = data.timestamp.hour
            y_true.append(data.consumption)
            # Prediction using trained model
            if 0 <= hour < 24:
                prediction = local_weights[hour] + local_bias
                y_pred.append(max(0.0, prediction))
        
        logger.info(f"[{self.household_id}] Evaluating on {len(y_true)} TEST data points (unseen during training)...")
        
        # Calculate evaluation metrics (need at least 2 points)
        if len(y_true) >= 2:
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            
            logger.info(
                f"📊 [{self.household_id}] Local Model Evaluation (on TEST set): "
                f"MSE={mse:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}"
            )
            
            evaluation_metrics = {
                "MSE": float(mse),
                "MAE": float(mae),
                "RMSE": float(rmse)
            }
        else:
            logger.warning(f"[{self.household_id}] Insufficient test data for evaluation (need 2+, have {len(y_true)})")
            evaluation_metrics = None
        
        # Send updated model back to central with evaluation metrics
        self.send(self.central_actor, {
            "type": "model_update",
            "household_id": self.household_id,
            "model_weights": {
                "weights": local_weights,
                "bias": local_bias
            },
            "evaluation_metrics": evaluation_metrics
        })
    
    def _update_prediction_model(self, message):
        """Update local prediction model from federated learning"""
        model_weights = message.get("model_weights")
        logger.info(f"{self.household_id} received updated global model")
        
    def _receive_initial_model(self, message):
        """Receive initial model from central actor"""
        logger.info(f"{self.household_id} received initial model from central")
        
    def _sync_crdt(self, message):
        """Synchronize CRDT state with other actors"""
        remote_state = message.get("crdt_state")
        self.crdt_register.merge(remote_state)
        logger.info(f"{self.household_id} synchronized CRDT state")
        
    def _stop_monitoring(self):
        """Stop monitoring"""
        self.is_monitoring = False
        logger.info(f"{self.household_id} stopped monitoring")
    
    def _toggle_room(self, message):
        """Toggle a room on/off"""
        room_name = message.get("room_name")
        self.simulator.toggle_room(room_name)
    
    def _send_room_status(self, sender):
        """Send current status of all rooms"""
        room_status = self.simulator.get_room_status()
    
    def _shutdown_room(self, message):
        """Shutdown a specific room due to high consumption"""
        room_name = message.get("room_name")
        reason = message.get("reason", "unknown")
        
        # Transition to CRITICAL state
        self._transition_state(HouseholdState.CRITICAL)
        
        if self.simulator:
            success = self.simulator.turn_off_room(room_name)
            if success:
                logger.warning(
                    f"🔴 [{self.household_id}] Room '{room_name}' SHUTDOWN by central actor "
                    f"(Reason: {reason})"
                )
                # Return to ACTIVE state
                self._transition_state(HouseholdState.ACTIVE)
            else:
                logger.error(f"[{self.household_id}] Failed to shutdown room '{room_name}'")
    
    def _reduce_consumption(self, message):
        """Reduce overall consumption by turning off some rooms"""
        target_reduction = message.get("target_reduction", 0.2)
        reason = message.get("reason", "unknown")
        
        # Transition to REDUCING state
        self._transition_state(HouseholdState.REDUCING)
        
        if self.simulator:
            rooms_turned_off = self.simulator.reduce_consumption(target_reduction)
            if rooms_turned_off:
                logger.warning(
                    f"📉 [{self.household_id}] Reduced consumption by turning off: {', '.join(rooms_turned_off)} "
                    f"(Reason: {reason})"
                )
            else:
                logger.info(f"[{self.household_id}] No rooms available to turn off for consumption reduction")
        
        # Return to ACTIVE state
        self._transition_state(HouseholdState.ACTIVE)
    
    def _transition_state(self, new_state: HouseholdState):
        """Transition to a new state with logging and history"""
        old_state = self.state
        self.state = new_state
        self.state_history.append({
            "from": old_state.value,
            "to": new_state.value,
            "timestamp": datetime.now()
        })
        logger.info(f"🔄 [{self.household_id}] State transition: {old_state.value} → {new_state.value}")
    
    def _handle_exit_request(self):
        """Handle actor shutdown request"""
        logger.warning(f"⚠️ [{self.household_id}] Received exit request, shutting down gracefully...")
        self._transition_state(HouseholdState.SHUTDOWN)
        self.shutdown_requested = True
        
        # Notify central actor
        if self.central_actor:
            self.send(self.central_actor, {
                "type": "household_shutdown",
                "household_id": self.household_id,
                "final_state": self.state.value,
                "state_history": self.state_history
            })
        
        logger.info(f"✓ [{self.household_id}] Shutdown complete")
    
    def _handle_child_exit(self, message):
        """Handle child actor exit"""
        logger.info(f"👶 [{self.household_id}] Child actor exited: {message.childAddress}")
