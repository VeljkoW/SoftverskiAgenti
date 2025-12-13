"""
Main entry point for the electricity consumption monitoring system
"""
import logging
import time
from thespian.actors import ActorSystem
from actors.household_actor import HouseholdActor
from actors.central_actor import CentralActor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Initialize and start the actor system"""
    logger.info("Starting electricity monitoring system...")
    
    # Initialize actor system with TCP base for distributed communication
    actor_system = ActorSystem('multiprocTCPBase')
    
    try:
        # Create multiple central actors for distributed coordination
        central_actor_1 = actor_system.createActor(CentralActor)
        central_actor_2 = actor_system.createActor(CentralActor)
        logger.info("Central actors created (distributed architecture)")
        
        # Register central actors as peers
        actor_system.tell(central_actor_1, {
            "type": "register_peer_central",
            "central_id": "central_2"
        })
        
        actor_system.tell(central_actor_2, {
            "type": "register_peer_central", 
            "central_id": "central_1"
        })
        
        # Create household actors
        household_1 = actor_system.createActor(HouseholdActor)
        household_2 = actor_system.createActor(HouseholdActor)
        household_3 = actor_system.createActor(HouseholdActor)
        household_4 = actor_system.createActor(HouseholdActor)
        logger.info("Household actors created")
        
        # Distribute households across central actors
        # Households 1-2 connect to central_1
        actor_system.tell(household_1, {
            "type": "register",
            "central_actor": central_actor_1,
            "household_id": "household_1"
        })
        
        actor_system.tell(household_2, {
            "type": "register",
            "central_actor": central_actor_1,
            "household_id": "household_2"
        })
        
        # Households 3-4 connect to central_2
        actor_system.tell(household_3, {
            "type": "register",
            "central_actor": central_actor_2,
            "household_id": "household_3"
        })
        
        actor_system.tell(household_4, {
            "type": "register",
            "central_actor": central_actor_2,
            "household_id": "household_4"
        })
        
        logger.info("Distributed system running with 2 central actors and 4 households.")
        logger.info("Press Ctrl+C to stop.")
        
        # Keep the system running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Shutting down system...")
    finally:
        actor_system.shutdown()


if __name__ == "__main__":
    main()
