"""
Docker entry point - creates ONLY the actor specified by ACTOR_TYPE env var
"""
# IMMEDIATE PRINT - NO LOGGING BUFFER ISSUES
print("=" * 70)
print("DOCKER_ENTRY.PY IS EXECUTING!!!")
print("=" * 70)

import logging
import time
import os
import signal
import sys
from thespian.actors import ActorSystem, ActorExitRequest
from actors.household_actor import HouseholdActor
from actors.central_actor import CentralActor  # Needed for globalName lookup

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)

print(f"ACTOR_TYPE from env: {os.getenv('ACTOR_TYPE', 'NOT_SET')}")

# Global flag for graceful shutdown
shutdown_requested = False

def signal_handler(signum, frame):
    """Handle SIGTERM and SIGINT for graceful shutdown"""
    global shutdown_requested
    signal_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
    logger.warning(f"⚠️ Received {signal_name} - initiating graceful shutdown...")
    shutdown_requested = True


def main():
    actor_type = os.getenv('ACTOR_TYPE', 'household')
    
    logger.info(f"==============================================")
    logger.info(f"  Docker Entry Point")
    logger.info(f"  ACTOR_TYPE={actor_type}")
    logger.info(f"  ENV VARS: {dict(os.environ)}")
    logger.info(f"==============================================")
    
    if actor_type == 'central':
        run_central()
    elif actor_type == 'household':
        run_household()
    else:
        logger.error(f"Unknown ACTOR_TYPE: {actor_type}")


def run_central():
    """Run ONLY ONE central actor in this container"""
    global shutdown_requested
    
    # Register signal handlers FIRST
    signal.signal(signal.SIGTERM, signal_handler)  # Docker stop
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    
    central_id = os.getenv('CENTRAL_ID', 'central_1')
    actor_port = int(os.getenv('ACTOR_PORT', 1900))
    peer_centrals_str = os.getenv('PEER_CENTRALS', '')
    
    logger.info(f"Starting Central Actor: {central_id} on port {actor_port}")
    
    # TCP-based system for P2P communication
    capabilities = {
        'Admin Port': actor_port,
        'Convention Address.IPv4': ('0.0.0.0', actor_port)
    }
    
    # Check if this is Convention Leader or regular node
    convention_addr = os.environ.get('THESPIAN_CONVENTION_ADDR')
    if convention_addr:
        # This is a regular node - connect to Convention Leader
        host, port = convention_addr.split(':')
        capabilities['Convention Address.IPv4'] = (host, int(port))
        logger.info(f"Connecting to Convention Leader at {convention_addr}")
    
    actor_system = ActorSystem('multiprocTCPBase', capabilities=capabilities)
    
    try:
        # Create ONLY this central actor
        central_actor = actor_system.createActor(CentralActor, 
                                                 globalName=f"central_{central_id}")
        
        # Send initialization message with central_id
        actor_system.tell(central_actor, {
            "type": "init_central",
            "central_id": central_id
        })
        
        logger.info(f"✓ {central_id} running on port {actor_port}")
        
        # Discover peer centrals after short delay
        if peer_centrals_str:
            time.sleep(3)
            for peer_spec in peer_centrals_str.split(','):
                if ':' in peer_spec:
                    peer_host, peer_port = peer_spec.strip().split(':')
                    logger.info(f"Registering peer: {peer_host}:{peer_port}")
                    
                    # Send peer registration message
                    actor_system.tell(central_actor, {
                        "type": "discover_peer",
                        "peer_id": peer_host,
                        "peer_host": peer_host,
                        "peer_port": int(peer_port)
                    })
        
        logger.info(f"✓ {central_id} ready with P2P network")
        
        # PROAKTIVNO traži household aktere - mora da čeka da se kreiraju
        time.sleep(2)
        logger.info(f"Discovering household actors...")
        actor_system.tell(central_actor, {"type": "discover_households"})
        
        # Keep running until shutdown requested
        logger.info(f"✓ {central_id} running - waiting for shutdown signal...")
        while not shutdown_requested:
            time.sleep(0.5)  # Check flag every 0.5s
        
        # Graceful shutdown
        logger.warning(f"🛑 Shutting down {central_id} gracefully...")
        actor_system.tell(central_actor, ActorExitRequest())
        time.sleep(2)  # Give actor time to cleanup
            
    except KeyboardInterrupt:
        logger.info(f"⚠️ KeyboardInterrupt - Shutting down {central_id}")
        actor_system.tell(central_actor, ActorExitRequest())
        time.sleep(2)
    except Exception as e:
        logger.error(f"❌ Error in {central_id}: {e}", exc_info=True)
    finally:
        logger.info(f"🔌 {central_id} shutting down ActorSystem...")
        actor_system.shutdown()
        logger.info(f"✓ {central_id} shutdown complete")


def run_household():
    """Run ONLY ONE household actor in this container"""
    global shutdown_requested
    
    # Register signal handlers FIRST
    signal.signal(signal.SIGTERM, signal_handler)  # Docker stop
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    
    household_id = int(os.getenv('HOUSEHOLD_ID', 1))
    central_host = os.getenv('CENTRAL_HOST', 'central_1')
    central_port = int(os.getenv('CENTRAL_PORT', 1900))
    
    logger.info(f"Starting Household Actor: household_{household_id}")
    logger.info(f"Will connect to: {central_host}:{central_port}")
    
    # Get Convention Leader address from environment
    convention_addr = os.environ.get('THESPIAN_CONVENTION_ADDR', f"{central_host}:{central_port}")
    conv_host, conv_port = convention_addr.split(':')
    
    # TCP-based system - connect to Convention Leader
    actor_system = ActorSystem('multiprocTCPBase', 
                                capabilities={'Convention Address.IPv4': (conv_host, int(conv_port))})
    
    try:
        # Create ONLY this household actor
        household_actor = actor_system.createActor(HouseholdActor, 
                                                     globalName=f"household_{household_id}")
        logger.info(f"✓ household_{household_id} created with globalName")
        
        # Send initial config - household will discover central via convention
        actor_system.tell(household_actor, {
            "type": "register",
            "household_id": f"household_{household_id}",
            "central_globalname": central_host  # Central's globalName
        })
        
        logger.info(f"✓ household_{household_id} sent registration request to {central_host}")
        
        # Keep running until shutdown requested
        logger.info(f"✓ household_{household_id} running - waiting for shutdown signal...")
        while not shutdown_requested:
            time.sleep(0.5)  # Check flag every 0.5s
        
        # Graceful shutdown
        logger.warning(f"🛑 Shutting down household_{household_id} gracefully...")
        actor_system.tell(household_actor, ActorExitRequest())
        time.sleep(2)  # Give actor time to cleanup
            
    except KeyboardInterrupt:
        logger.info(f"⚠️ KeyboardInterrupt - Shutting down household_{household_id}")
        actor_system.tell(household_actor, ActorExitRequest())
        time.sleep(2)
    except Exception as e:
        logger.error(f"❌ Error in household_{household_id}: {e}", exc_info=True)
    finally:
        logger.info(f"🔌 household_{household_id} shutting down ActorSystem...")
        actor_system.shutdown()
        logger.info(f"✓ household_{household_id} shutdown complete")


if __name__ == "__main__":
    main()
