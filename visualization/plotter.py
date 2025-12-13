"""
Matplotlib-based visualization for electricity consumption and federated learning.
"""

import os
import logging
from datetime import datetime
from typing import List, Dict, Any
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Docker
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from collections import defaultdict
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ConsumptionPlotter:
    """Creates visualizations for electricity consumption and FL training."""
    
    def __init__(self, output_dir: str = "plots"):
        """
        Initialize plotter.
        
        Args:
            output_dir: Directory to save plot files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        
    def plot_household_consumption(self, 
                                   consumption_data: List[Dict[str, Any]], 
                                   filename: str = "household_consumption.png"):
        """
        Plot consumption over time for all households.
        
        Args:
            consumption_data: List of dicts with keys: household_id, timestamp, total_consumption
            filename: Output filename
        """
        if not consumption_data:
            logger.warning("No consumption data to plot")
            return
        
        # Clear any existing figures to prevent clipping
        plt.close('all')
            
        # Organize data by household
        households = defaultdict(list)
        for data in consumption_data:
            household_id = data.get('household_id', 'Unknown')
            households[household_id].append({
                'timestamp': data.get('timestamp', datetime.now()),
                'consumption': data.get('total_consumption', 0)
            })
        
        # Create plot
        fig, ax = plt.subplots(figsize=(14, 7))
        
        for household_id, data_points in sorted(households.items()):
            if not data_points:
                continue
                
            timestamps = [d['timestamp'] for d in data_points]
            consumptions = [d['consumption'] for d in data_points]
            
            ax.plot(timestamps, consumptions, marker='o', linewidth=2, 
                   markersize=4, label=household_id, alpha=0.8)
        
        ax.set_xlabel('Time', fontsize=12, fontweight='bold')
        ax.set_ylabel('Consumption (kWh)', fontsize=12, fontweight='bold')
        ax.set_title('Electricity Consumption Over Time by Household', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.xticks(rotation=45, ha='right')
        
        fig.tight_layout(pad=2.0)
        output_path = os.path.join(self.output_dir, filename)
        fig.savefig(output_path, dpi=150, facecolor='white', edgecolor='white', pad_inches=0.5)
        plt.close(fig)
        plt.close('all')
        
        logger.info(f"Saved household consumption plot to {output_path}")
        
    def plot_room_consumption(self, 
                             consumption_data: List[Dict[str, Any]], 
                             filename: str = "room_consumption.png"):
        """
        Plot consumption distribution by room type.
        
        Args:
            consumption_data: List of dicts with room_details
            filename: Output filename
        """
        if not consumption_data:
            logger.warning("No consumption data to plot")
            return
        
        # Clear any existing figures to prevent clipping
        plt.close('all')
        plt.clf()
            
        # Aggregate consumption by room type
        room_consumption = defaultdict(list)
        
        for data in consumption_data:
            room_details = data.get('room_details', {})
            # room_details is dict like {"Living Room": 2.5, "Kitchen": 1.8}
            for room_name, consumption in room_details.items():
                if consumption > 0:  # Only active rooms have consumption > 0
                    room_consumption[room_name].append(consumption)
        
        if not room_consumption:
            logger.warning("No room consumption data available")
            return
        
        # Calculate statistics
        room_types = []
        avg_consumptions = []
        std_consumptions = []
        
        for room_type in sorted(room_consumption.keys()):
            consumptions = room_consumption[room_type]
            room_types.append(room_type.replace('_', ' ').title())
            avg_consumptions.append(np.mean(consumptions))
            std_consumptions.append(np.std(consumptions))
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=(12, 7))
        
        x_pos = np.arange(len(room_types))
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(room_types)))
        
        bars = ax.bar(x_pos, avg_consumptions, yerr=std_consumptions, 
                     capsize=5, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        ax.set_xlabel('Room Type', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Consumption (kWh)', fontsize=12, fontweight='bold')
        ax.set_title('Average Electricity Consumption by Room Type', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(room_types, rotation=15, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, avg_consumptions)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2f}',
                   ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        plt.tight_layout(pad=2.0)
        output_path = os.path.join(self.output_dir, filename)
        fig.savefig(output_path, dpi=150, facecolor='white', edgecolor='white', pad_inches=0.5)
        plt.close(fig)
        plt.close('all')
        
        logger.info(f"Saved room consumption plot to {output_path}")
        
    def plot_fl_training(self, 
                        training_history: List[Dict[str, Any]], 
                        filename: str = "fl_training.png"):
        """
        Plot federated learning training progress.
        
        Args:
            training_history: List of dicts with keys: round, loss, num_participants
            filename: Output filename
        """
        if not training_history:
            logger.warning("No training history to plot")
            return
        
        # Clear any existing plots
        plt.close('all')
            
        rounds = [h['round'] for h in training_history]
        losses = [h.get('loss', 0) for h in training_history]
        participants = [h.get('num_participants', 0) for h in training_history]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Training Loss
        ax1.plot(rounds, losses, marker='o', linewidth=2.5, 
                markersize=8, color='#e74c3c', label='Training Loss')
        ax1.set_xlabel('FL Round', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax1.set_title('Federated Learning Training Loss', 
                     fontsize=14, fontweight='bold', pad=15)
        ax1.legend(loc='best', framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        
        # Add trend line
        if len(rounds) > 1:
            z = np.polyfit(rounds, losses, 1)
            p = np.poly1d(z)
            ax1.plot(rounds, p(rounds), "--", alpha=0.6, color='#c0392b', 
                    linewidth=2, label='Trend')
            ax1.legend(loc='best', framealpha=0.9)
        
        # Plot 2: Participants per Round
        ax2.bar(rounds, participants, color='#3498db', alpha=0.8, 
               edgecolor='black', linewidth=1.5)
        ax2.set_xlabel('FL Round', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Number of Participants', fontsize=12, fontweight='bold')
        ax2.set_title('Federated Learning Participation', 
                     fontsize=14, fontweight='bold', pad=15)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (r, p) in enumerate(zip(rounds, participants)):
            ax2.text(r, p, str(p), ha='center', va='bottom', 
                    fontweight='bold', fontsize=10)
        
        fig.tight_layout(pad=2.0)
        output_path = os.path.join(self.output_dir, filename)
        fig.savefig(output_path, dpi=150, facecolor='white', edgecolor='white', pad_inches=0.5)
        plt.close(fig)
        plt.close('all')
        
        logger.info(f"Saved FL training plot to {output_path}")
        
    def plot_consumption_heatmap(self,
                                consumption_data: List[Dict[str, Any]],
                                filename: str = "consumption_heatmap.png"):
        """
        Create heatmap of consumption by household and time.
        
        Args:
            consumption_data: List of consumption data points
            filename: Output filename
        """
        if not consumption_data:
            logger.warning("No consumption data for heatmap")
            return
        
        # Clear any existing plots
        plt.close('all')
        
        # Convert to DataFrame
        df_data = []
        for data in consumption_data:
            df_data.append({
                'household': data.get('household_id', 'Unknown'),
                'timestamp': data.get('timestamp', datetime.now()),
                'consumption': data.get('total_consumption', 0)
            })
        
        df = pd.DataFrame(df_data)
        
        if df.empty:
            return
        
        # Extract hour from timestamp
        df['hour'] = df['timestamp'].dt.hour
        
        # Pivot table: households x hours
        pivot = df.pivot_table(values='consumption', 
                              index='household', 
                              columns='hour', 
                              aggfunc='mean')
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(14, 8))
        
        im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto', interpolation='nearest')
        
        # Set ticks
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_xticklabels([f"{int(h):02d}:00" for h in pivot.columns])
        ax.set_yticklabels(pivot.index)
        
        # Labels
        ax.set_xlabel('Hour of Day', fontsize=12, fontweight='bold')
        ax.set_ylabel('Household', fontsize=12, fontweight='bold')
        ax.set_title('Consumption Heatmap by Household and Hour', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Consumption (kWh)', fontsize=11, fontweight='bold')
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        
        fig.tight_layout(pad=2.0)
        output_path = os.path.join(self.output_dir, filename)
        fig.savefig(output_path, dpi=150, facecolor='white', edgecolor='white', pad_inches=0.5)
        plt.close(fig)
        plt.close('all')
        
        logger.info(f"Saved consumption heatmap to {output_path}")
        
    def plot_summary_dashboard(self,
                              consumption_data: List[Dict[str, Any]],
                              training_history: List[Dict[str, Any]],
                              filename: str = "dashboard.png"):
        """
        Create comprehensive dashboard with multiple plots.
        
        Args:
            consumption_data: Consumption data points
            training_history: FL training history
            filename: Output filename
        """
        # Clear any existing plots
        plt.close('all')
        
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # 1. Consumption over time
        ax1 = fig.add_subplot(gs[0, :])
        households = defaultdict(list)
        for data in consumption_data:
            household_id = data.get('household_id', 'Unknown')
            households[household_id].append({
                'timestamp': data.get('timestamp', datetime.now()),
                'consumption': data.get('total_consumption', 0)
            })
        
        for household_id, data_points in sorted(households.items()):
            if data_points:
                timestamps = [d['timestamp'] for d in data_points]
                consumptions = [d['consumption'] for d in data_points]
                ax1.plot(timestamps, consumptions, marker='o', linewidth=2, 
                        markersize=4, label=household_id, alpha=0.8)
        
        ax1.set_xlabel('Time', fontweight='bold')
        ax1.set_ylabel('Consumption (kWh)', fontweight='bold')
        ax1.set_title('Consumption Over Time', fontsize=13, fontweight='bold')
        ax1.legend(loc='best', ncol=2, fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 2. Room consumption
        ax2 = fig.add_subplot(gs[1, 0])
        room_consumption = defaultdict(list)
        for data in consumption_data:
            room_details = data.get('room_details', {})
            # room_details is dict like {"Living Room": 2.5, "Kitchen": 1.8}
            for room_name, consumption in room_details.items():
                if consumption > 0:  # Only active rooms
                    room_consumption[room_name].append(consumption)
        
        if room_consumption:
            room_types = []
            avg_consumptions = []
            for room_type in sorted(room_consumption.keys()):
                room_types.append(room_type.replace('_', ' ').title())
                avg_consumptions.append(np.mean(room_consumption[room_type]))
            
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(room_types)))
            ax2.bar(range(len(room_types)), avg_consumptions, color=colors, alpha=0.8)
            ax2.set_xticks(range(len(room_types)))
            ax2.set_xticklabels(room_types, rotation=45, ha='right', fontsize=8)
            ax2.set_ylabel('Avg Consumption (kWh)', fontweight='bold')
            ax2.set_title('Room Type Consumption', fontsize=13, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. FL Training Loss
        ax3 = fig.add_subplot(gs[1, 1])
        if training_history:
            rounds = [h['round'] for h in training_history]
            losses = [h.get('loss', 0) for h in training_history]
            ax3.plot(rounds, losses, marker='o', linewidth=2.5, 
                    markersize=8, color='#e74c3c')
            ax3.set_xlabel('FL Round', fontweight='bold')
            ax3.set_ylabel('Loss', fontweight='bold')
            ax3.set_title('FL Training Progress', fontsize=13, fontweight='bold')
            ax3.grid(True, alpha=0.3)
        
        # 4. Total consumption statistics
        ax4 = fig.add_subplot(gs[2, 0])
        household_totals = defaultdict(float)
        for data in consumption_data:
            household_id = data.get('household_id', 'Unknown')
            household_totals[household_id] += data.get('total_consumption', 0)
        
        if household_totals:
            households_sorted = sorted(household_totals.items(), key=lambda x: x[1], reverse=True)
            names = [h[0] for h in households_sorted]
            totals = [h[1] for h in households_sorted]
            
            ax4.barh(range(len(names)), totals, color='#2ecc71', alpha=0.8)
            ax4.set_yticks(range(len(names)))
            ax4.set_yticklabels(names, fontsize=9)
            ax4.set_xlabel('Total Consumption (kWh)', fontweight='bold')
            ax4.set_title('Total Consumption by Household', fontsize=13, fontweight='bold')
            ax4.grid(True, alpha=0.3, axis='x')
        
        # 5. Active rooms distribution
        ax5 = fig.add_subplot(gs[2, 1])
        room_activity = defaultdict(int)
        for data in consumption_data:
            room_details = data.get('room_details', {})
            # room_details is dict like {"Living Room": 2.5, "Kitchen": 1.8}
            for room_name, consumption in room_details.items():
                if consumption > 0:  # Only active rooms
                    room_activity[room_name] += 1
        
        if room_activity:
            labels = [rt.replace('_', ' ').title() for rt in sorted(room_activity.keys())]
            sizes = [room_activity[rt] for rt in sorted(room_activity.keys())]
            colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
            
            ax5.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, 
                   startangle=90, textprops={'fontsize': 9})
            ax5.set_title('Active Room Distribution', fontsize=13, fontweight='bold')
        
        plt.suptitle('Electricity Consumption Monitoring Dashboard', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        fig.tight_layout(pad=2.0)
        output_path = os.path.join(self.output_dir, filename)
        fig.savefig(output_path, dpi=150, facecolor='white', edgecolor='white', pad_inches=0.5)
        plt.close(fig)
        plt.close('all')
        
        logger.info(f"Saved dashboard to {output_path}")
    
    def plot_evaluation_metrics(self, 
                               evaluation_metrics: List[Dict[str, Any]], 
                               filename: str = "evaluation_metrics.png"):
        """
        Plot evaluation metrics (MSE, MAE, RMSE) over training rounds.
        
        Args:
            evaluation_metrics: List of dicts with keys: round, MSE, MAE, RMSE
            filename: Output filename
        """
        if not evaluation_metrics:
            logger.warning("No evaluation metrics to plot")
            return
        
        plt.close('all')
        
        rounds = [m['round'] for m in evaluation_metrics]
        mse_values = [m['MSE'] for m in evaluation_metrics]
        mae_values = [m['MAE'] for m in evaluation_metrics]
        rmse_values = [m['RMSE'] for m in evaluation_metrics]
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
        
        # Plot MSE
        ax1.plot(rounds, mse_values, marker='o', linewidth=2.5, 
                markersize=8, color='#e74c3c', label='MSE')
        ax1.set_xlabel('FL Round', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Mean Squared Error', fontsize=12, fontweight='bold')
        ax1.set_title('Mean Squared Error (MSE) per Training Round', 
                     fontsize=14, fontweight='bold', pad=15)
        ax1.legend(loc='best', framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        
        # Add trend line for MSE
        if len(rounds) > 1:
            z = np.polyfit(rounds, mse_values, 1)
            p = np.poly1d(z)
            ax1.plot(rounds, p(rounds), "--", alpha=0.6, color='#c0392b', 
                    linewidth=2, label='Trend')
            ax1.legend(loc='best', framealpha=0.9)
        
        # Plot MAE
        ax2.plot(rounds, mae_values, marker='s', linewidth=2.5, 
                markersize=8, color='#f39c12', label='MAE')
        ax2.set_xlabel('FL Round', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Mean Absolute Error', fontsize=12, fontweight='bold')
        ax2.set_title('Mean Absolute Error (MAE) per Training Round', 
                     fontsize=14, fontweight='bold', pad=15)
        ax2.legend(loc='best', framealpha=0.9)
        ax2.grid(True, alpha=0.3)
        
        # Add trend line for MAE
        if len(rounds) > 1:
            z = np.polyfit(rounds, mae_values, 1)
            p = np.poly1d(z)
            ax2.plot(rounds, p(rounds), "--", alpha=0.6, color='#d68910', 
                    linewidth=2, label='Trend')
            ax2.legend(loc='best', framealpha=0.9)
        
        # Plot RMSE
        ax3.plot(rounds, rmse_values, marker='^', linewidth=2.5, 
                markersize=8, color='#9b59b6', label='RMSE')
        ax3.set_xlabel('FL Round', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Root Mean Squared Error', fontsize=12, fontweight='bold')
        ax3.set_title('Root Mean Squared Error (RMSE) per Training Round', 
                     fontsize=14, fontweight='bold', pad=15)
        ax3.legend(loc='best', framealpha=0.9)
        ax3.grid(True, alpha=0.3)
        
        # Add trend line for RMSE
        if len(rounds) > 1:
            z = np.polyfit(rounds, rmse_values, 1)
            p = np.poly1d(z)
            ax3.plot(rounds, p(rounds), "--", alpha=0.6, color='#7d3c98', 
                    linewidth=2, label='Trend')
            ax3.legend(loc='best', framealpha=0.9)
        
        fig.tight_layout(pad=2.0)
        output_path = os.path.join(self.output_dir, filename)
        fig.savefig(output_path, dpi=150, facecolor='white', edgecolor='white', pad_inches=0.5)
        plt.close(fig)
        plt.close('all')
        
        logger.info(f"Saved evaluation metrics plot to {output_path}")
