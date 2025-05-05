# Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib.patches import Rectangle
from collections import defaultdict
import pandas as pd
import os

# Set Chinese font for matplotlib
plt.rcParams['font.sans-serif'] = ['SimHei']  # Using SimHei font for Chinese characters
plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display issue

# Global parameter settings
SIMULATION_TIMES = 3000  # Number of simulation runs
DAYS = 200               # Number of days in each simulation
TAX_RATE = 0.1           # Tax rate (10%)
FINE_RATE = 2            # Fine multiplier when caught evading taxes
BASE_PRICES = np.round(np.arange(2.0, 5.1, 0.5), 1)  # Base price range for goods
init_asset = 100         # Initial wealth for each person
cool_day_bad = 1         # Cooldown days for bad actors after being caught
cool_day_neutral = 2     # Cooldown days for neutral actors after being caught
evade_rate_bad = 0.8     # Probability bad actors will evade taxes
evade_rate_neutral = 0.5 # Probability neutral actors will evade taxes
audit_rate = 0.2         # Probability of being audited when evading taxes
discount_rate = 0.95     # Discount rate when evading taxes
c = 0.0                  # Incremental cost for secondary transactions

# Define Person class to represent individuals in the simulation
class Person:
    def __init__(self, role, occupation, pid):
        self.role = role        # 'good', 'bad', or 'neutral'
        self.occupation = occupation  # 'farmer' or 'herder'
        self.pid = pid          # Person ID
        self.wealth = init_asset  # Current wealth
        self.position = 0       # Position (not currently used)
        self.cool_down = 0      # Days remaining in cooldown after being caught

def initialize_people():
    """Initialize 6 people with different roles and occupations"""
    return [
        Person('good', 'farmer', 0),
        Person('bad', 'farmer', 1),
        Person('neutral', 'farmer', 2),
        Person('good', 'herder', 3),
        Person('bad', 'herder', 4),
        Person('neutral', 'herder', 5)
    ]

def get_seller_price(seller):
    """
    Determine the selling price and whether tax is being evaded
    Returns: (price, is_evading)
    """
    base_price = random.choice(BASE_PRICES)
    if seller.role in ['bad', 'neutral']:
        if seller.cool_down > 0:
            return base_price, False  # During cooldown, don't evade taxes
        else:
            current_evade_rate = evade_rate_bad if seller.role == 'bad' else evade_rate_neutral
            if random.random() < current_evade_rate:
                return round(base_price * discount_rate, 1), True  # Evading with discounted price
            else:
                return base_price, False
    else:
        return base_price, False  # Good actors never evade

def buyer_decision(price):
    """Determine if a buyer will purchase at the given price"""
    return random.random() < min(2.1 / (price**1.3), 1.0)

def run_simulation():
    """Run a single simulation and return the results"""
    people = initialize_people()
    farmers = [p for p in people if p.occupation == 'farmer']
    herders = [p for p in people if p.occupation == 'herder']
    
    day_records = []  # Store daily statistics
    tax_revenue = {'tax': 0, 'fine': 0}  # Track tax and fine revenue

    for _ in range(DAYS):
        daily_stats = defaultdict(float)  # Initialize daily statistics
        tax_revenue = {'tax': 0, 'fine': 0}  # Reset daily tax revenue
        # Initialize role-based statistics
        role_volume = {'good': 0, 'bad': 0, 'neutral': 0}  # Sales volume by role
        role_count = {'good': 0, 'bad': 0, 'neutral': 0}   # Transaction count by role
        role_tax = {'good': 0, 'bad': 0, 'neutral': 0}     # Tax paid by role
        role_fine = {'bad': 0, 'neutral': 0}               # Fines paid by role
        
        # Calculate probability of secondary transactions based on previous day's revenue
        if day_records:
            prev_day = day_records[-1]
            p = (prev_day['tax'] + prev_day['fine'] - c) / 6
        else:
            p = 0.0
        j = 2 if random.random() < p else 1  # Number of transaction rounds (1 or 2)

        for _ in range(j):
            # Shuffle the order of transactions
            random.shuffle(farmers)
            random.shuffle(herders)
            
            # Pair farmers and herders for transactions
            for farmer, herder in zip(farmers, herders):
                # Farmer sells rice
                rice_price, evade = get_seller_price(farmer)
                if buyer_decision(rice_price):
                    # Update wealth
                    farmer.wealth += rice_price
                    herder.wealth -= rice_price
                    daily_stats['rice'] += 1
                    # Update role statistics
                    role_volume[farmer.role] += rice_price
                    role_count[farmer.role] += 1
                    
                    # Calculate and process taxes
                    tax_due = rice_price * TAX_RATE
                    if evade:
                        if random.random() < audit_rate:  # Audited and caught
                            tax_due = rice_price * TAX_RATE / discount_rate  # Original tax amount
                            fine = tax_due * FINE_RATE
                            farmer.wealth -= (tax_due + fine)
                            tax_revenue['tax'] += tax_due
                            tax_revenue['fine'] += fine
                            # Set cooldown period
                            farmer.cool_down = cool_day_bad if farmer.role == 'bad' else cool_day_neutral
                            role_tax[farmer.role] += tax_due
                            role_fine[farmer.role] += fine
                    else:
                        farmer.wealth -= tax_due
                        tax_revenue['tax'] += tax_due
                        role_tax[farmer.role] += tax_due

                # Herder sells meat
                meat_price, evade = get_seller_price(herder)
                if buyer_decision(meat_price):
                    herder.wealth += meat_price
                    farmer.wealth -= meat_price
                    daily_stats['meat'] += 1
                    # Update role statistics
                    role_volume[herder.role] += meat_price
                    role_count[herder.role] += 1
                    
                    # Calculate and process taxes
                    tax_due = meat_price * TAX_RATE
                    if evade:
                        if random.random() < audit_rate:  # Audited and caught
                            tax_due = rice_price * TAX_RATE / discount_rate  # Original tax amount
                            fine = tax_due * FINE_RATE
                            herder.wealth -= (tax_due + fine)
                            tax_revenue['tax'] += tax_due
                            tax_revenue['fine'] += fine
                            # Set cooldown period
                            herder.cool_down = cool_day_bad if herder.role == 'bad' else cool_day_neutral
                            role_tax[herder.role] += tax_due
                            role_fine[herder.role] += fine
                    else:
                        herder.wealth -= tax_due
                        tax_revenue['tax'] += tax_due
                        role_tax[herder.role] += tax_due

        # Record daily statistics
        daily_stats['tax'] = tax_revenue['tax']
        daily_stats['fine'] = tax_revenue['fine']
        # Add role-based statistics
        daily_stats['good_volume'] = role_volume['good']
        daily_stats['bad_volume'] = role_volume['bad']
        daily_stats['neutral_volume'] = role_volume['neutral']
        daily_stats['good_count'] = role_count['good']
        daily_stats['bad_count'] = role_count['bad']
        daily_stats['neutral_count'] = role_count['neutral']
        daily_stats['good_tax'] = role_tax['good']
        daily_stats['bad_tax'] = role_tax['bad']
        daily_stats['neutral_tax'] = role_tax['neutral']
        daily_stats['bad_fine'] = role_fine['bad']
        daily_stats['neutral_fine'] = role_fine['neutral']
        
        # Distribute tax revenue equally among all people
        total = tax_revenue['tax'] + tax_revenue['fine']
        per_capita = total / len(people)
        for p in people:
            p.wealth += per_capita
            if p.cool_down > 0:
                p.cool_down -= 1  # Reduce cooldown days
        daily_stats['prob'] = per_capita  # Record per capita distribution
        
        # Record asset data by role
        daily_stats.update({
            'good_asset': sum(p.wealth for p in people if p.role == 'good'),
            'bad_asset': sum(p.wealth for p in people if p.role == 'bad'),
            'neutral_asset': sum(p.wealth for p in people if p.role == 'neutral')
        })
        
        day_records.append(daily_stats)
    
    return day_records

def calculate_averages(all_results):
    """Calculate averages across all simulation runs"""
    averaged = [defaultdict(float) for _ in range(DAYS)]
    
    for day in range(DAYS):
        for key in all_results[0][day].keys():
            total = sum(run[day].get(key, 0) for run in all_results)
            averaged[day][key] = total / SIMULATION_TIMES
    
    return averaged

# Run multiple simulations
all_sim_results = []
for i in range(SIMULATION_TIMES):
    # print(f"Running simulation {i+1}/{SIMULATION_TIMES}...")
    all_sim_results.append(run_simulation())

# Calculate average results
averaged_data = calculate_averages(all_sim_results)

# Create visualization figures
fig = plt.figure(figsize=(15, 30))  # Larger figure size for more subplots

# Configure grid layout
gs = fig.add_gridspec(24, 2, 
                     width_ratios=[1, 1],  # Equal column widths
                     wspace=0.25)         # Horizontal spacing between subplots

# Assign subplot positions
ax_charts = [
    fig.add_subplot(gs[5:10, 0]),   # Transaction volume
    fig.add_subplot(gs[5:10, 1]),   # Transaction count
    fig.add_subplot(gs[11:16, 0]),  # Tax by role
    fig.add_subplot(gs[11:16, 1]),  # Asset by role
    fig.add_subplot(gs[17:22, 0]),  # Fines by role
    fig.add_subplot(gs[17:22, 1])   # Secondary transaction probability
]

# Chart configuration: (title, labels, data keys, subplot index, colors)
chart_config = [
    ('Transaction Volume', ['good', 'bad', 'neutral'], ['good_volume', 'bad_volume', 'neutral_volume'], 0, ['lightblue', 'purple', 'yellow']),
    ('Transaction Count', ['good', 'bad', 'neutral'], ['good_count', 'bad_count', 'neutral_count'], 1, ['lightblue', 'purple', 'yellow']),
    ('Tax', ['good', 'bad', 'neutral'], ['good_tax', 'bad_tax', 'neutral_tax'], 2, ['lightblue', 'purple', 'yellow']),
    ('Asset', ['good', 'bad', 'neutral'], ['good_asset', 'bad_asset', 'neutral_asset'], 3, ['lightblue', 'purple', 'yellow']),
    ('Fines Paid by Role', ['bad', 'neutral'], ['bad_fine', 'neutral_fine'], 4, ['purple', 'yellow']),
    ('Probability of secondary transactions', ['prob'], ['prob'], 5, ['gray'])
]

# Generate charts
for title, labels, keys, idx, colors in chart_config:
    ax = ax_charts[idx]
    ax.clear()
    ax.set_title(f"{title}", fontsize=12)
    if len(keys) > 1:
        for label, key, color in zip(labels, keys, colors):
            data = [d[key] for d in averaged_data]
            ax.plot(data, label=label, linewidth=0.5, color=color)
        # Place legend outside the chart
        ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1, 1))
    else:
        data = [d[keys[0]] for d in averaged_data]
        ax.plot(data, linewidth=1.5, color=colors[0])
    ax.grid(True, alpha=0.3)  # Add light grid lines

# Adjust layout to accommodate legends
plt.tight_layout(pad=3, h_pad=5, w_pad=5)  # Increase spacing between subplots
fig.subplots_adjust(left=0.05, right=0.92, top=1.23, bottom=-0.06)

# Save results to Excel
df = pd.DataFrame(averaged_data)
script_dir = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(script_dir, f'0.4cost_simulation_results{audit_rate:.1f}.xlsx')
df.to_excel(filename, index=False)

# Save visualization as image
image_filename = os.path.join(script_dir, f'0.4cost_simulation_results{audit_rate:.1f}.png')
plt.savefig(image_filename, dpi=300, bbox_inches='tight')

# Display the visualization
plt.show()