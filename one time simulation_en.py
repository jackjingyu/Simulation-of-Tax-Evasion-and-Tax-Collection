# Import necessary libraries
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import random
from matplotlib.patches import Rectangle
from collections import defaultdict
import pandas as pd
import os

# Set Chinese font for matplotlib
plt.rcParams['font.sans-serif'] = ['SimHei']  # Use SimHei font for Chinese characters
plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display issue

# Global parameter settings
DAYS = 200  # Total simulation days
TAX_RATE = 0.1  # Tax rate on transactions
FINE_RATE = 2  # Fine multiplier for tax evasion
BASE_PRICES = np.round(np.arange(2.0, 5.1, 0.5), 1)  # Base price range for goods
init_asset = 100  # Initial wealth for each person
cool_day_bad = 1  # Cooldown days for bad actors after being caught
cool_day_neutral = 2  # Cooldown days for neutral actors after being caught
evade_rate_bad = 0.8  # Probability bad actors will evade taxes
evade_rate_neutral = 0.5  # Probability neutral actors will evade taxes
audit_rate = 0.2  # Probability of auditing a transaction
discount_rate = 0.95  # Discount rate when evading taxes
c = 0.2  # Cost parameter for secondary transaction probability

# Define Person class to represent individuals in the simulation
class Person:
    def __init__(self, role, occupation, pid):
        self.role = role  # 'good', 'bad', or 'neutral'
        self.occupation = occupation  # 'farmer' or 'herder'
        self.pid = pid  # Person ID
        self.wealth = init_asset  # Current wealth
        self.position = 0  # Position in the visualization
        self.cool_down = 0  # Days remaining in cooldown after being caught

# Initialize 6 people with different roles and occupations
people = [
    Person('good', 'farmer', 0),
    Person('bad', 'farmer', 1),
    Person('neutral', 'farmer', 2),
    Person('good', 'herder', 3),
    Person('bad', 'herder', 4),
    Person('neutral', 'herder', 5)
]

# Separate people into farmers and herders for easier access
farmers = [p for p in people if p.occupation == 'farmer']
herders = [p for p in people if p.occupation == 'herder']

# Initialize tax revenue tracking and daily records
tax_revenue = {'tax': 0, 'fine': 0}  # Track taxes and fines separately
day_records = []  # Store daily statistics

# Create figure and subplot layout
fig = plt.figure(figsize=(24, 18))  # Large figure size
gs = fig.add_gridspec(17, 2)  # Grid layout with 17 rows, 2 columns

# Main simulation visualization area (top 4 rows spanning both columns)
ax_sim = fig.add_subplot(gs[0:4, :])

# Create 6 chart areas for different statistics
ax_charts = [
    fig.add_subplot(gs[5:8, 0]),  # Chart 1: Transaction volume by role
    fig.add_subplot(gs[5:8, 1]),  # Chart 2: Transaction count by role
    fig.add_subplot(gs[9:12, 0]),  # Chart 3: Taxes paid by role
    fig.add_subplot(gs[9:12, 1]),  # Chart 4: Assets by role
    fig.add_subplot(gs[13:16, 0]),  # Chart 5: Fines paid by role
    fig.add_subplot(gs[13:16, 1])  # Chart 6: Secondary transaction probability
]

# Create visual elements for the simulation scene
market_patch = Rectangle((-0.3, 0.2), 0.6, 0.6, fc='lightgray')  # Market area
road_left = Rectangle((-1, 0.2), 0.4, 0.6, fc='green')  # Farmer side
road_right = Rectangle((0.6, 0.2), 0.4, 0.6, fc='red')  # Herder side

def init_scene():
    """Initialize the simulation visualization scene"""
    ax_sim.clear()
    ax_sim.set_xlim(-1, 1)
    ax_sim.set_ylim(0, 1)
    ax_sim.axis('off')  # Hide axes
    
    # Add visual elements
    ax_sim.add_patch(market_patch)
    ax_sim.add_patch(road_left)
    ax_sim.add_patch(road_right)
    
    # Add labels
    ax_sim.text(0, 0.5, 'market', ha='center', va='center', fontsize=24)
    ax_sim.text(-0.95, 0.45, 'farmer', fontsize=18)
    ax_sim.text(0.8, 0.45, 'herder', fontsize=18)
    return ax_sim,

def get_seller_price(seller):
    """
    Determine the selling price for a seller, considering tax evasion
    
    Args:
        seller: Person object representing the seller
        
    Returns:
        tuple: (price, evade_flag) where price is the selling price and 
               evade_flag indicates if tax is being evaded
    """
    base_price = random.choice(BASE_PRICES)  # Random base price
    
    # Check if seller might evade taxes
    if seller.role in ['bad', 'neutral']:
        if seller.cool_down > 0:  # In cooldown period after being caught
            return base_price, False  # Don't evade during cooldown
        else:
            # Determine evasion probability based on role
            current_evade_rate = evade_rate_bad if seller.role == 'bad' else evade_rate_neutral
            if random.random() < current_evade_rate:
                # Return discounted price when evading
                return round(base_price * discount_rate, 1), True  
            else:
                return base_price, False  # Not evading
    else:
        return base_price, False  # Good sellers never evade

def buyer_decision(price):
    """
    Determine if a buyer will purchase at the given price
    
    Args:
        price: The asking price
        
    Returns:
        bool: True if buyer accepts, False otherwise
    """
    # Probability decreases as price increases
    return random.random() < min(2.1 / (price**1.3), 1.0)

def trading_round():
    """
    Simulate one round of trading between farmers and herders
    
    Returns:
        dict: Daily statistics including transactions, taxes, and fines
    """
    daily_stats = defaultdict(float)
    
    # Reset daily tax revenue
    global tax_revenue
    tax_revenue = {'tax': 0, 'fine': 0}
    
    # Initialize counters
    daily_stats.update({
        'good_count': 0, 
        'bad_count': 0, 
        'neutral_count': 0,
        'good_tax': 0,
        'bad_tax': 0,
        'neutral_tax': 0
    })

    # Calculate probability of secondary transactions based on previous day's revenue
    if day_records:
        prev_day = day_records[-1]
        p = (prev_day['tax'] + prev_day['fine'] - c) / 6
    else:
        p = 0.0
    
    # Determine number of trading rounds (1 or 2)
    j = 2 if random.random() < p else 1

    for _ in range(j):
        # Shuffle order of trading pairs
        random.shuffle(farmers)
        random.shuffle(herders)
        
        # Pair farmers with herders
        for farmer, herder in zip(farmers, herders):
            # Farmer sells rice
            rice_price, evade = get_seller_price(farmer)
            if buyer_decision(rice_price):
                # Update wealth
                farmer.wealth += rice_price
                herder.wealth -= rice_price
                daily_stats['rice'] += 1
                
                # Record transaction by role
                daily_stats[f'{farmer.role}_volume'] += rice_price
                daily_stats[f'{farmer.role}_count'] += 1
                
                # Calculate tax due
                tax_due = rice_price * TAX_RATE
                
                if evade:
                    if random.random() < audit_rate:  # Audited and caught
                        # Calculate actual tax due (without discount)
                        tax_due = rice_price * TAX_RATE / discount_rate
                        fine = tax_due * FINE_RATE
                        
                        # Apply penalty
                        farmer.wealth -= (tax_due + fine)
                        tax_revenue['tax'] += tax_due
                        tax_revenue['fine'] += fine
                        
                        # Set cooldown period
                        farmer.cool_down = cool_day_bad if farmer.role == 'bad' else cool_day_neutral
                        
                        # Record taxes and fines
                        daily_stats[f'{farmer.role}_tax'] += tax_due
                        daily_stats[f'{farmer.role}_fine'] += fine
                else:
                    # Normal tax payment
                    farmer.wealth -= tax_due
                    tax_revenue['tax'] += tax_due
                    daily_stats[f'{farmer.role}_tax'] += tax_due

            # Herder sells meat (same logic as above)
            meat_price, evade = get_seller_price(herder)
            if buyer_decision(meat_price):
                herder.wealth += meat_price
                farmer.wealth -= meat_price
                daily_stats['meat'] += 1
                daily_stats[f'{herder.role}_volume'] += meat_price
                daily_stats[f'{herder.role}_count'] += 1
                
                tax_due = meat_price * TAX_RATE
                if evade:
                    if random.random() < audit_rate:
                        tax_due = rice_price * TAX_RATE / discount_rate
                        fine = tax_due * FINE_RATE
                        herder.wealth -= (tax_due + fine)
                        tax_revenue['tax'] += tax_due
                        tax_revenue['fine'] += fine
                        herder.cool_down = cool_day_bad if herder.role == 'bad' else cool_day_neutral
                        daily_stats[f'{herder.role}_tax'] += tax_due
                        daily_stats[f'{herder.role}_fine'] += fine
                else:
                    herder.wealth -= tax_due
                    tax_revenue['tax'] += tax_due
                    daily_stats[f'{herder.role}_tax'] += tax_due

    # Record final tax and fine amounts
    daily_stats['tax'] = tax_revenue['tax']
    daily_stats['fine'] = tax_revenue['fine']
    
    # Calculate total transaction volume
    daily_stats['total_volume'] = daily_stats['good_volume'] + daily_stats['bad_volume'] + daily_stats['neutral_volume']
    
    return daily_stats

def distribute_round():
    """
    Distribute collected taxes and fines equally among all people
    and update cooldown counters
    
    Returns:
        float: The per capita distribution amount
    """
    total = tax_revenue['tax'] + tax_revenue['fine']
    per_capita = total / len(people)
    
    # Distribute wealth and update cooldowns
    for p in people:
        p.wealth += per_capita
        if p.cool_down > 0:
            p.cool_down -= 1
    
    return total / 6  # Return per capita amount for tracking

def update_charts():
    """
    Update all statistical charts with latest data
    """
    # Configuration for each chart (title, labels, data keys, chart index, colors)
    chart_config = [
        # Chart 1: Transaction volume by role
        ('transaction volume', ['good', 'bad', 'neutral'], ['good_volume', 'bad_volume', 'neutral_volume'], 0, ['lightblue', 'purple', 'yellow']),
        # Chart 2: Transaction count by role
        ('transaction count', ['good', 'bad', 'neutral'], ['good_count', 'bad_count', 'neutral_count'], 1, ['lightblue', 'purple', 'yellow']),
        # Chart 3: Taxes paid by role
        ('tax', ['good', 'bad', 'neutral'], ['good_tax', 'bad_tax', 'neutral_tax'], 2, ['lightblue', 'purple', 'yellow']),
        # Chart 4: Assets by role
        ('asset by role', ['good', 'bad', 'neutral'], ['good', 'bad', 'neutral'], 3, ['lightblue', 'purple', 'yellow']),
        # Chart 5: Fines paid by bad and neutral actors
        ('Fines', ['bad', 'neutral'], ['bad_fine', 'neutral_fine'], 4, ['purple', 'yellow']),
        # Chart 6: Secondary transaction probability
        ('Probability of secondary transactions', ['prob'], ['prob'], 5, ['black'])
    ]

    # Update each chart
    for title, labels, keys, idx, colors in chart_config:
        ax = ax_charts[idx]
        ax.clear()
        ax.set_title(title, fontsize=12)
        
        if len(keys) > 1:  # For charts with multiple lines
            for label, key, color in zip(labels, keys, colors):
                data = [d.get(key, 0) for d in day_records]
                ax.plot(data, label=label, linewidth=0.5, color=color)
            ax.legend(fontsize=8, 
                    loc='upper left', 
                    bbox_to_anchor=(1.02, 1),
                    borderaxespad=0.)
        else:  # For single-line charts
            data = [d.get(keys[0], 0) for d in day_records]
            ax.plot(data, linewidth=1.5, color=colors[0])
        
        ax.grid(True, alpha=0.3)  # Add light grid
    
    # Adjust layout
    plt.tight_layout()
    fig.subplots_adjust(right=0.9)

def save_to_excel():
    """
    Save simulation results to an Excel file
    """
    # Convert records to DataFrame
    df = pd.DataFrame(day_records)
    
    # Add additional calculated columns
    df['total_tax'] = df['tax'] + df['fine']
    df['avg_wealth'] = (df['good'] + df['bad'] + df['neutral']) / 6
    
    # Save to Excel in the same directory as the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(script_dir, f"tax_simulation_results_{DAYS}days.xlsx")
    df.to_excel(filename, index_label='Day')
    print(f"Simulation data saved to {filename}")

def update(frame):
    """
    Update function for animation
    
    Args:
        frame: Current animation frame number
        
    Returns:
        tuple: Updated artists for animation
    """
    current_day = frame // 2  # Each day has two phases (trading and distribution)
    phase = frame % 2  # 0 = trading phase, 1 = distribution phase
    
    # Initialize scene
    init_scene()
    
    # Position mapping for visualization
    role_y = {'good': 0.7, 'bad': 0.5, 'neutral': 0.3}  
    
    # Draw people in their positions
    for p in people:
        display_pos = 2 if phase == 0 else 0  # Position index for animation
        # Calculate x position based on occupation
        x = -0.55 + 0.3 * display_pos if p.occupation == 'farmer' else 0.55 - 0.3 * display_pos
        y = role_y[p.role]
        # Set color based on role
        color = {'good':'lightblue', 'bad':'purple', 'neutral':'yellow'}[p.role]
        ax_sim.scatter(x, y, s=200, c=color, edgecolor='black')
    
    # Perform trading and distribution on appropriate frames
    if phase == 0 and current_day >= len(day_records):
        # Trading phase for new day
        daily_stats = trading_round()
        # Distribution phase
        daily_stats['prob'] = distribute_round()
        
        # Record wealth statistics by role and occupation
        daily_stats.update({
            'farmer': sum(p.wealth for p in people if p.occupation == 'farmer'),
            'herder': sum(p.wealth for p in people if p.occupation == 'herder'),
            'good': sum(p.wealth for p in people if p.role == 'good'),
            'bad': sum(p.wealth for p in people if p.role == 'bad'),
            'neutral': sum(p.wealth for p in people if p.role == 'neutral')
        })
        day_records.append(daily_stats)
    
    # Update charts if we have data
    if day_records:
        update_charts()
    
    return ax_sim,

# Create animation
ani = animation.FuncAnimation(
    fig, 
    update, 
    frames=DAYS*2,  # Two frames per day
    init_func=init_scene,
    interval=1,
    blit=False,
    repeat=False,
    cache_frame_data=False
)

# Show the animation
plt.show()

# After animation completes, save data to Excel
save_to_excel()

# Note: The commented-out run_simulation() function at the bottom provides
# an alternative non-animated version of the simulation that runs all days
# at once and shows final results. It's currently disabled in favor of
# the animated version.