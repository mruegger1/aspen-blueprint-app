# Aspen Comparative Property Finder

A powerful tool for finding and analyzing comparable properties in the Aspen real estate market. This tool uses advanced filtering logic to match properties with similar characteristics and provides comprehensive analysis of market comparisons.

## Features

- **Address-Based Search**: Find properties by address with fuzzy matching
- **Advanced Filtering**: Match properties based on key value drivers:
  - Size (square footage)
  - Bedrooms and bathrooms
  - Price per square foot
  - Property condition
  - Demo score
  - Short-term rental eligibility
  - TDR (Transferable Development Rights) eligibility
- **Intelligent Match Scoring**: Properties are scored based on multiple factors
- **Adaptive Filtering**: Automatically relaxes criteria when too few matches are found
- **Comprehensive Analysis**: Detailed statistics about comparable properties
- **Market Comparison**: Compare selected properties to the broader market
- **Export Results**: Save analysis to CSV for further processing

## Installation

### Prerequisites

- Python 3.6+
- pandas
- numpy
- fuzzywuzzy

### Setup

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/aspen-comp-finder.git
   cd aspen-comp-finder
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Interactive Menu Interface

The easiest way to use the tool is through the interactive menu interface:

```bash
python run_comp_menu.py
```

This will:
1. Guide you through entering a property address
2. Allow you to customize analysis parameters
3. Generate a detailed report of comparable properties
4. Save results to an outputs directory

### Command Line with Direct Address

Run the tool with a specific address from the command line:

```bash
python run_comp_menu.py --address "1227 Mountain View Dr, Aspen"
```

Additional options:
- `--limit` - Maximum number of comps to return (default: 5)
- `--min-comps` - Minimum number of comps to try to find (default: 3)
- `--similarity` - Similarity threshold for matching (default: 0.7)
- `--price-diff` - Maximum price difference percentage (default: 0.35)
- `--sqft-diff` - Maximum square footage difference (default: 500)
- `--output-dir` - Directory for output files (default: 'outputs')

Example with all options:
```bash
python run_comp_menu.py --address "1227 Mountain View Dr, Aspen" --limit 10 --min-comps 5 --similarity 0.6 --price-diff 0.4 --sqft-diff 750 --output-dir "my_analysis"
```

### Traditional Command Line Interface

The original command line interface is still available:

```bash
python run_comp_finder.py --address "1227 Mountain View Dr, Aspen"
```

### API Usage

```python
from aspen_comp_finder import EnhancedCompFinder, run_comp_analysis_by_address

# Quick analysis by address
result = run_comp_analysis_by_address(
    address="1227 Mountain View Dr, Aspen",
    min_comps=3,
    similarity_threshold=0.7,
    export_dir="outputs"
)

# Or create a finder instance for more control
finder = EnhancedCompFinder()
result = finder.run_comp_analysis_by_address(
    address_query="1227 Mountain View Dr", 
    limit=5,
    min_comps=3,
    similarity_threshold=0.8
)
```

## Output

The tool outputs a detailed analysis including:
- Subject property details
- Comparable properties and their match scores
- Price statistics (mean, median, range)
- Property condition distribution
- Demo score statistics
- Value-add type distribution
- Market comparison metrics

Results are also exported to a CSV file for further analysis, typically in the `outputs` directory.

## Filter Relaxation

When strict filtering criteria yield too few comparable properties, the tool automatically relaxes the filters in stages:

1. Original criteria
2. Widen numeric ranges by 25%
3. Widen numeric ranges by 50% and relax condition requirements
4. Widen numeric ranges by 100% and relax condition & STR eligibility
5. Extreme relaxation (last resort)

The tool logs which filters were relaxed to ensure transparency in the analysis.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

# Aspen Comp Finder

A modular system for finding comparable real estate properties in Aspen.

## Project Structure

```
aspen_comp_finder/
├── src/
│   └── aspen_comp_finder/
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   └── data_loader.py         # Loads and preprocesses CSV data
│       ├── pricing/
│       │   ├── __init__.py
│       │   └── price_adjuster.py      # Applies quarterly price normalization
│       ├── premium/
│       │   ├── __init__.py
│       │   └── premium_calculator.py  # Applies building and street premium logic
│       ├── scoring/
│       │   ├── __init__.py
│       │   └── scoring_engine.py      # Modular comp scoring logic
│       ├── filters/
│       │   ├── __init__.py
│       │   └── filters.py             # Basic filters and progressive relaxation
│       └── classic_finder.py          # Orchestrates all components via ClassicCompFinder
│
├── scripts/
│   ├── run_comp_finder.py             # CLI runner for finding comps
│   └── run_comp_menu.py               # Interactive CLI tool (optional)
│
├── config/
│   ├── weights.json                   # Custom scoring weights
│   └── premiums.json                  # Building and street premiums
│
├── data/
│   └── aspen_mvp_final_scored.csv     # Main property dataset
│
├── tests/
│   └── test_scoring_engine.py (etc)   # Unit tests
│
├── outputs/                           # Generated comp reports
├── README.md
└── requirements.txt
```

## Features

- **Modular Architecture**: Each component is designed to be reusable and independently maintainable
- **Building Premiums**: Accounts for premium differences between buildings
- **Street Premiums**: Adjusts for location desirability at the street level
- **Quarterly Price Adjustments**: Normalizes prices based on market appreciation over time
- **Progressive Filter Relaxation**: Automatically relaxes filters to find minimum number of comps
- **Configurable Scoring Weights**: Customizable importance of different matching criteria
- **Export Capabilities**: Save comp results to CSV files for further analysis

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/aspen_comp_finder.git
   cd aspen_comp_finder
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Command Line Interface

You can run the comp finder using the CLI tool:

```bash
python scripts/run_comp_finder.py --bedrooms 2 --bathrooms 2 --property-type "Condo" --area "Core" --str-eligible
```

#### Common Options

- `--bedrooms` - Number of bedrooms
- `--bathrooms` - Number of bathrooms
- `--property-type` - Type of property (Condo, Townhouse, etc.)
- `--area` - Neighborhood or area
--str-eligible - Property is eligible for short-term rentals
--condition - Property condition (Excellent, Good, Average, Fair, Poor)
--max-price - Maximum price to consider
--sqft-min - Minimum square footage
--sqft-max - Maximum square footage
--months-back - How many months of sales history to consider (default: 24)
--listing-status - Filter by listing status (A=Active, P=Pending, S=Sold)
--reference-property - Find comps similar to a specific property address
--limit - Maximum number of comps to return (default: 5)
--min-comps - Minimum number of comps to try to find (default: 3)

Python API
You can also use the Aspen Comp Finder as a library in your Python code:
from aspen_comp_finder import ClassicCompFinder

# Initialize comp finder
finder = ClassicCompFinder(csv_path="path/to/data.csv", time_adjust=True)

# Find comps
results = finder.find_classic_comps(
    bedrooms=2,
    bathrooms=2,
    property_type="Condo",
    area="Core",
    str_eligible=True,
    months_back=24,
    limit=5
)

# Access results
comps = results["comps"]
stats = results["stats"]
subject = results["subject"]

# Print top comps
for _, comp in comps.iterrows():
    print(f"{comp['full_address']} - Match Score: {comp['match_score']:.1f}")
