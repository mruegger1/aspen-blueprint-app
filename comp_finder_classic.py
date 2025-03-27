#!/usr/bin/env python3
"""
Classic Comp Finder - Traditional Comparable Property Matcher
------------------------------------------------------------
Enhanced version with building premiums, street premiums and quarterly adjustments.
Finds comparable properties based on standard buyer/appraiser criteria.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import re

# Set up logging
logger = logging.getLogger("comp_finder_classic")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Building premium dictionary
building_premiums = {
    "Monarch on the park": 0.604,
    "DurCondo": 0.48,
    "MtQueen": 0.424,
    "DurMews": 0.343,
    "TwnSteAs": 0.267,
    "NorthNell": 0.172,
    "DerBerghof": 0.146,
    "Dolomite": 0.128,
    "Aspen Alps": 0.106,
    "Aspen Square": 0.085,
    "ObermeyerPlace": 0.068,
    "FifthAve": 0.063,
    "ShadowMtn": 0.052,
    "210Cooper": 0.04,
    "SouthPt": 0.029
}

# Street premium dictionary
street_premiums = {
    "galena": 0.271,
    "monarch": 0.186,
    "durant": 0.032,
    "mill": 0.025,
    "cooper": 0.02,
    "hyman": -0.016,
    "aspen": -0.059,
    "hopkins": -0.075,
    "main": -0.203
}

# Quarterly appreciation rates (format: YYYYQN -> appreciation rate)
quarterly_appreciation = {
    "2021Q1": 0.032,
    "2021Q2": 0.045,
    "2021Q3": 0.051,
    "2021Q4": 0.048,
    "2022Q1": 0.037,
    "2022Q2": 0.029,
    "2022Q3": 0.025,
    "2022Q4": 0.023,
    "2023Q1": 0.023,
    "2023Q2": 0.020,
    "2023Q3": 0.018,
    "2023Q4": 0.020,
    "2024Q1": 0.015,
    "2024Q2": 0.012,
    "2024Q3": 0.016,
    "2024Q4": 0.014,
    "2025Q1": 0.000  # Current quarter - no adjustment
}

class ClassicCompFinder:
    """Enhanced version of the Classic Comp Finder with premium features."""
    
    def __init__(self, csv_path=None, time_adjust=True):
        """
        Initialize the classic comp finder.
        
        Args:
            csv_path (str): Path to CSV file with property data
            time_adjust (bool): Whether to adjust prices for time
        """
        # Load data
        self.data, self.file_path = self._load_data(csv_path)
        self.time_adjust = time_adjust
        
        # Apply adjustments
        if time_adjust:
            self.data = self._adjust_prices_for_time(self.data)
        
        # Apply premium adjustments
        self.data = self._apply_building_premium(self.data)
        self.data = self._apply_street_premium(self.data)
    
    def _load_data(self, csv_path=None):
        """
        Load property data from CSV file.
        
        Args:
            csv_path (str): Path to CSV file
            
        Returns:
            tuple: (DataFrame of properties, path to file)
        """
        # Define potential CSV file locations
        potential_paths = []
        
        # Add the provided path if it exists
        if csv_path and os.path.exists(csv_path):
            potential_paths.append(csv_path)
        
        # Add default locations
        default_paths = [
            "aspen_mvp_final_scored.csv",
            "data/aspen_mvp_final_scored.csv",
            "../../Python Analysis/Data/mvp/aspen_mvp_final_scored.csv",
            "../data/aspen_mvp_final_scored.csv",
            os.path.expanduser("~/Desktop/Aspen Real Estate Analysis Master/Python Analysis/Data/mvp/aspen_mvp_final_scored.csv")
        ]
        
        for path in default_paths:
            if os.path.exists(path):
                potential_paths.append(path)
        
        # Try to load from the first available path
        if not potential_paths:
            raise FileNotFoundError("Could not find property data CSV file. Please provide a valid path.")
        
        # Load the first available file
        file_path = potential_paths[0]
        logger.info(f"Loading data from: {file_path}")
        
        try:
            data = pd.read_csv(file_path)
            logger.info(f"Loaded {len(data)} properties from CSV")
            
            # Ensure critical columns exist
            required_cols = ['bedrooms', 'total_baths', 'adjusted_sold_price', 'total_sqft', 'sold_date']
            missing_cols = [col for col in required_cols if col not in data.columns]
            
            if missing_cols:
                logger.warning(f"Missing required columns: {', '.join(missing_cols)}")
            
            # Convert date columns
            date_columns = ['sold_date', 'list_date', 'close_date']
            for col in date_columns:
                if col in data.columns:
                    data[f"{col}_dt"] = pd.to_datetime(data[col], errors="coerce")
            
            return data, file_path
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def _get_quarter_from_date(self, date):
        """
        Extract quarter identifier from a date (format: YYYYQN).
        
        Args:
            date (datetime): Date to extract quarter from
            
        Returns:
            str: Quarter identifier (e.g., "2023Q2")
        """
        year = date.year
        quarter = (date.month - 1) // 3 + 1
        return f"{year}Q{quarter}"
    
    def _adjust_prices_for_time(self, df):
        """
        Apply quarterly time-based price adjustments to account for market appreciation/depreciation.
        
        Args:
            df (DataFrame): Property data
            
        Returns:
            DataFrame: Data with adjusted prices
        """
        logger.info("=== Price Adjustment For Time (Quarterly) ===")
        
        # Create adjusted price columns if they don't exist
        if 'adjusted_sold_price_time' not in df.columns:
            df['adjusted_sold_price_time'] = df['adjusted_sold_price'].copy()
        
        if 'adjusted_price_per_sqft_time' not in df.columns:
            # Make sure we have price_per_sqft column
            if 'price_per_sqft' not in df.columns and 'total_sqft' in df.columns:
                df['price_per_sqft'] = df['adjusted_sold_price'] / df['total_sqft']
            
            df['adjusted_price_per_sqft_time'] = df['price_per_sqft'].copy()
        
        # Get current quarter
        current_date = datetime.now()
        current_quarter = self._get_quarter_from_date(current_date)
        
        logger.info(f"Current quarter: {current_quarter}")
        logger.info("Applying quarterly appreciation rates to normalize prices")
        
        # Process each row
        for idx, row in df.iterrows():
            if pd.notna(row.get('sold_date_dt')):
                sale_date = row['sold_date_dt']
                sale_quarter = self._get_quarter_from_date(sale_date)
                
                # Skip adjustment if already in current quarter
                if sale_quarter == current_quarter:
                    continue
                
                # Calculate cumulative adjustment factor
                adjustment_factor = 1.0
                
                # Get all quarters between sale quarter and current quarter
                quarters = []
                year = sale_date.year
                q = (sale_date.month - 1) // 3 + 1
                
                while f"{year}Q{q}" != current_quarter:
                    q += 1
                    if q > 4:
                        q = 1
                        year += 1
                    quarterly_key = f"{year}Q{q}"
                    quarters.append(quarterly_key)
                
                # Apply appreciation for each quarter
                for q in quarters:
                    if q in quarterly_appreciation:
                        adjustment_factor *= (1 + quarterly_appreciation[q])
                    else:
                        # If quarter not found, use default appreciation
                        adjustment_factor *= 1.01  # 1% default quarterly appreciation
                
                # Apply adjustment
                df.at[idx, 'adjusted_sold_price_time'] = row['adjusted_sold_price'] * adjustment_factor
                if 'price_per_sqft' in df.columns and pd.notna(row['price_per_sqft']):
                    df.at[idx, 'adjusted_price_per_sqft_time'] = row['price_per_sqft'] * adjustment_factor
                
                # Add quarter info for reference
                df.at[idx, 'sale_quarter'] = sale_quarter
                df.at[idx, 'quarters_adjustment'] = adjustment_factor
        
        return df
    
    def _extract_street_name(self, address):
        """
        Extract street name from a property address.
        
        Args:
            address (str): Property address
            
        Returns:
            str: Extracted street name or empty string if not found
        """
        if not address or not isinstance(address, str):
            return ""
        
        # Try to extract street name
        # Common patterns: "123 Main St", "123 N Main St"
        address = address.lower().strip()
        
        # Remove any unit numbers
        address = re.sub(r'\bunit\s+\w+\b', '', address)
        address = re.sub(r'\bapt\s+\w+\b', '', address)
        address = re.sub(r'\b#\s*\w+\b', '', address)
        
        # Extract parts
        parts = address.split()
        
        # Skip number and directions
        directions = ['n', 's', 'e', 'w', 'north', 'south', 'east', 'west']
        start_idx = 0
        
        # Skip the number
        if len(parts) > 0 and parts[0].isdigit():
            start_idx = 1
        
        # Skip direction if present
        if len(parts) > start_idx and parts[start_idx].lower() in directions:
            start_idx += 1
        
        # The next part should be the street name
        if len(parts) > start_idx:
            return parts[start_idx].lower()
        
        return ""
    
    def _apply_building_premium(self, df):
        """
        Apply building premium adjustments to property data.
        
        Args:
            df (DataFrame): Property data
            
        Returns:
            DataFrame: Data with building premium column
        """
        logger.info("=== Applying Building Premium Adjustments ===")
        
        # Create building premium column
        df['building_premium'] = 0.0
        
        # Check if sub_loc column exists
        if 'sub_loc' not in df.columns:
            logger.warning("sub_loc column not found, skipping building premium adjustments")
            return df
        
        # Map building premiums
        for idx, row in df.iterrows():
            if pd.notna(row['sub_loc']):
                sub_loc = row['sub_loc'].strip()
                
                # Try exact match first
                if sub_loc in building_premiums:
                    df.at[idx, 'building_premium'] = building_premiums[sub_loc]
                else:
                    # Try fuzzy match
                    for building, premium in building_premiums.items():
                        if building.lower() in sub_loc.lower() or sub_loc.lower() in building.lower():
                            df.at[idx, 'building_premium'] = premium
                            break
        
        # Log buildings found
        buildings_found = df[df['building_premium'] > 0]['sub_loc'].unique()
        logger.info(f"Applied building premiums to {len(buildings_found)} buildings")
        
        return df
    
    def _apply_street_premium(self, df):
        """
        Apply street premium adjustments to property data.
        
        Args:
            df (DataFrame): Property data
            
        Returns:
            DataFrame: Data with street premium column
        """
        logger.info("=== Applying Street Premium Adjustments ===")
        
        # Create street premium column
        df['street_premium'] = 0.0
        
        # Extract street names from addresses
        address_col = None
        for col in ['full_address', 'address', 'street_address', 'location']:
            if col in df.columns:
                address_col = col
                break
        
        if not address_col:
            logger.warning("No address column found, skipping street premium adjustments")
            return df
        
        # Extract street names and apply premiums
        df['extracted_street'] = df[address_col].apply(self._extract_street_name)
        
        # Apply street premiums
        for idx, row in df.iterrows():
            if pd.notna(row['extracted_street']) and row['extracted_street']:
                street = row['extracted_street'].lower()
                
                # Try exact match first
                if street in street_premiums:
                    df.at[idx, 'street_premium'] = street_premiums[street]
                else:
                    # Try fuzzy match
                    for premium_street, premium in street_premiums.items():
                        if premium_street in street or street in premium_street:
                            df.at[idx, 'street_premium'] = premium
                            break
        
        # Log streets found
        streets_found = df[df['street_premium'] != 0]['extracted_street'].unique()
        logger.info(f"Applied street premiums to {len(streets_found)} streets")
        
        return df
    
    def _score_comps(self, subject, comps, weights=None):
        """
        Score comps based on similarity to subject property using weighted criteria.
        
        Args:
            subject (Series): Subject property
            comps (DataFrame): Potential comparable properties
            weights (dict): Scoring weights for different criteria
            
        Returns:
            DataFrame: Comps with match scores
        """
        # Define default weights if not provided
        if weights is None:
            weights = {
                'bedrooms': 2.0,
                'bathrooms': 2.0,
                'property_type': 2.0,
                'area': 1.5,
                'condition': 1.5,
                'sqft': 1.0,
                'str': 1.0,
                'recency': 1.0,
                'building_premium': 1.2,
                'street_premium': 0.8
            }
        
        # Create a copy of comps to work with
        df = comps.copy()
        df['match_score'] = 0.0
        
        # 1. Score bedrooms match
        if 'bedrooms' in subject and 'bedrooms' in df.columns:
            bedrooms = subject['bedrooms']
            df['bedroom_score'] = 1.0 - (0.25 * np.abs(df['bedrooms'] - bedrooms))
            df['bedroom_score'] = df['bedroom_score'].clip(0, 1)
            df['match_score'] += weights['bedrooms'] * df['bedroom_score']
        
        # 2. Score bathrooms match
        if 'total_baths' in subject and 'total_baths' in df.columns:
            bathrooms = subject['total_baths']
            df['bathroom_score'] = 1.0 - (0.25 * np.abs(df['total_baths'] - bathrooms))
            df['bathroom_score'] = df['bathroom_score'].clip(0, 1)
            df['match_score'] += weights['bathrooms'] * df['bathroom_score']
        
        # 3. Score property type match
        if 'resolved_property_type' in subject and 'resolved_property_type' in df.columns:
            property_type = subject['resolved_property_type']
            
            # Property type map
            property_type_map = {
                'Condo': ['Condo', 'Condominium', 'Apartment'],
                'Townhouse': ['Townhouse', 'Townhome', 'Town House', 'Townhome/Condo'],
                'Single Family': ['Single Family', 'House', 'Detached', 'Single-Family']
            }
            
            # Normalize property type
            normalized_type = None
            for category, variants in property_type_map.items():
                if property_type.lower() in [v.lower() for v in variants]:
                    normalized_type = category
                    break
            
            if normalized_type:
                df['property_type_score'] = 0.0
                for category, variants in property_type_map.items():
                    match_score = 1.0 if category == normalized_type else 0.3
                    mask = df['resolved_property_type'].str.lower().isin([v.lower() for v in variants])
                    df.loc[mask, 'property_type_score'] = match_score
            else:
                # Direct comparison
                df['property_type_score'] = np.where(
                    df['resolved_property_type'].str.lower() == property_type.lower(), 
                    1.0, 
                    0.0
                )
            
            df['match_score'] += weights['property_type'] * df['property_type_score']
        
        # 4. Score area match
        if 'area' in subject and 'area' in df.columns:
            area = subject['area']
            
            # Area proximity map
            area_proximity = {
                'Core': ['Core', 'Downtown', 'Central Core'],
                'West End': ['West End', 'West Side'],
                'East End': ['East End', 'East Side'],
                'Red Mountain': ['Red Mountain', 'Red Mtn'],
                'Smuggler': ['Smuggler', 'Smuggler Mountain'],
                'McLain Flats': ['McLain Flats', 'McLain'],
                'Woody Creek': ['Woody Creek'],
                'Starwood': ['Starwood']
            }
            
            # Adjacent areas
            adjacent_areas = {
                'Core': ['West End', 'East End'],
                'West End': ['Core', 'Red Mountain'],
                'East End': ['Core', 'Smuggler'],
                'Red Mountain': ['West End'],
                'Smuggler': ['East End']
            }
            
            # Normalize area
            normalized_area = None
            for category, variants in area_proximity.items():
                if area.lower() in [v.lower() for v in variants]:
                    normalized_area = category
                    break
            
            if normalized_area:
                df['area_score'] = 0.0
                for category, variants in area_proximity.items():
                    # Exact match gets 1.0
                    match_score = 1.0 if category == normalized_area else 0.0
                    
                    # Adjacent areas get 0.7
                    if category != normalized_area and normalized_area in adjacent_areas and category in adjacent_areas[normalized_area]:
                        match_score = 0.7
                    
                    mask = df['area'].str.lower().isin([v.lower() for v in variants])
                    df.loc[mask, 'area_score'] = match_score
            else:
                # Direct comparison
                df['area_score'] = np.where(
                    df['area'].str.lower() == area.lower(), 
                    1.0, 
                    0.0
                )
            
            df['match_score'] += weights['area'] * df['area_score']
        
        # 5. Score condition match
        if 'improved_condition' in subject and 'improved_condition' in df.columns:
            condition = subject['improved_condition']
            condition_values = ['Excellent', 'Good', 'Average', 'Fair', 'Poor']
            
            # Find index of target condition
            if condition in condition_values:
                target_idx = condition_values.index(condition)
                
                # Score based on distance in condition scale
                df['condition_score'] = 0.0
                for idx, row in df.iterrows():
                    if pd.isna(row['improved_condition']) or row['improved_condition'] not in condition_values:
                        df.at[idx, 'condition_score'] = 0.5  # Default for unknown
                    else:
                        prop_idx = condition_values.index(row['improved_condition'])
                        distance = abs(target_idx - prop_idx)
                        
                        # Score decreases with distance
                        if distance == 0:
                            df.at[idx, 'condition_score'] = 1.0
                        elif distance == 1:
                            df.at[idx, 'condition_score'] = 0.8
                        else:
                            df.at[idx, 'condition_score'] = max(0, 1.0 - (distance * 0.3))
            else:
                # Direct comparison
                df['condition_score'] = np.where(
                    df['improved_condition'] == condition,
                    1.0,
                    0.5
                )
            
            df['match_score'] += weights['condition'] * df['condition_score']
        
        # 6. Score square footage match
        if 'total_sqft' in subject and 'total_sqft' in df.columns:
            target_sqft = subject['total_sqft']
            
            # Score based on percentage difference
            df['sqft_score'] = 0.0
            for idx, row in df.iterrows():
                if pd.isna(row['total_sqft']) or row['total_sqft'] == 0:
                    df.at[idx, 'sqft_score'] = 0.5  # Default for missing data
                else:
                    pct_diff = abs(row['total_sqft'] - target_sqft) / target_sqft
                    
                    # Higher score for closer match
                    if pct_diff <= 0.05:  # Within 5%
                        df.at[idx, 'sqft_score'] = 1.0
                    elif pct_diff <= 0.10:  # Within 10%
                        df.at[idx, 'sqft_score'] = 0.9
                    elif pct_diff <= 0.20:  # Within 20%
                        df.at[idx, 'sqft_score'] = 0.8
                    elif pct_diff <= 0.30:  # Within 30%
                        df.at[idx, 'sqft_score'] = 0.6
                    else:
                        df.at[idx, 'sqft_score'] = max(0, 1.0 - pct_diff)
            
            df['match_score'] += weights['sqft'] * df['sqft_score']
        
        # 7. Score STR eligibility match
        if 'str_eligible' in subject and 'str_eligible' in df.columns:
            str_eligible = subject['str_eligible']
            # Standardize format
            if isinstance(str_eligible, str):
                str_eligible = str_eligible.lower() in ['yes', 'true', 'y', '1']
            
            df['str_score'] = np.where(
                df['str_eligible'].astype(str).str.lower().isin(['yes', 'true', 'y', '1']) == str_eligible,
                1.0,
                0.0
            )
            
            df['match_score'] += weights['str'] * df['str_score']
        
        # 8. Score recency (for sold properties)
        if 'sold_date_dt' in df.columns:
            # More recent sales get higher scores
            now = datetime.now()
            df['days_since_sale'] = (now - df['sold_date_dt']).dt.days
            
            # Normalize to 0-1 range
            max_days = df['days_since_sale'].max()
            min_days = df['days_since_sale'].min()
            
            if max_days > min_days:
                df['recency_score'] = 1.0 - ((df['days_since_sale'] - min_days) / (max_days - min_days))
            else:
                df['recency_score'] = 1.0
            
            df['match_score'] += weights['recency'] * df['recency_score']
        
        # 9. Score building premium match
        if 'building_premium' in subject and 'building_premium' in df.columns:
            subject_premium = subject['building_premium']
            
            # Score based on premium difference
            df['building_premium_score'] = 0.0
            for idx, row in df.iterrows():
                premium_diff = abs(row['building_premium'] - subject_premium)
                
                # Closer premium gets higher score
                if premium_diff <= 0.05:
                    df.at[idx, 'building_premium_score'] = 1.0
                elif premium_diff <= 0.1:
                    df.at[idx, 'building_premium_score'] = 0.9
                elif premium_diff <= 0.2:
                    df.at[idx, 'building_premium_score'] = 0.8
                elif premium_diff <= 0.3:
                    df.at[idx, 'building_premium_score'] = 0.6
                else:
                    df.at[idx, 'building_premium_score'] = max(0, 1.0 - premium_diff)
            
            df['match_score'] += weights['building_premium'] * df['building_premium_score']
        
        # 10. Score street premium match
        if 'street_premium' in subject and 'street_premium' in df.columns:
            subject_premium = subject['street_premium']
            
            # Score based on premium difference
            df['street_premium_score'] = 0.0
            for idx, row in df.iterrows():
                premium_diff = abs(row['street_premium'] - subject_premium)
                
                # Closer premium gets higher score
                if premium_diff <= 0.05:
                    df.at[idx, 'street_premium_score'] = 1.0
                elif premium_diff <= 0.1:
                    df.at[idx, 'street_premium_score'] = 0.9
                elif premium_diff <= 0.2:
                    df.at[idx, 'street_premium_score'] = 0.7
                else:
                    df.at[idx, 'street_premium_score'] = max(0, 1.0 - premium_diff)
            
            df['match_score'] += weights['street_premium'] * df['street_premium_score']
        
        # Normalize match score
        total_weight = sum(weights.values())
        df['match_score'] = (df['match_score'] / total_weight) * 100
        
        return df
    
    def _filter_basic_criteria(self, df, criteria):
        """
        Apply basic filters to property data.
        
        Args:
            df (DataFrame): Property data
            criteria (dict): Filter criteria
            
        Returns:
            DataFrame: Filtered data
        """
        # Make a copy of the data
        filtered_df = df.copy()
        initial_count = len(filtered_df)
        
        # Apply filters
        if criteria.get('bedrooms') is not None:
            filtered_df = filtered_df[filtered_df['bedrooms'] == criteria['bedrooms']]
        
        if criteria.get('bathrooms') is not None:
            filtered_df = filtered_df[filtered_df['total_baths'] == criteria['bathrooms']]
        
        if criteria.get('max_price') is not None:
            price_col = 'adjusted_sold_price_time' if 'adjusted_sold_price_time' in filtered_df.columns else 'adjusted_sold_price'
            filtered_df = filtered_df[filtered_df[price_col] <= criteria['max_price']]
        
        if criteria.get('year_built_min') is not None and 'year_built' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['year_built'] >= criteria['year_built_min']]
        
        # Filter by listing status
        if criteria.get('listing_status') is not None and 'listing_status' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['listing_status'] == criteria['listing_status']]
        
        # Filter for recent sales (only for sold properties)
        if criteria.get('listing_status') in [None, 'S', 'Sold', 'SOLD'] and 'sold_date_dt' in filtered_df.columns:
            months_back = criteria.get('months_back', 24)
            cutoff_date = datetime.now() - timedelta(days=30 * months_back)
            filtered_df = filtered_df[filtered_df['sold_date_dt'] >= cutoff_date]
        
        logger.info(f"Applied basic filters: {initial_count} -> {len(filtered_df)} properties")
        
        # Exclude subject property if provided
        if criteria.get('exclude_address'):
            exclude_address = criteria['exclude_address']
            address_col = None
            
            # Find address column
            for col in ['full_address', 'address', 'street_address', 'location']:
                if col in filtered_df.columns:
                    address_col = col
                    break
            
            if address_col:
                # Count before
                before_count = len(filtered_df)
                
                # Exclude the subject property
                filtered_df = filtered_df[~filtered_df[address_col].str.contains(exclude_address, case=False, na=False)]
                
                # Log if found and excluded
                if len(filtered_df) < before_count:
                    logger.info(f"Excluded subject property: {exclude_address}")
            
            # Also try excluding by list_number if present
            if 'list_number' in filtered_df.columns and criteria.get('exclude_list_number'):
                filtered_df = filtered_df[filtered_df['list_number'] != criteria['exclude_list_number']]
        
        return filtered_df
    
    def find_classic_comps(
        self,
        bedrooms=None,
        bathrooms=None,
        property_type=None,
        area=None,
        str_eligible=None,
        condition=None,
        max_price=None,
        sqft_min=None,
        sqft_max=None,
        year_built_min=None,
        months_back=24,
        listing_status=None,  # Can be "A" (Active), "P" (Pending), "S" (Sold)
        limit=5,
        min_comps=3,  # Minimum number of comps to try to find
        exclude_address=None,  # Address of the subject property to exclude
        exclude_list_number=None,  # List number of subject property to exclude
        export_results=False,
        export_dir="outputs",
        export_filename=None,
        custom_weights=None  # Custom scoring weights
    ):
        """
        Find comparable properties based on classic buyer/appraiser criteria.
        
        Args:
            bedrooms (int): Number of bedrooms to match
            bathrooms (float): Number of bathrooms to match (can be .5 increments)
            property_type (str): Type of property (condo, townhome, single-family)
            area (str): Neighborhood or area
            str_eligible (bool): Whether short-term rental is allowed
            condition (str): Property condition (Excellent, Good, Average)
            max_price (float): Maximum price to consider
            sqft_min (float): Minimum square footage
            sqft_max (float): Maximum square footage
            year_built_min (int): Minimum year built
            months_back (int): How many months of sales to consider
            listing_status (str): Filter by listing status (A=Active, P=Pending, S=Sold)
            limit (int): Maximum number of comps to return
            min_comps (int): Minimum number of comps to try to find
            exclude_address (str): Address of subject property to exclude from results
            exclude_list_number (str): List number of subject property to exclude
            export_results (bool): Whether to export results to CSV
            export_dir (str): Directory for export files
            export_filename (str): Custom filename for export
            custom_weights (dict): Custom scoring weights
            
        Returns:
            dict: Results containing matched comps and statistics
        """
        # Build criteria description for logging
        criteria_parts = []
        if bedrooms is not None:
            criteria_parts.append(f"{bedrooms}BR")
        if bathrooms is not None:
            criteria_parts.append(f"{bathrooms}BA")
        if area is not None:
            criteria_parts.append(f"in {area}")
        if listing_status is not None:
            status_desc = {
                "A": "Active",
                "P": "Pending",
                "S": "Sold"
            }.get(listing_status, listing_status)
            criteria_parts.append(f"({status_desc})")
            
        criteria_str = "/".join(criteria_parts)
        logger.info(f"Finding classic comps with criteria: {criteria_str}")
        
        # Create a criteria dictionary
        criteria = {
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'property_type': property_type,
            'area': area,
            'str_eligible': str_eligible,
            'condition': condition,
            'max_price': max_price,
            'sqft_min': sqft_min,
            'sqft_max': sqft_max,
            'year_built_min': year_built_min,
            'months_back': months_back,
            'listing_status': listing_status,
            'exclude_address': exclude_address,
            'exclude_list_number': exclude_list_number
        }
        
        # Apply basic filters
        filtered_df = self._filter_basic_criteria(self.data, criteria)
        initial_count = len(self.data)
        after_basic_filters = len(filtered_df)
        
        # If not enough comps, try relaxing criteria progressively
        if after_basic_filters < min_comps:
            logger.warning(f"Not enough comps found ({after_basic_filters} < {min_comps}). Trying to relax criteria.")
            relaxation_steps = []
            
            # Stage 1: Relax price range by 25%
            if max_price is not None:
                criteria['max_price'] = max_price * 1.25
                relaxed_df = self._filter_basic_criteria(self.data, criteria)
                if len(relaxed_df) >= min_comps:
                    logger.info(f"Found {len(relaxed_df)} comps after relaxing price range by 25%")
                    filtered_df = relaxed_df
                    relaxation_steps.append("price_range_25pct")
                    after_basic_filters = len(filtered_df)
            
            # Stage 2: Relax square footage range by 25%
            if len(filtered_df) < min_comps and (sqft_min is not None or sqft_max is not None):
                if sqft_min is not None:
                    criteria['sqft_min'] = sqft_min * 0.75
                if sqft_max is not None:
                    criteria['sqft_max'] = sqft_max * 1.25
                relaxed_df = self._filter_basic_criteria(self.data, criteria)
                if len(relaxed_df) >= min_comps:
                    logger.info(f"Found {len(relaxed_df)} comps after relaxing sqft range by 25%")
                    filtered_df = relaxed_df
                    relaxation_steps.append("sqft_range_25pct")
                    after_basic_filters = len(filtered_df)
            
            # Stage 3: Increase months_back by 50%
            if len(filtered_df) < min_comps and months_back is not None:
                criteria['months_back'] = int(months_back * 1.5)
                relaxed_df = self._filter_basic_criteria(self.data, criteria)
                if len(relaxed_df) >= min_comps:
                    logger.info(f"Found {len(relaxed_df)} comps after extending time range to {criteria['months_back']} months")
                    filtered_df = relaxed_df
                    relaxation_steps.append("time_range_extended")
                    after_basic_filters = len(filtered_df)
            
            # Stage 4: Remove property type restriction
            if len(filtered_df) < min_comps and property_type is not None:
                criteria['property_type'] = None
                relaxed_df = self._filter_basic_criteria(self.data, criteria)
                if len(relaxed_df) >= min_comps:
                    logger.info(f"Found {len(relaxed_df)} comps after removing property type filter")
                    filtered_df = relaxed_df
                    relaxation_steps.append("property_type_removed")
                    after_basic_filters = len(filtered_df)
            
            # Log relaxation steps
            if relaxation_steps:
                logger.info(f"Applied filter relaxation: {', '.join(relaxation_steps)}")
        
        # Check if we still don't have enough data after relaxation
        if len(filtered_df) == 0:
            logger.warning("No properties match the basic criteria. Try relaxing filters.")
            return {
                "comps": pd.DataFrame(),
                "stats": {},
                "filter_stats": {
                    "initial_count": initial_count,
                    "after_basic_filters": 0
                }, "message": "No properties match the basic criteria. Try relaxing filters.",
                "filter_relaxation_applied": []
            }
        
        # Find subject property for comparison, or create a synthetic one from criteria
        subject = None
        if exclude_address:
            # Try to find the subject property in the original dataset
            address_col = None
            for col in ['full_address', 'address', 'street_address', 'location']:
                if col in self.data.columns:
                    address_col = col
                    break
            
            if address_col:
                subject_matches = self.data[self.data[address_col].str.contains(exclude_address, case=False, na=False)]
                if len(subject_matches) > 0:
                    subject = subject_matches.iloc[0]
                    logger.info(f"Found subject property: {subject.get(address_col)}")
        
        # If subject not found, create a synthetic one from criteria
        if subject is None:
            logger.info("Creating synthetic subject property from criteria")
            subject = pd.Series({
                'bedrooms': bedrooms,
                'total_baths': bathrooms,
                'resolved_property_type': property_type,
                'area': area,
                'str_eligible': str_eligible,
                'improved_condition': condition,
                'total_sqft': (sqft_min + sqft_max) / 2 if sqft_min is not None and sqft_max is not None else None
            })
            
            # Try to infer building and street premiums
            if area and 'sub_loc' in self.data.columns:
                # Find properties in the same area to infer building
                area_properties = self.data[self.data['area'] == area]
                if len(area_properties) > 0:
                    # Use most common sub_loc in that area
                    sub_loc_counts = area_properties['sub_loc'].value_counts()
                    if len(sub_loc_counts) > 0:
                        subject['sub_loc'] = sub_loc_counts.index[0]
                        # Find building premium
                        for building, premium in building_premiums.items():
                            if building.lower() in subject['sub_loc'].lower():
                                subject['building_premium'] = premium
                                break
                        else:
                            subject['building_premium'] = 0.0
                    else:
                        subject['building_premium'] = 0.0
                else:
                    subject['building_premium'] = 0.0
            else:
                subject['building_premium'] = 0.0
            
            # Infer street premium
            if exclude_address:
                street = self._extract_street_name(exclude_address)
                if street in street_premiums:
                    subject['street_premium'] = street_premiums[street]
                else:
                    subject['street_premium'] = 0.0
            else:
                subject['street_premium'] = 0.0
        
        # Score comps based on similarity to subject
        scored_comps = self._score_comps(subject, filtered_df, weights=custom_weights)
        
        # Sort by match score and take top results
        comps = scored_comps.sort_values('match_score', ascending=False).head(limit)
        
        logger.info(f"Found {len(comps)} comparable properties")
        
        # Prepare statistics
        if len(comps) > 0 and 'adjusted_sold_price_time' in comps.columns:
            avg_price = comps['adjusted_sold_price_time'].mean()
            median_price = comps['adjusted_sold_price_time'].median()
            price_range = (comps['adjusted_sold_price_time'].min(), comps['adjusted_sold_price_time'].max())
            
            stats = {
                "average_price": avg_price,
                "median_price": median_price,
                "price_range": price_range
            }
            
            if 'adjusted_price_per_sqft_time' in comps.columns:
                avg_price_sqft = comps['adjusted_price_per_sqft_time'].mean()
                median_price_sqft = comps['adjusted_price_per_sqft_time'].median()
                price_sqft_range = (comps['adjusted_price_per_sqft_time'].min(), comps['adjusted_price_per_sqft_time'].max())
                
                stats.update({
                    "average_price_per_sqft": avg_price_sqft,
                    "median_price_per_sqft": median_price_sqft,
                    "price_per_sqft_range": price_sqft_range
                })
                
            # Building premium statistics
            if 'building_premium' in comps.columns:
                avg_building_premium = comps['building_premium'].mean()
                building_premium_range = (comps['building_premium'].min(), comps['building_premium'].max())
                
                stats.update({
                    "average_building_premium": avg_building_premium,
                    "building_premium_range": building_premium_range
                })
            
            # Street premium statistics
            if 'street_premium' in comps.columns:
                avg_street_premium = comps['street_premium'].mean()
                street_premium_range = (comps['street_premium'].min(), comps['street_premium'].max())
                
                stats.update({
                    "average_street_premium": avg_street_premium,
                    "street_premium_range": street_premium_range
                })
        else:
            stats = {}
        
        # Export results if requested
        export_path = None
        if export_results and len(comps) > 0:
            export_path = self._export_results(comps, subject, export_dir, export_filename, bedrooms, bathrooms, area)
        
        # Return results
        return {
            "comps": comps,
            "subject": subject,
            "stats": stats,
            "filter_stats": {
                "initial_count": initial_count,
                "after_basic_filters": after_basic_filters,
                "final_comps": len(comps)
            },
            "export_path": export_path if export_path else None,
            "filter_relaxation_applied": relaxation_steps if 'relaxation_steps' in locals() else []
        }
    
    def _export_results(self, comps, subject, export_dir, export_filename=None, bedrooms=None, bathrooms=None, area=None):
        """
        Export results to CSV file.
        
        Args:
            comps (DataFrame): Comparable properties
            subject (Series): Subject property
            export_dir (str): Directory for export
            export_filename (str): Custom filename
            bedrooms (int): Number of bedrooms filter
            bathrooms (float): Number of bathrooms filter
            area (str): Area filter
            
        Returns:
            str: Path to exported file
        """
        # Create export directory if it doesn't exist
        os.makedirs(export_dir, exist_ok=True)
        
        # Generate filename if not provided
        if not export_filename:
            timestamp = datetime.now().strftime("%Y%m%d")
            criteria = []
            
            if bedrooms is not None:
                criteria.append(f"{bedrooms}BR")
            
            if bathrooms is not None:
                criteria.append(f"{bathrooms}BA")
            
            if area:
                # Remove spaces and special characters
                clean_area = re.sub(r'[^a-zA-Z0-9]', '_', area)
                criteria.append(clean_area)
            
            criteria_str = "_".join(criteria) if criteria else "all"
            export_filename = f"classic_comps_{criteria_str}_{timestamp}.csv"
        
        # Ensure file has .csv extension
        if not export_filename.endswith('.csv'):
            export_filename += '.csv'
        
        # Full path for export
        export_path = os.path.join(export_dir, export_filename)
        
        # Export the DataFrame
        comps.to_csv(export_path, index=False)
        logger.info(f"Results exported to {export_path}")
        
        return export_path


def run_classic_comp_search(
    bedrooms=None,
    bathrooms=None,
    property_type=None,
    area=None,
    str_eligible=None,
    condition=None,
    max_price=None,
    sqft_min=None,
    sqft_max=None,
    year_built_min=None,
    months_back=24,
    listing_status=None,  # Can be "A" (Active), "P" (Pending), "S" (Sold)
    reference_property_address=None,  # Can search for comps similar to a specific property
    min_comps=3,  # Minimum number of comps to try to find
    limit=5,
    csv_path=None,
    export_results=True,
    export_dir="outputs",
    custom_weights=None  # Custom scoring weights
):
    """
    Run a classic comparable property search.
    
    Args:
        (same as find_classic_comps method)
        
    Returns:
        dict: Results dictionary
    """
    try:
        # Initialize the finder
        finder = ClassicCompFinder(csv_path=csv_path, time_adjust=True)
        
        # If a reference property is provided, find its details to use as search criteria
        if reference_property_address:
            logger.info(f"Finding comps relative to reference property: {reference_property_address}")
            # Find the reference property in the dataset
            address_col = None
            for col in ['full_address', 'address', 'street_address', 'location']:
                if col in finder.data.columns:
                    address_col = col
                    break
                    
            if address_col:
                reference_props = finder.data[finder.data[address_col].str.contains(reference_property_address, case=False, na=False)]
                
                if len(reference_props) == 0:
                    # Try alternative address fields
                    for addr_field in ['street_address', 'address', 'location']:
                        if addr_field in finder.data.columns:
                            reference_props = finder.data[finder.data[addr_field].str.contains(reference_property_address, case=False, na=False)]
                            if len(reference_props) > 0:
                                break
                
                if len(reference_props) > 0:
                    reference_prop = reference_props.iloc[0]
                    logger.info(f"Found reference property: {reference_prop.get(address_col)}")
                    
                    # Use the reference property's attributes if not explicitly provided
                    if bedrooms is None and 'bedrooms' in reference_prop:
                        bedrooms = reference_prop['bedrooms']
                        logger.info(f"Using reference property bedrooms: {bedrooms}")
                    
                    if bathrooms is None and 'total_baths' in reference_prop:
                        bathrooms = reference_prop['total_baths']
                        logger.info(f"Using reference property bathrooms: {bathrooms}")
                    
                    if property_type is None and 'resolved_property_type' in reference_prop:
                        property_type = reference_prop['resolved_property_type']
                        logger.info(f"Using reference property type: {property_type}")
                    
                    if area is None and 'area' in reference_prop:
                        area = reference_prop['area']
                        logger.info(f"Using reference property area: {area}")
                    
                    if condition is None and 'improved_condition' in reference_prop:
                        condition = reference_prop['improved_condition']
                        logger.info(f"Using reference property condition: {condition}")
                    
                    if str_eligible is None and 'str_eligible' in reference_prop:
                        str_eligible = reference_prop['str_eligible']
                        logger.info(f"Using reference property STR eligibility: {str_eligible}")
                    
                    if sqft_min is None and sqft_max is None and 'total_sqft' in reference_prop:
                        # Use ±20% of the reference property's square footage
                        ref_sqft = reference_prop['total_sqft']
                        sqft_min = ref_sqft * 0.8
                        sqft_max = ref_sqft * 1.2
                        logger.info(f"Using reference property sqft range: {sqft_min:.0f} - {sqft_max:.0f}")
                    
                    # Get list_number for exclusion
                    exclude_list_number = reference_prop.get('list_number')
                else:
                    logger.warning(f"Reference property not found: {reference_property_address}")
                    exclude_list_number = None
            else:
                logger.warning("No address column found in dataset")
                exclude_list_number = None
        else:
            exclude_list_number = None
        
        # Run the search
        results = finder.find_classic_comps(
            bedrooms=bedrooms,
            bathrooms=bathrooms,
            property_type=property_type,
            area=area,
            str_eligible=str_eligible,
            condition=condition,
            max_price=max_price,
            sqft_min=sqft_min,
            sqft_max=sqft_max,
            year_built_min=year_built_min,
            months_back=months_back,
            listing_status=listing_status,
            limit=limit,
            min_comps=min_comps,
            exclude_address=reference_property_address,
            exclude_list_number=exclude_list_number,
            export_results=export_results,
            export_dir=export_dir,
            custom_weights=custom_weights
        )
        
        # Print summary to console
        comps = results["comps"]
        stats = results["stats"]
        
        print("\n" + "=" * 80)
        criteria_list = []
        if bedrooms is not None:
            criteria_list.append(f"{bedrooms}BR")
        if bathrooms is not None:
            criteria_list.append(f"{bathrooms}BA")
        if property_type:
            criteria_list.append(property_type)
        if area:
            criteria_list.append(f"in {area}")
        if str_eligible is not None:
            criteria_list.append("STR-eligible" if str_eligible else "non-STR")
        
        # Add listing status to criteria
        status_label = ""
        if listing_status == "A":
            status_label = "ACTIVE LISTINGS"
        elif listing_status == "P":
            status_label = "PENDING LISTINGS"
        elif listing_status in ["S", "Sold", "SOLD"]:
            status_label = "SOLD PROPERTIES"
        
        criteria_str = " ".join(criteria_list)
        
        if reference_property_address:
            print(f"CLASSIC COMP FINDER: Properties similar to {reference_property_address}")
            if status_label:
                print(f"{status_label} - {criteria_str}")
        else:
            if status_label:
                print(f"CLASSIC COMP FINDER: {status_label}")
                print(f"Criteria: {criteria_str}")
            else:
                print(f"CLASSIC COMP FINDER: {criteria_str}")
        
        print("=" * 80)
        
        if len(comps) == 0:
            print("\nNo comparable properties found matching your criteria.")
            print("Try relaxing some filters or expanding your search area.")
            return results
        
        print(f"\nFound {len(comps)} comparable properties")
        
        # Show filter relaxation if applied
        if "filter_relaxation_applied" in results and results["filter_relaxation_applied"]:
            relaxed_filters = results["filter_relaxation_applied"]
            print("\nFilter relaxation was applied to find enough comps:")
            for rf in relaxed_filters:
                if rf == "price_range_25pct":
                    print("  - Price range increased by 25%")
                elif rf == "sqft_range_25pct":
                    print("  - Square footage range expanded by 25%")
                elif rf == "time_range_extended":
                    print(f"  - Time range extended to {months_back * 1.5:.0f} months")
                elif rf == "property_type_removed":
                    print("  - Property type filter removed")
                else:
                    print(f"  - {rf}")
        
        if stats:
            # Determine which price field to use (list price for active, sold price for sold)
            price_field = "average_price"
            price_sqft_field = "average_price_per_sqft"
            price_label = "Price"
            
            # Adjust labels based on listing status
            if listing_status == "A":
                price_label = "List Price"
            elif listing_status in ["S", "Sold", "SOLD"]:
                price_label = "Sold Price"
            
            print(f"\n{price_label.upper()} STATISTICS:")
            if price_field in stats:
                print(f"  Average {price_label}: ${stats[price_field]:,.0f}")
            if "median_price" in stats:
                print(f"  Median {price_label}: ${stats['median_price']:,.0f}")
            if "price_range" in stats:
                print(f"  {price_label} Range: ${stats['price_range'][0]:,.0f} - ${stats['price_range'][1]:,.0f}")
            
            if price_sqft_field in stats:
                print(f"\n{price_label.upper()} PER SQFT STATISTICS:")
                print(f"  Average: ${stats[price_sqft_field]:,.2f}/sqft")
                print(f"  Median: ${stats['median_price_per_sqft']:,.2f}/sqft")
                print(f"  Range: ${stats['price_per_sqft_range'][0]:,.2f} - ${stats['price_per_sqft_range'][1]:,.2f}/sqft")
            
            # Building premium stats
            if "average_building_premium" in stats:
                print("\nBUILDING PREMIUM STATISTICS:")
                print(f"  Average Premium: {stats['average_building_premium']:.1%}")
                print(f"  Range: {stats['building_premium_range'][0]:.1%} - {stats['building_premium_range'][1]:.1%}")
            
            # Street premium stats
            if "average_street_premium" in stats:
                print("\nSTREET PREMIUM STATISTICS:")
                print(f"  Average Premium: {stats['average_street_premium']:.1%}")
                print(f"  Range: {stats['street_premium_range'][0]:.1%} - {stats['street_premium_range'][1]:.1%}")
        
        # Display top comps
        print("\nTOP COMPARABLE PROPERTIES:")
        
        # Determine display columns
        display_cols = []
        
        # Core columns
        display_cols.extend(['full_address', 'sold_date', 'adjusted_sold_price_time', 'match_score'])
        
        # Add filtered criteria columns
        if bedrooms is not None:
            display_cols.append('bedrooms')
        if bathrooms is not None:
            display_cols.append('total_baths')
        if property_type is not None:
            display_cols.append('resolved_property_type')
        if area is not None:
            display_cols.append('area')
        if condition is not None:
            display_cols.append('improved_condition')
        if str_eligible is not None:
            display_cols.append('str_eligible')
        if sqft_min is not None or sqft_max is not None:
            display_cols.append('total_sqft')
        
        # Add premium columns if available
        if 'building_premium' in comps.columns:
            display_cols.append('building_premium')
        if 'street_premium' in comps.columns:
            display_cols.append('street_premium')
        
        # Make sure we have all columns before displaying
        display_cols = [col for col in display_cols if col in comps.columns]
        
        # Display top properties
        for idx, comp in comps.iterrows():
            print(f"\n{idx+1}. {comp.get('full_address', 'Address not available')}")
            print(f"   Match Score: {comp.get('match_score', 0):.1f}")
            
            # Display price with time adjustment note
            if 'adjusted_sold_price_time' in comp:
                original_price = comp.get('adjusted_sold_price', comp['adjusted_sold_price_time'])
                print(f"   Price: ${comp['adjusted_sold_price_time']:,.0f}")
                
                if abs(original_price - comp['adjusted_sold_price_time']) > 100:
                    print(f"   (Original: ${original_price:,.0f}, adjusted for time)")
            
            # Display key attributes
            attrs = []
            if 'bedrooms' in comp:
                attrs.append(f"{int(comp['bedrooms'])}BR")
            if 'total_baths' in comp:
                attrs.append(f"{comp['total_baths']}BA")
            if 'total_sqft' in comp:
                attrs.append(f"{int(comp['total_sqft'])} sqft")
            if attrs:
                print(f"   {' | '.join(attrs)}")
            
            # Display property type and condition
            type_cond = []
            if 'resolved_property_type' in comp:
                type_cond.append(comp['resolved_property_type'])
            if 'improved_condition' in comp:
                type_cond.append(comp['improved_condition'])
            if type_cond:
                print(f"   {' | '.join(type_cond)}")
            
            # Show STR eligibility
            if 'str_eligible' in comp:
                print(f"   STR Eligible: {comp['str_eligible']}")
            
            # Show sale date
            if 'sold_date' in comp:
                print(f"   Sold: {comp['sold_date']}")
            
            # Show premium information
            if 'building_premium' in comp and comp['building_premium'] > 0:
                print(f"   Building Premium: {comp['building_premium']:.1%}")
            
            if 'street_premium' in comp and comp['street_premium'] != 0:
                sign = "+" if comp['street_premium'] > 0 else ""
                print(f"   Street Premium: {sign}{comp['street_premium']:.1%}")
        
        # Note export location
        if export_results:
            if "export_path" in results and results["export_path"]:
                print(f"\nResults exported to: {results['export_path']}")
            else:
                print(f"\nResults exported to: {export_dir}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in classic comp search: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            "error": str(e),
            "comps": pd.DataFrame(),
            "stats": {},
            "filter_stats": {}
        }


if __name__ == "__main__":
    # Simple test of the functionality
    run_classic_comp_search(
        bedrooms=2,
        bathrooms=2,
        property_type="Condo",
        area="Core",
        str_eligible=True,
        months_back=36,  # Look back 3 years for more data
        # Use "A" for active or comment out to see all listings
        # listing_status="A",
        limit=5
    )
