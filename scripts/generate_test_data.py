#!/usr/bin/env python3
"""
Generate synthetic test data for DIAS testing.

This script creates realistic parcel and elevation data for flood impact analysis testing.

Usage:
    python scripts/generate_test_data.py --parcels 100 --region houston --output test_data/
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

# Try to import DBF writing library
try:
    from dbfpy3 import dbf
    HAS_DBFPY = True
except ImportError:
    try:
        import simpledbf
        HAS_DBFPY = False
        print("Warning: dbfpy3 not found, using simpledbf (limited functionality)")
    except ImportError:
        print("Error: Neither dbfpy3 nor simpledbf found.")
        print("Install with: pip install dbfpy3")
        sys.exit(1)


# Region configurations
REGIONS = {
    'houston': {
        'name': 'Houston, TX',
        'lat_range': (29.65, 29.85),
        'lon_range': (-95.50, -95.30),
        'description': 'Houston area with Gulf Coast flood risk'
    },
    'miami': {
        'name': 'Miami, FL',
        'lat_range': (25.70, 25.85),
        'lon_range': (-80.30, -80.15),
        'description': 'Miami area with Atlantic Coast flood risk'
    }
}


def generate_coordinates(n_parcels: int, 
                        lat_range: Tuple[float, float],
                        lon_range: Tuple[float, float],
                        min_distance: float = 0.001,
                        seed: int = 42) -> np.ndarray:
    """
    Generate spatially distributed coordinates with minimum distance constraint.
    
    Args:
        n_parcels: Number of parcels to generate
        lat_range: (min_lat, max_lat) in decimal degrees
        lon_range: (min_lon, max_lon) in decimal degrees
        min_distance: Minimum distance between parcels in degrees (~111 meters)
        seed: Random seed for reproducibility
    
    Returns:
        Array of shape (n_parcels, 2) with [lat, lon] coordinates
    """
    np.random.seed(seed)
    
    coords = []
    max_attempts = n_parcels * 10
    attempts = 0
    
    while len(coords) < n_parcels and attempts < max_attempts:
        # Generate random coordinate
        lat = np.random.uniform(lat_range[0], lat_range[1])
        lon = np.random.uniform(lon_range[0], lon_range[1])
        
        # Check minimum distance from existing points
        if len(coords) == 0:
            coords.append([lat, lon])
        else:
            coords_array = np.array(coords)
            distances = np.sqrt((coords_array[:, 0] - lat)**2 + (coords_array[:, 1] - lon)**2)
            if np.all(distances >= min_distance):
                coords.append([lat, lon])
        
        attempts += 1
    
    if len(coords) < n_parcels:
        print(f"Warning: Only generated {len(coords)} parcels (requested {n_parcels})")
        print(f"Consider reducing min_distance or increasing lat/lon range")
    
    return np.array(coords)


def generate_elevations(coords: np.ndarray,
                       elevation_range: Tuple[float, float] = (0, 20),
                       n_seed_points: int = 5,
                       seed: int = 42) -> np.ndarray:
    """
    Generate spatially correlated elevations using distance-weighted interpolation.
    
    Creates realistic elevation gradients by interpolating from random seed points.
    
    Args:
        coords: Array of shape (n, 2) with [lat, lon] coordinates
        elevation_range: (min_elevation, max_elevation) in feet
        n_seed_points: Number of seed points for interpolation
        seed: Random seed for reproducibility
    
    Returns:
        Array of elevations in feet
    """
    np.random.seed(seed)
    n_parcels = len(coords)
    
    # Generate random seed points with elevations
    seed_coords = coords[np.random.choice(n_parcels, n_seed_points, replace=False)]
    seed_elevations = np.random.uniform(elevation_range[0], elevation_range[1], n_seed_points)
    
    # Calculate elevations using inverse distance weighting
    elevations = np.zeros(n_parcels)
    
    for i, coord in enumerate(coords):
        # Calculate distances to all seed points
        distances = np.sqrt(np.sum((seed_coords - coord)**2, axis=1))
        
        # Avoid division by zero
        distances = np.maximum(distances, 1e-10)
        
        # Inverse distance weighting
        weights = 1.0 / distances
        weights /= weights.sum()
        
        # Interpolate elevation
        elevations[i] = np.dot(weights, seed_elevations)
    
    # Add some random variation
    noise = np.random.normal(0, 1.0, n_parcels)
    elevations += noise
    
    # Clip to range
    elevations = np.clip(elevations, elevation_range[0], elevation_range[1])
    
    return elevations


def generate_property_values(elevations: np.ndarray,
                            has_structure: np.ndarray,
                            land_use: np.ndarray,
                            seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate realistic property values correlated with elevation and land use.
    
    Args:
        elevations: Array of elevations in feet
        has_structure: Boolean array indicating if parcel has structure
        land_use: Array of land use types (0=Residential, 1=Commercial, 2=Industrial)
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (land_values, building_values) in USD
    """
    np.random.seed(seed)
    n_parcels = len(elevations)
    
    # Base land value: higher elevation = higher value (flood premium)
    # Normalize elevations to [0, 1]
    norm_elevation = (elevations - elevations.min()) / (elevations.max() - elevations.min() + 1e-10)
    
    # Base land value: $30,000 - $200,000
    land_values = np.random.normal(80000, 30000, n_parcels)
    
    # Elevation premium: +20% for high elevation
    elevation_factor = 1.0 + 0.2 * norm_elevation
    land_values *= elevation_factor
    
    # Land use multipliers
    land_use_multipliers = {
        0: 1.0,    # Residential
        1: 2.0,    # Commercial
        2: 1.5     # Industrial
    }
    
    for use_type, multiplier in land_use_multipliers.items():
        mask = land_use == use_type
        land_values[mask] *= multiplier
    
    # Clip to reasonable range
    land_values = np.clip(land_values, 30000, 500000)
    
    # Building values
    building_values = np.zeros(n_parcels)
    
    # Only parcels with structures have building value
    structure_mask = has_structure
    n_structures = structure_mask.sum()
    
    if n_structures > 0:
        # Base building value: $100,000 - $600,000
        building_values[structure_mask] = np.random.normal(250000, 100000, n_structures)
        
        # Land use multipliers for buildings
        for use_type, multiplier in land_use_multipliers.items():
            mask = structure_mask & (land_use == use_type)
            building_values[mask] *= multiplier
        
        # Elevation correlation (higher = more expensive construction)
        building_values[structure_mask] *= elevation_factor[structure_mask]
        
        # Clip to reasonable range
        building_values = np.clip(building_values, 0, 1500000)
    
    return land_values, building_values


def write_dbf(data: pd.DataFrame, filename: str) -> None:
    """
    Write DataFrame to DBF file.
    
    Args:
        data: DataFrame with parcel data
        filename: Output DBF filename
    """
    if not HAS_DBFPY:
        raise ImportError("dbfpy3 required for DBF writing. Install with: pip install dbfpy3")
    
    # Create DBF file
    db = dbf.Dbf(filename, new=True)
    
    # Define fields
    db.addField(
        ("PARCELID", "C", 10),
        ("LAT", "N", 12, 8),
        ("LON", "N", 12, 8),
        ("LANDVALUE", "N", 12, 2),
        ("BLDGVALUE", "N", 12, 2),
        ("LANDUSE", "C", 20),
        ("STRUCTURE", "C", 1),
        ("OWNER", "C", 50),
        ("ZONING", "C", 10),
    )
    
    # Add records
    for _, row in data.iterrows():
        rec = db.newRecord()
        rec["PARCELID"] = row["PARCELID"]
        rec["LAT"] = float(row["LAT"])
        rec["LON"] = float(row["LON"])
        rec["LANDVALUE"] = float(row["LANDVALUE"])
        rec["BLDGVALUE"] = float(row["BLDGVALUE"])
        rec["LANDUSE"] = row["LANDUSE"]
        rec["STRUCTURE"] = row["STRUCTURE"]
        rec["OWNER"] = row["OWNER"]
        rec["ZONING"] = row["ZONING"]
        rec.store()
    
    db.close()
    print(f"✅ DBF file written: {filename}")


def write_csv(data: pd.DataFrame, filename: str) -> None:
    """
    Write DataFrame to CSV file.
    
    Args:
        data: DataFrame with data
        filename: Output CSV filename
    """
    data.to_csv(filename, index=False)
    print(f"✅ CSV file written: {filename}")


def generate_test_data(n_parcels: int = 100,
                      region: str = 'houston',
                      output_dir: str = 'test_data',
                      seed: int = 42) -> None:
    """
    Generate complete test dataset for DIAS.
    
    Args:
        n_parcels: Number of parcels to generate
        region: Region name ('houston' or 'miami')
        output_dir: Output directory path
        seed: Random seed for reproducibility
    """
    print("=" * 60)
    print("  DIAS Test Data Generation")
    print("=" * 60)
    print()
    
    # Validate region
    if region not in REGIONS:
        print(f"Error: Unknown region '{region}'")
        print(f"Available regions: {', '.join(REGIONS.keys())}")
        sys.exit(1)
    
    region_config = REGIONS[region]
    print(f"Region: {region_config['name']}")
    print(f"Description: {region_config['description']}")
    print(f"Parcels: {n_parcels}")
    print(f"Seed: {seed} (for reproducibility)")
    print()
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Generate coordinates
    print("📍 Generating parcel coordinates...")
    coords = generate_coordinates(
        n_parcels,
        region_config['lat_range'],
        region_config['lon_range'],
        seed=seed
    )
    actual_parcels = len(coords)
    print(f"   Generated {actual_parcels} parcel locations")
    
    # Step 2: Generate elevations
    print("📏 Generating elevations...")
    elevations = generate_elevations(coords, seed=seed)
    print(f"   Elevation range: {elevations.min():.2f} - {elevations.max():.2f} feet")
    
    # Calculate distribution
    low = (elevations < 5).sum()
    medium = ((elevations >= 5) & (elevations < 10)).sum()
    high = ((elevations >= 10) & (elevations < 15)).sum()
    very_high = (elevations >= 15).sum()
    print(f"   Distribution: {low} low (<5ft), {medium} medium (5-10ft), "
          f"{high} high (10-15ft), {very_high} very high (15+ft)")
    
    # Step 3: Generate land use and structures
    print("🏗️  Generating land use and structures...")
    np.random.seed(seed)
    
    # Land use: 70% Residential, 20% Commercial, 10% Industrial
    land_use_codes = np.random.choice([0, 1, 2], actual_parcels, p=[0.7, 0.2, 0.1])
    land_use_names = ["Residential", "Commercial", "Industrial"]
    land_use = np.array([land_use_names[code] for code in land_use_codes])
    
    # Structures: 85% have buildings
    has_structure = np.random.choice([True, False], actual_parcels, p=[0.85, 0.15])
    
    print(f"   {(land_use == 'Residential').sum()} Residential, "
          f"{(land_use == 'Commercial').sum()} Commercial, "
          f"{(land_use == 'Industrial').sum()} Industrial")
    print(f"   {has_structure.sum()} parcels with structures")
    
    # Step 4: Generate property values
    print("💰 Generating property values...")
    land_values, building_values = generate_property_values(
        elevations, has_structure, land_use_codes, seed=seed
    )
    print(f"   Land value range: ${land_values.min():,.0f} - ${land_values.max():,.0f}")
    print(f"   Building value range: ${building_values.min():,.0f} - ${building_values.max():,.0f}")
    
    # Step 5: Create DataFrame
    print("📋 Creating data tables...")
    
    # Parcel data
    parcel_data = pd.DataFrame({
        'PARCELID': [f"P{i:05d}" for i in range(1, actual_parcels + 1)],
        'LAT': coords[:, 0],
        'LON': coords[:, 1],
        'LANDVALUE': land_values,
        'BLDGVALUE': building_values,
        'LANDUSE': land_use,
        'STRUCTURE': ['Y' if s else 'N' for s in has_structure],
        'OWNER': [f"Owner {i}" for i in range(1, actual_parcels + 1)],
        'ZONING': [f"Z{i % 5 + 1}" for i in range(actual_parcels)],
    })
    
    # Elevation data
    elevation_data = pd.DataFrame({
        'PARCELID': [f"P{i:05d}" for i in range(1, actual_parcels + 1)],
        'ELEVATION': elevations,
        'LAT': coords[:, 0],
        'LON': coords[:, 1],
    })
    
    # Step 6: Write files
    print("💾 Writing output files...")
    
    # Write DBF
    dbf_file = output_path / "parcels_test.dbf"
    try:
        write_dbf(parcel_data, str(dbf_file))
    except Exception as e:
        print(f"⚠️  Warning: Could not write DBF file: {e}")
        print("   Falling back to CSV only")
        csv_file = output_path / "parcels_test.csv"
        write_csv(parcel_data, str(csv_file))
    
    # Write elevation CSV
    elevation_file = output_path / "elevations_test.csv"
    write_csv(elevation_data, str(elevation_file))
    
    # Write summary
    summary_file = output_path / "README.txt"
    with open(summary_file, 'w') as f:
        f.write("DIAS Test Data\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Generated: {pd.Timestamp.now()}\n")
        f.write(f"Region: {region_config['name']}\n")
        f.write(f"Parcels: {actual_parcels}\n")
        f.write(f"Seed: {seed}\n\n")
        f.write("Files:\n")
        f.write(f"  - parcels_test.dbf (or .csv) - Parcel attribute data\n")
        f.write(f"  - elevations_test.csv - Elevation data\n\n")
        f.write("Statistics:\n")
        f.write(f"  Elevation: {elevations.min():.2f} - {elevations.max():.2f} ft\n")
        f.write(f"  Land value: ${land_values.min():,.0f} - ${land_values.max():,.0f}\n")
        f.write(f"  Building value: ${building_values.min():,.0f} - ${building_values.max():,.0f}\n")
    
    print(f"✅ Summary written: {summary_file}")
    
    print()
    print("=" * 60)
    print("✅ Test data generation complete!")
    print("=" * 60)
    print(f"\nOutput directory: {output_path.absolute()}")
    print(f"Files generated:")
    print(f"  - parcels_test.dbf (or .csv)")
    print(f"  - elevations_test.csv")
    print(f"  - README.txt")


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic test data for DIAS testing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--parcels',
        type=int,
        default=100,
        help='Number of parcels to generate'
    )
    
    parser.add_argument(
        '--region',
        type=str,
        default='houston',
        choices=list(REGIONS.keys()),
        help='Geographic region for test data'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='test_data',
        help='Output directory for generated files'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    try:
        generate_test_data(
            n_parcels=args.parcels,
            region=args.region,
            output_dir=args.output,
            seed=args.seed
        )
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

