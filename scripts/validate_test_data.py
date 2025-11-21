#!/usr/bin/env python3
"""
Validate generated test data for DIAS.

This script checks that generated test data meets requirements and can be loaded by DIAS.

Usage:
    python scripts/validate_test_data.py --data test_data/
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

try:
    from dbfread import DBF
    HAS_DBF = True
except ImportError:
    HAS_DBF = False
    print("Warning: dbfread not installed, cannot validate DBF files")


def validate_parcel_data(file_path: str) -> Dict[str, any]:
    """
    Validate parcel data file (DBF or CSV).
    
    Args:
        file_path: Path to parcel data file
    
    Returns:
        Dictionary with validation results
    """
    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    file_path = Path(file_path)
    
    # Check file exists
    if not file_path.exists():
        results['valid'] = False
        results['errors'].append(f"File not found: {file_path}")
        return results
    
    # Load data based on extension
    try:
        if file_path.suffix.lower() == '.dbf':
            if not HAS_DBF:
                results['errors'].append("dbfread not installed, cannot read DBF")
                results['valid'] = False
                return results
            
            # Read DBF
            records = []
            for record in DBF(str(file_path), encoding='utf-8'):
                records.append(dict(record))
            data = pd.DataFrame(records)
        
        elif file_path.suffix.lower() == '.csv':
            data = pd.read_csv(file_path)
        
        else:
            results['valid'] = False
            results['errors'].append(f"Unsupported file format: {file_path.suffix}")
            return results
    
    except Exception as e:
        results['valid'] = False
        results['errors'].append(f"Error reading file: {e}")
        return results
    
    # Required fields
    required_fields = [
        'PARCELID', 'LAT', 'LON', 'LANDVALUE', 'BLDGVALUE',
        'LANDUSE', 'STRUCTURE'
    ]
    
    # Check all required fields present
    missing_fields = [f for f in required_fields if f not in data.columns]
    if missing_fields:
        results['valid'] = False
        results['errors'].append(f"Missing required fields: {missing_fields}")
        return results
    
    # Validate PARCELID
    if data['PARCELID'].duplicated().any():
        results['valid'] = False
        results['errors'].append("Duplicate PARCELID values found")
    
    # Validate LAT
    if not data['LAT'].between(-90, 90).all():
        results['valid'] = False
        results['errors'].append("LAT values outside valid range [-90, 90]")
    
    # Validate LON
    if not data['LON'].between(-180, 180).all():
        results['valid'] = False
        results['errors'].append("LON values outside valid range [-180, 180]")
    
    # Validate LANDVALUE
    if (data['LANDVALUE'] < 0).any():
        results['valid'] = False
        results['errors'].append("Negative LANDVALUE found")
    
    if (data['LANDVALUE'] == 0).all():
        results['warnings'].append("All LANDVALUE are zero")
    
    # Validate BLDGVALUE
    if (data['BLDGVALUE'] < 0).any():
        results['valid'] = False
        results['errors'].append("Negative BLDGVALUE found")
    
    # Validate STRUCTURE field
    structure_values = data['STRUCTURE'].unique()
    if not all(v in ['Y', 'N'] for v in structure_values):
        results['valid'] = False
        results['errors'].append("STRUCTURE field must contain only 'Y' or 'N'")
    
    # Check consistency: STRUCTURE='N' should have BLDGVALUE=0
    no_structure = data[data['STRUCTURE'] == 'N']
    if (no_structure['BLDGVALUE'] > 0).any():
        results['warnings'].append(
            "Some parcels with STRUCTURE='N' have non-zero BLDGVALUE"
        )
    
    # Collect statistics
    results['stats'] = {
        'n_parcels': len(data),
        'lat_range': (data['LAT'].min(), data['LAT'].max()),
        'lon_range': (data['LON'].min(), data['LON'].max()),
        'land_value_range': (data['LANDVALUE'].min(), data['LANDVALUE'].max()),
        'building_value_range': (data['BLDGVALUE'].min(), data['BLDGVALUE'].max()),
        'n_with_structures': (data['STRUCTURE'] == 'Y').sum(),
        'land_uses': data['LANDUSE'].value_counts().to_dict()
    }
    
    return results


def validate_elevation_data(file_path: str, parcel_ids: List[str] = None) -> Dict[str, any]:
    """
    Validate elevation data file.
    
    Args:
        file_path: Path to elevation CSV file
        parcel_ids: Optional list of parcel IDs to check against
    
    Returns:
        Dictionary with validation results
    """
    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    file_path = Path(file_path)
    
    # Check file exists
    if not file_path.exists():
        results['valid'] = False
        results['errors'].append(f"File not found: {file_path}")
        return results
    
    # Load CSV
    try:
        data = pd.read_csv(file_path)
    except Exception as e:
        results['valid'] = False
        results['errors'].append(f"Error reading file: {e}")
        return results
    
    # Required fields
    required_fields = ['PARCELID', 'ELEVATION', 'LAT', 'LON']
    
    # Check all required fields present
    missing_fields = [f for f in required_fields if f not in data.columns]
    if missing_fields:
        results['valid'] = False
        results['errors'].append(f"Missing required fields: {missing_fields}")
        return results
    
    # Validate PARCELID
    if data['PARCELID'].duplicated().any():
        results['valid'] = False
        results['errors'].append("Duplicate PARCELID values found")
    
    # Check against parcel data if provided
    if parcel_ids is not None:
        parcel_set = set(parcel_ids)
        elevation_set = set(data['PARCELID'])
        
        if parcel_set != elevation_set:
            missing_in_elevation = parcel_set - elevation_set
            extra_in_elevation = elevation_set - parcel_set
            
            if missing_in_elevation:
                results['valid'] = False
                results['errors'].append(
                    f"{len(missing_in_elevation)} parcel IDs in parcel data but not in elevation data"
                )
            
            if extra_in_elevation:
                results['warnings'].append(
                    f"{len(extra_in_elevation)} parcel IDs in elevation data but not in parcel data"
                )
    
    # Validate ELEVATION (reasonable range for feet)
    if (data['ELEVATION'] < -100).any() or (data['ELEVATION'] > 1000).any():
        results['warnings'].append("Elevation values outside typical range [-100, 1000] feet")
    
    # Validate LAT/LON
    if not data['LAT'].between(-90, 90).all():
        results['valid'] = False
        results['errors'].append("LAT values outside valid range [-90, 90]")
    
    if not data['LON'].between(-180, 180).all():
        results['valid'] = False
        results['errors'].append("LON values outside valid range [-180, 180]")
    
    # Collect statistics
    results['stats'] = {
        'n_parcels': len(data),
        'elevation_range': (data['ELEVATION'].min(), data['ELEVATION'].max()),
        'elevation_mean': data['ELEVATION'].mean(),
        'elevation_std': data['ELEVATION'].std(),
        'lat_range': (data['LAT'].min(), data['LAT'].max()),
        'lon_range': (data['LON'].min(), data['LON'].max()),
    }
    
    return results


def print_results(name: str, results: Dict[str, any]) -> None:
    """Print validation results."""
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    
    if results['valid']:
        print("✅ VALID")
    else:
        print("❌ INVALID")
    
    if results['errors']:
        print("\n🔴 Errors:")
        for error in results['errors']:
            print(f"  - {error}")
    
    if results['warnings']:
        print("\n⚠️  Warnings:")
        for warning in results['warnings']:
            print(f"  - {warning}")
    
    if results['stats']:
        print("\n📊 Statistics:")
        for key, value in results['stats'].items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            elif isinstance(value, tuple):
                if isinstance(value[0], float):
                    print(f"  {key}: {value[0]:.2f} - {value[1]:.2f}")
                else:
                    print(f"  {key}: {value[0]} - {value[1]}")
            elif isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Validate DIAS test data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--data',
        type=str,
        default='test_data',
        help='Directory containing test data'
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data)
    
    print("=" * 60)
    print("  DIAS Test Data Validation")
    print("=" * 60)
    print(f"\nData directory: {data_dir.absolute()}")
    
    # Find parcel data file
    parcel_file = None
    for ext in ['.dbf', '.csv']:
        candidate = data_dir / f"parcels_test{ext}"
        if candidate.exists():
            parcel_file = candidate
            break
    
    if not parcel_file:
        print("\n❌ Error: No parcel data file found (parcels_test.dbf or parcels_test.csv)")
        sys.exit(1)
    
    # Find elevation file
    elevation_file = data_dir / "elevations_test.csv"
    if not elevation_file.exists():
        print(f"\n❌ Error: Elevation file not found: {elevation_file}")
        sys.exit(1)
    
    # Validate parcel data
    print(f"\n📋 Validating parcel data: {parcel_file.name}")
    parcel_results = validate_parcel_data(parcel_file)
    print_results("Parcel Data Validation", parcel_results)
    
    # Get parcel IDs for cross-validation
    parcel_ids = None
    if parcel_results['valid']:
        try:
            if parcel_file.suffix.lower() == '.dbf':
                records = [dict(r) for r in DBF(str(parcel_file), encoding='utf-8')]
                parcel_ids = [r['PARCELID'] for r in records]
            else:
                df = pd.read_csv(parcel_file)
                parcel_ids = df['PARCELID'].tolist()
        except:
            pass
    
    # Validate elevation data
    print(f"\n📏 Validating elevation data: {elevation_file.name}")
    elevation_results = validate_elevation_data(elevation_file, parcel_ids)
    print_results("Elevation Data Validation", elevation_results)
    
    # Overall result
    print("\n" + "=" * 60)
    if parcel_results['valid'] and elevation_results['valid']:
        print("✅ ALL VALIDATION CHECKS PASSED")
        print("=" * 60)
        print("\nTest data is ready for use with DIAS.")
        sys.exit(0)
    else:
        print("❌ VALIDATION FAILED")
        print("=" * 60)
        print("\nPlease fix the errors above before using this data.")
        sys.exit(1)


if __name__ == '__main__':
    main()

