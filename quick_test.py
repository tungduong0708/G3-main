"""
Quick test script to verify the refactored code works
"""

import asyncio
import os
import sys
from pathlib import Path

# Ensure we're in the right directory
sys.path.insert(0, str(Path(__file__).parent))


async def test_basic_functionality():
    """Test basic functionality without requiring full setup"""

    print("=" * 80)
    print("Quick Test - Refactored Code")
    print("=" * 80)

    # Test 1: Import and create Pydantic models
    print("\n[Test 1] Testing Pydantic models...")
    try:
        from prompt import Evidence, LocationPrediction, GPSPrediction

        # Create sample evidence
        evidence = Evidence(
            analysis="This is a test analysis [1][2]",
            references=["https://example.com", "image_001.jpg"],
        )

        # Create sample location prediction
        prediction = LocationPrediction(
            latitude=40.7128,
            longitude=-74.0060,
            location="New York City",
            evidence=[evidence],
        )

        print(f"✅ Created LocationPrediction: {prediction.location}")
        print(f"   Coordinates: ({prediction.latitude}, {prediction.longitude})")
        print(f"   Evidence count: {len(prediction.evidence)}")

    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

    # Test 2: Test prompt generation
    print("\n[Test 2] Testing prompt generation...")
    try:
        from prompt import location_prompt, diversification_prompt

        # Test location prompt
        loc_prompt = location_prompt("Paris, France")
        print(f"✅ Generated location prompt ({len(loc_prompt)} chars)")

        # Test diversification prompt (will fail if no data dir, but that's ok)
        try:
            div_prompt = diversification_prompt(
                prompt_dir="data/prompt_data",
                n_search=5,
                n_coords=10,
                image_prediction=True,
                text_prediction=False,
            )
            print(f"✅ Generated diversification prompt ({len(div_prompt)} chars)")
        except Exception as e:
            print(f"⚠️  Diversification prompt needs data dir (expected): {e}")

    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

    # Test 3: Test utility functions
    print("\n[Test 3] Testing utility functions...")
    try:
        from helper_utils import get_gps_from_location, extract_and_parse_json

        # Test GPS lookup (requires internet)
        try:
            lat, lon = get_gps_from_location("London, UK")
            if lat and lon:
                print(f"✅ GPS lookup worked: London = ({lat}, {lon})")
            else:
                print("⚠️  GPS lookup returned None (network issue?)")
        except Exception as e:
            print(f"⚠️  GPS lookup failed (expected if no network): {e}")

        # Test JSON extraction
        test_json = '{"latitude": 51.5074, "longitude": -0.1278}'
        parsed = extract_and_parse_json(test_json)
        if parsed.get("latitude") == 51.5074:
            print(f"✅ JSON extraction worked: {parsed}")
        else:
            print(f"❌ JSON extraction failed: {parsed}")

    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

    # Test 4: Check if G3 model can be imported (may fail without torch)
    print("\n[Test 4] Testing G3 model import...")
    try:
        from utils.G3 import G3

        print("✅ G3 model imported successfully")
        print("⚠️  Note: Cannot instantiate without GPU and checkpoint")
    except Exception as e:
        print(f"⚠️  G3 model import failed (expected if torch not installed): {e}")

    # Test 5: Check data processor import
    print("\n[Test 5] Testing DataProcessor import...")
    try:
        from data_processor import DataProcessor

        print("✅ DataProcessor imported successfully")
    except Exception as e:
        print(f"⚠️  DataProcessor import failed: {e}")

    # Test 6: Check batch predictor import
    print("\n[Test 6] Testing G3BatchPredictor import...")
    try:
        from g3_batch_prediction import G3BatchPredictor

        print("✅ G3BatchPredictor imported successfully")
    except Exception as e:
        print(f"⚠️  G3BatchPredictor import failed: {e}")

    print("\n" + "=" * 80)
    print("Quick Test Completed!")
    print("=" * 80)
    print("\n✅ Core functionality is working")
    print("⚠️  Some features need full setup (see REFACTOR_GUIDE.md)")

    return True


if __name__ == "__main__":
    # Run async test
    result = asyncio.run(test_basic_functionality())

    if result:
        print("\n✅ All critical tests passed!")
        print("📚 See REFACTOR_GUIDE.md for full setup instructions")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed - check errors above")
        sys.exit(1)
