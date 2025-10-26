import requests
import base64
import os
import re
import pandas as pd
import numpy as np
import ast
from pandarallel import pandarallel
from tqdm import tqdm
import argparse
import google.generativeai as genai
from PIL import Image
from functools import partial
import threading
import collections
import time as global_time

# Global lock for safe file writing
file_lock = threading.Lock()

# Global rate limiter for API requests (15 requests per minute)
request_times = collections.deque()
rate_limit_lock = threading.Lock()


def wait_for_rate_limit():
    """Ensure we don't exceed 15 requests per minute"""
    with rate_limit_lock:
        current_time = global_time.time()

        # Remove requests older than 60 seconds
        while request_times and current_time - request_times[0] > 60:
            request_times.popleft()

        # If we have 15 requests in the last minute, wait
        if len(request_times) >= 15:
            wait_time = (
                60 - (current_time - request_times[0]) + 1
            )  # Add 1 second buffer
            print(f"Rate limit: waiting {wait_time:.1f}s...")
            global_time.sleep(wait_time)

            # Clean up old requests after waiting
            current_time = global_time.time()
            while request_times and current_time - request_times[0] > 60:
                request_times.popleft()

        # Record this request
        request_times.append(current_time)


def save_row_to_csv(row, file_path):
    """Save a single row to CSV file safely with threading lock"""
    with file_lock:
        try:
            # Check if file exists
            if os.path.exists(file_path):
                # Read existing data
                df_existing = pd.read_csv(file_path)
                # Find the row to update based on IMG_ID
                mask = df_existing["IMG_ID"] == row["IMG_ID"]
                if mask.any():
                    # Update existing row
                    for col in row.index:
                        df_existing.loc[mask, col] = row[col]
                else:
                    # Append new row if not found
                    df_existing = pd.concat(
                        [df_existing, row.to_frame().T], ignore_index=True
                    )
                df_existing.to_csv(file_path, index=False)
            else:
                # Create new file with this row
                row.to_frame().T.to_csv(file_path, index=False)
            print(f"✅ Saved row {row['IMG_ID']} to {file_path}")
        except Exception as e:
            print(f"❌ Error saving row {row['IMG_ID']}: {e}")


def check_conditions(coord_str):
    """Check if coordinates are empty or None"""
    if isinstance(coord_str, str):
        if (
            coord_str.startswith("[]")
            or coord_str.startswith("None")
            or coord_str == "[]"
        ):
            return True
    elif isinstance(coord_str, list):
        if len(coord_str) == 0:
            return True
    try:
        coords = ast.literal_eval(str(coord_str))
        return len(coords) == 0
    except:
        return True


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_response(
    image_path,
    base_url,
    api_key,
    model_name,
    detail="low",
    max_tokens=200,
    temperature=1.2,
    n=10,
):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    img = Image.open(image_path)

    prompt = """Suppose you are an expert in geo-localization, you have the ability to give two number GPS coordination given an image.
    Please give me the location of the given image.
    Remember, you must have an answer, just output your best guess, don't answer me that you can't give a location.
    Your answer should be in the following JSON format without any other information: {"latitude": float,"longitude": float}.
    Your answer should be in the following JSON format without any other information: {"latitude": float,"longitude": float}."""

    ans = []
    for _ in range(n):
        try:
            response = model.generate_content(
                [prompt, img],
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
            )
            ans.append(response.text)
        except Exception as e:
            print(f"Error generating response: {e}")
            ans.append('{"latitude": 0.0,"longitude": 0.0}')
    return ans


def get_response_rag(
    image_path,
    base_url,
    api_key,
    model_name,
    candidates_gps,
    reverse_gps,
    detail="low",
    max_tokens=200,
    temperature=1.2,
    n=10,
):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    img = Image.open(image_path)

    prompt = f"""Suppose you are an expert in geo-localization, Please analyze this image and give me a guess of the location.
    Your answer must be to the coordinates level in (latitude, longitude) format.
    For your reference, these are coordinates of some similar images: {candidates_gps}, and these are coordinates of some dissimilar images: {reverse_gps}.
    Remember, you must have an answer, just output your best guess, don't answer me that you can't give an location.
    Your answer should be in the following JSON format without any other information: {{"latitude": float,"longitude": float}}.
    Your answer should be in the following JSON format without any other information: {{"latitude": float,"longitude": float}}."""

    ans = []
    for _ in range(n):
        try:
            response = model.generate_content(
                [prompt, img],
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
            )
            ans.append(response.text)
        except Exception as e:
            print(f"Error generating response: {e}")
            ans.append('{"latitude": 0.0,"longitude": 0.0}')
    return ans


def process_row(row, base_url, api_key, model_name, image_path, result_file_path):
    import os
    import re
    import google.generativeai as genai
    from PIL import Image
    import time
    import ast

    def check_conditions_inline(coord_str):
        """Check if coordinates are empty or None - inline version"""
        if isinstance(coord_str, str):
            if (
                coord_str.startswith("[]")
                or coord_str.startswith("None")
                or coord_str == "[]"
            ):
                return True
        elif isinstance(coord_str, list):
            if len(coord_str) == 0:
                return True
        try:
            coords = ast.literal_eval(str(coord_str))
            return len(coords) == 0
        except:
            return True

    def save_row_to_csv_inline(row, file_path):
        """Save a single row to CSV file - inline version (simplified without locks)"""
        import pandas as pd

        try:
            # Check if file exists
            if os.path.exists(file_path):
                # Read existing data
                df_existing = pd.read_csv(file_path)
                # Find the row to update based on IMG_ID
                mask = df_existing["IMG_ID"] == row["IMG_ID"]
                if mask.any():
                    # Update existing row with proper dtype handling
                    for col in row.index:
                        if col in df_existing.columns:
                            # Convert column to object type to avoid dtype conflicts
                            if df_existing[col].dtype != "object":
                                df_existing[col] = df_existing[col].astype("object")
                            df_existing.loc[mask, col] = str(row[col])
                        else:
                            # Add new column if it doesn't exist
                            df_existing[col] = None
                            df_existing.loc[mask, col] = str(row[col])
                else:
                    # Append new row if not found - convert row to string values
                    row_str = row.astype(str)
                    df_existing = pd.concat(
                        [df_existing, row_str.to_frame().T], ignore_index=True
                    )
                df_existing.to_csv(file_path, index=False)
            else:
                # Create new file with this row - convert to string values
                row_str = row.astype(str)
                row_str.to_frame().T.to_csv(file_path, index=False)
            print(f"✅ Saved row {row['IMG_ID']} to {file_path}")
        except Exception as e:
            print(f"❌ Error saving row {row['IMG_ID']}: {e}")

    # Check if coordinates already exist and are not empty
    if "coordinates" in row and not check_conditions_inline(row["coordinates"]):
        print(
            f"⏭ Skipping {row['IMG_ID']} - coordinates already exist: {row['coordinates']}"
        )
        return row

    image_file_path = os.path.join(image_path, row["IMG_ID"])
    try:
        # Inline get_response function to avoid import issues
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        img = Image.open(image_file_path)

        prompt = """Suppose you are an expert in geo-localization, you have the ability to give two number GPS coordination given an image.
        Please give me the location of the given image.
        Remember, you must have an answer, just output your best guess, don't answer me that you can't give a location.
        Your answer should be in the following JSON format without any other information: {"latitude": float,"longitude": float}.
        Your answer should be in the following JSON format without any other information: {"latitude": float,"longitude": float}."""

        ans = []
        # Reduce to 1 response per image to avoid rate limits (was 10)
        for attempt in range(1):
            max_retries = 5
            for retry in range(max_retries):
                try:
                    resp = model.generate_content(
                        [prompt, img],
                        generation_config=genai.types.GenerationConfig(
                            temperature=1.2,
                            max_output_tokens=200,
                        ),
                    )

                    # Check if response has valid content
                    if resp.candidates and len(resp.candidates) > 0:
                        candidate = resp.candidates[0]
                        if candidate.finish_reason == 1:  # STOP (successful completion)
                            # Extract JSON from response text (from first { to last })
                            response_text = resp.text
                            first_brace = response_text.find("{")
                            last_brace = response_text.rfind("}")
                            if (
                                first_brace != -1
                                and last_brace != -1
                                and first_brace <= last_brace
                            ):
                                extracted_json = response_text[
                                    first_brace : last_brace + 1
                                ]
                                ans.append(extracted_json)
                            else:
                                print(
                                    f"No valid JSON found in response for {row['IMG_ID']}"
                                )
                                ans.append('{"latitude": 0.0,"longitude": 0.0}')
                        else:
                            # Handle blocked/filtered responses
                            print(
                                f"Response blocked for {row['IMG_ID']}, finish_reason: {candidate.finish_reason}"
                            )
                            ans.append('{"latitude": 0.0,"longitude": 0.0}')
                    else:
                        print(f"No valid response for {row['IMG_ID']}")
                        ans.append('{"latitude": 0.0,"longitude": 0.0}')

                    break  # Success, exit retry loop
                except Exception as e:
                    error_msg = str(e)
                    if "429" in error_msg or "quota" in error_msg.lower():
                        if retry < max_retries - 1:
                            # Exponential backoff: 2^(retry+1) seconds (2, 4, 8, 16, 32)
                            wait_time = 2 ** (retry + 3)
                            print(
                                f"Rate limit hit for {row['IMG_ID']}, waiting {wait_time}s (attempt {retry + 1}/{max_retries})..."
                            )
                            time.sleep(wait_time)
                        else:
                            print(f"Max retries reached for {row['IMG_ID']}: {e}")
                            ans.append('{"latitude": 0.0,"longitude": 0.0}')
                    else:
                        print(f"Error generating response for {row['IMG_ID']}: {e}")
                        ans.append('{"latitude": 0.0,"longitude": 0.0}')
                        break
        response = ans
    except Exception as e:
        response = "None"
        print(e)

    row["response"] = response

    # Extract coordinates immediately after getting response
    pattern = r"[-+]?\d+\.\d+"
    coordinates = re.findall(pattern, str(response))
    row["coordinates"] = coordinates

    # Warning if extraction failed
    if len(coordinates) == 0:
        print(
            f"⚠ Warning: No coordinates extracted for IMG_ID={row['IMG_ID']}, response: {str(response)[:100]}..."
        )

    # Save this row immediately after processing
    save_row_to_csv_inline(row, result_file_path)

    return row


def process_row_rag(
    row,
    base_url,
    api_key,
    model_name,
    image_path,
    rag_sample_num,
    rag_file_path,
):
    import os
    import re
    import google.generativeai as genai
    from PIL import Image
    import time
    import ast

    def check_conditions_inline(coord_str):
        """Check if coordinates are empty or None - inline version"""
        if isinstance(coord_str, str):
            if (
                coord_str.startswith("[]")
                or coord_str.startswith("None")
                or coord_str == "[]"
            ):
                return True
        elif isinstance(coord_str, list):
            if len(coord_str) == 0:
                return True
        try:
            coords = ast.literal_eval(str(coord_str))
            return len(coords) == 0
        except:
            return True

    def save_row_to_csv_inline(row, file_path):
        """Save a single row to CSV file - inline version (simplified without locks)"""
        import pandas as pd

        try:
            # Check if file exists
            if os.path.exists(file_path):
                # Read existing data
                df_existing = pd.read_csv(file_path)
                # Find the row to update based on IMG_ID
                mask = df_existing["IMG_ID"] == row["IMG_ID"]
                if mask.any():
                    # Update existing row with proper dtype handling
                    for col in row.index:
                        if col in df_existing.columns:
                            # Convert column to object type to avoid dtype conflicts
                            if df_existing[col].dtype != "object":
                                df_existing[col] = df_existing[col].astype("object")
                            df_existing.loc[mask, col] = str(row[col])
                        else:
                            # Add new column if it doesn't exist
                            df_existing[col] = None
                            df_existing.loc[mask, col] = str(row[col])
                else:
                    # Append new row if not found - convert row to string values
                    row_str = row.astype(str)
                    df_existing = pd.concat(
                        [df_existing, row_str.to_frame().T], ignore_index=True
                    )
                df_existing.to_csv(file_path, index=False)
            else:
                # Create new file with this row - convert to string values
                row_str = row.astype(str)
                row_str.to_frame().T.to_csv(file_path, index=False)
            print(f"✅ Saved row {row['IMG_ID']} to {file_path}")
        except Exception as e:
            print(f"❌ Error saving row {row['IMG_ID']}: {e}")

    # Check if coordinates already exist and are not empty
    # For RAG prediction, we expect coordinates to be strings like "[gps1], [gps2], ..."

    # Check if rag_coordinates already exist and are not empty
    if "rag_coordinates" in row and not check_conditions_inline(row["rag_coordinates"]):
        print(
            f"⏭ Skipping RAG {row['IMG_ID']} - coordinates already exist: {row['rag_coordinates']}"
        )
        return row

    image_file_path = os.path.join(image_path, row["IMG_ID"])
    try:
        # candidates_gps = [eval(row[f'candidate_{i}_gps']) for i in range(rag_sample_num)]
        candidates_gps = [row[f"candidate_{i}_gps"] for i in range(rag_sample_num)]
        candidates_gps = str(candidates_gps)
        # reverse_gps = [eval(row[f'reverse_{i}_gps']) for i in range(rag_sample_num)]
        reverse_gps = [row[f"reverse_{i}_gps"] for i in range(rag_sample_num)]
        reverse_gps = str(reverse_gps)

        # Inline get_response_rag function to avoid import issues
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        img = Image.open(image_file_path)

        prompt = f"""Suppose you are an expert in geo-localization, Please analyze this image and give me a guess of the location.
        Your answer must be to the coordinates level in (latitude, longitude) format.
        For your reference, these are coordinates of some similar images: {candidates_gps}, and these are coordinates of some dissimilar images: {reverse_gps}.
        Remember, you must have an answer, just output your best guess, don't answer me that you can't give an location.
        Your answer should be in the following JSON format without any other information: {{"latitude": float,"longitude": float}}.
        Your answer should be in the following JSON format without any other information: {{"latitude": float,"longitude": float}}."""

        ans = []
        # Reduce to 1 response per image to avoid rate limits (was 10)
        for attempt in range(1):
            max_retries = 5
            for retry in range(max_retries):
                try:
                    resp = model.generate_content(
                        [prompt, img],
                        generation_config=genai.types.GenerationConfig(
                            temperature=1.2,
                            max_output_tokens=200,
                        ),
                    )

                    # Check if response has valid content
                    if resp.candidates and len(resp.candidates) > 0:
                        candidate = resp.candidates[0]
                        if candidate.finish_reason == 1:  # STOP (successful completion)
                            # Extract JSON from response text (from first { to last })
                            response_text = resp.text
                            first_brace = response_text.find("{")
                            last_brace = response_text.rfind("}")
                            if (
                                first_brace != -1
                                and last_brace != -1
                                and first_brace <= last_brace
                            ):
                                extracted_json = response_text[
                                    first_brace : last_brace + 1
                                ]
                                ans.append(extracted_json)
                            else:
                                print(
                                    f"RAG no valid JSON found in response for {row['IMG_ID']}"
                                )
                                ans.append('{"latitude": 0.0,"longitude": 0.0}')
                        else:
                            # Handle blocked/filtered responses
                            print(
                                f"RAG response blocked for {row['IMG_ID']}, finish_reason: {candidate.finish_reason}"
                            )
                            ans.append('{"latitude": 0.0,"longitude": 0.0}')
                    else:
                        print(f"RAG no valid response for {row['IMG_ID']}")
                        ans.append('{"latitude": 0.0,"longitude": 0.0}')

                    # print(f"RAG response for {row['IMG_ID']}: {resp.text}")
                    break  # Success, exit retry loop
                except Exception as e:
                    error_msg = str(e)
                    if "429" in error_msg or "quota" in error_msg.lower():
                        if retry < max_retries - 1:
                            # Exponential backoff: 2^(retry+1) seconds (2, 4, 8, 16, 32)
                            wait_time = 2 ** (retry + 3)
                            print(
                                f"RAG Rate limit hit for {row['IMG_ID']}, waiting {wait_time}s (attempt {retry + 1}/{max_retries})..."
                            )
                            time.sleep(wait_time)
                        else:
                            print(f"Max retries reached for {row['IMG_ID']}: {e}")
                            ans.append('{"latitude": 0.0,"longitude": 0.0}')
                    else:
                        print(f"Error generating RAG response for {row['IMG_ID']}: {e}")
                        ans.append('{"latitude": 0.0,"longitude": 0.0}')
                        break
        response = ans
    except Exception as e:
        response = "None"
        print(e)

    row["rag_response"] = response

    # Extract coordinates immediately after getting RAG response
    pattern = r"[-+]?\d+\.\d+"
    rag_coordinates = re.findall(pattern, str(response))
    row["rag_coordinates"] = rag_coordinates

    # Warning if extraction failed
    if len(rag_coordinates) == 0:
        print(
            f"⚠ Warning: No RAG coordinates extracted for IMG_ID={row['IMG_ID']}, response: {str(response)[:100]}..."
        )

    # Save this row immediately after processing
    save_row_to_csv_inline(row, rag_file_path)

    return row


def run(args):
    api_key = args.api_key
    model_name = args.model_name
    base_url = args.base_url

    text_path = args.text_path
    image_path = args.image_path
    output_dir = args.output_dir
    result_filename = args.result_filename
    rag_filename = args.rag_filename
    process = args.process
    rag_sample_num = args.rag_sample_num
    searching_file_name = args.searching_file_name

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Construct full paths
    result_path = os.path.join(output_dir, result_filename)

    if process == "predict":
        print("=" * 80)
        print(
            "Running LLM Predictions with Real-time Coordinate Extraction (Zero-Shot)"
        )
        print("=" * 80)

        if os.path.exists(result_path):
            df = pd.read_csv(result_path)
            # Filter for rows that need processing: empty coordinates or response
            if "coordinates" not in df.columns:
                df["coordinates"] = "[]"  # Initialize if not exists
            df_rerun = df[df["coordinates"].apply(check_conditions)]
            print(f"Need to process: {df_rerun.shape[0]} rows with empty coordinates")
            if df_rerun.shape[0] > 0:
                print("⏳ Processing only rows with empty coordinates...")
                df_rerun = df_rerun.parallel_apply(
                    partial(
                        process_row,
                        base_url=base_url,
                        api_key=api_key,
                        model_name=model_name,
                        image_path=image_path,
                        result_file_path=result_path,
                    ),
                    axis=1,
                )
                df.update(df_rerun)
        else:
            df = pd.read_csv(text_path)
            df["coordinates"] = "[]"  # Initialize all as empty
            print(
                f"⏳ Processing {df.shape[0]} images with real-time coordinate extraction..."
            )
            df = df.parallel_apply(
                partial(
                    process_row,
                    base_url=base_url,
                    api_key=api_key,
                    model_name=model_name,
                    image_path=image_path,
                    result_file_path=result_path,
                ),
                axis=1,
            )

        # Check extraction results
        print("\n" + "=" * 80)
        print("Summary")
        print("=" * 80)
        empty_coords = df[df["coordinates"].apply(lambda x: len(x) == 0)]
        if len(empty_coords) > 0:
            print(f"⚠ WARNING: {len(empty_coords)} rows have empty coordinates!")
        else:
            print(f"✓ All {df.shape[0]} rows successfully extracted coordinates")

        # Save with coordinates
        df.to_csv(result_path, index=False)
        print(f"✓ Results with coordinates saved to {result_path}")
        print("=" * 80)

    if process == "rag":
        print("=" * 80)
        print("Running RAG Predictions with Real-time Coordinate Extraction")
        print("=" * 80)

        database_df = pd.read_csv("./data/MP16_Pro_filtered.csv")
        rag_file_path = os.path.join(output_dir, f"{rag_sample_num}_{rag_filename}")

        if not os.path.exists(rag_file_path):
            print("⏳ Loading indices and preparing candidate GPS coordinates...")
            df = pd.read_csv(text_path)
            I = np.load("./index/{}.npy".format(searching_file_name))
            reverse_I = np.load("./index/{}_reverse.npy".format(searching_file_name))

            # Initialize all GPS candidate columns first to ensure consistent column count
            for idx in range(rag_sample_num):
                df[f"candidate_{idx}_gps"] = ""
                df[f"reverse_{idx}_gps"] = ""

            # Check if file already exists and read it to continue from where we left off
            if os.path.exists(rag_file_path):
                existing_df = pd.read_csv(rag_file_path)
                # Find which rows already have GPS candidates prepared
                completed_rows = existing_df[
                    existing_df["candidate_0_gps"] != ""
                ].index.tolist()
                print(
                    f"Found {len(completed_rows)} rows already prepared, continuing from row {len(completed_rows)}"
                )
                df = existing_df.copy()
            else:
                completed_rows = []

            for i in tqdm(range(df.shape[0]), desc="Preparing GPS candidates"):
                # Skip if this row is already completed
                if i in completed_rows:
                    continue

                try:
                    candidate_idx_lis = I[i]
                    candidate_gps = database_df.loc[
                        candidate_idx_lis, ["LAT", "LON", "city", "state", "country"]
                    ].values
                    # Only process up to rag_sample_num candidates to ensure consistent columns
                    for idx, (latitude, longitude, city, state, country) in enumerate(
                        candidate_gps[:rag_sample_num]  # Limit to rag_sample_num
                    ):
                        df.loc[i, f"candidate_{idx}_gps"] = f"[{latitude}, {longitude}]"

                    reverse_idx_lis = reverse_I[i]
                    reverse_gps = database_df.loc[
                        reverse_idx_lis, ["LAT", "LON", "city", "state", "country"]
                    ].values
                    # Only process up to rag_sample_num reverse candidates to ensure consistent columns
                    for idx, (latitude, longitude, city, state, country) in enumerate(
                        reverse_gps[:rag_sample_num]  # Limit to rag_sample_num
                    ):
                        df.loc[i, f"reverse_{idx}_gps"] = f"[{latitude}, {longitude}]"

                    # Save immediately after preparing each row
                    df.to_csv(rag_file_path, index=False)

                    # Print progress every 100 rows
                    if (i + 1) % 100 == 0:
                        print(f"✅ GPS candidates prepared and saved for {i + 1} rows")

                except Exception as e:
                    print(f"❌ Error preparing GPS for row {i}: {e}")
                    continue

            print("✓ All GPS candidates prepared and saved")

            # df["rag_coordinates"] = "[]"  # Initialize all as empty
            # print(
            #     f"\n⏳ Running RAG predictions for {df.shape[0]} images with real-time coordinate extraction..."
            # )
            # # Use tqdm for better progress tracking during inference
            # df = df.parallel_apply(
            #     partial(
            #         process_row_rag,
            #         base_url=base_url,
            #         api_key=api_key,
            #         model_name=model_name,
            #         root_path=root_path,
            #         image_path=image_path,
            #         rag_sample_num=rag_sample_num,
            #         rag_file_path=rag_file_path,
            #     ),
            #     axis=1,
            # )
            # df.to_csv(rag_file_path, index=False)
        else:
            df = pd.read_csv(rag_file_path)

            # Check if rag_coordinates column exists and filter for empty ones
            if "rag_coordinates" not in df.columns:
                df["rag_coordinates"] = "[]"  # Initialize if not exists
            df_rerun = df[df["rag_coordinates"].apply(check_conditions)]
            print(
                f"Need to process: {df_rerun.shape[0]} rows with empty RAG coordinates"
            )
            if df_rerun.shape[0] > 0:
                print("⏳ Processing only rows with empty RAG coordinates...")
                df = df.parallel_apply(
                    partial(
                        process_row_rag,
                        base_url=base_url,
                        api_key=api_key,
                        model_name=model_name,
                        image_path=image_path,
                        rag_sample_num=rag_sample_num,
                        rag_file_path=rag_file_path,
                    ),
                    axis=1,
                )
                df.to_csv(rag_file_path, index=False)
            else:
                # Filter for rows with empty RAG coordinates
                if "rag_coordinates" not in df.columns:
                    df["rag_coordinates"] = "[]"
                df_rerun = df[df["rag_coordinates"].apply(check_conditions)]
                print(
                    f"Need Rerun: {df_rerun.shape[0]} rows with empty RAG coordinates"
                )

                if df_rerun.shape[0] > 0:
                    print("⏳ Processing only rows with empty RAG coordinates...")
                    df_rerun = df_rerun.parallel_apply(
                        partial(
                            process_row_rag,
                            base_url=base_url,
                            api_key=api_key,
                            model_name=model_name,
                            image_path=image_path,
                            rag_sample_num=rag_sample_num,
                            rag_file_path=rag_file_path,
                        ),
                        axis=1,
                    )
                    df.update(df_rerun)
                    df.to_csv(rag_file_path, index=False)

        # Check extraction results
        print("\n" + "=" * 80)
        print("Summary")
        print("=" * 80)

        empty_coords = df[df["rag_coordinates"].apply(lambda x: len(x) == 0)]
        if len(empty_coords) > 0:
            print(f"⚠ WARNING: {len(empty_coords)} rows have empty RAG coordinates!")
        else:
            print(f"✓ All {df.shape[0]} rows successfully extracted RAG coordinates")

        # Save with coordinates
        df.to_csv(rag_file_path, index=False)
        print(f"✓ Results with RAG coordinates saved to {rag_file_path}")
        print("=" * 80)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    args = argparse.ArgumentParser()
    api_key = os.getenv("API_KEY")
    model_name = "gemini-2.0-flash"  # or gemini-2.0-flash
    base_url = ""  # Not used for Gemini

    text_path = "./data/im2gps3k/im2gps3k_places365.csv"
    image_path = "./data/im2gps3k/images"
    output_dir = "./results"
    result_filename = "llm_predict_results_zs.csv"
    rag_filename = "llm_predict_results_rag.csv"
    process = "rag"  # predict, extract, rag, rag_extract
    rag_sample_nums = [15, 10, 5, 0]  # List of rag_sample_num values to run
    searching_file_name = "I_g3_im2gps3k"

    # Set to 1 worker to respect 15 requests/minute limit with 4-second delays
    pandarallel.initialize(progress_bar=True, nb_workers=8)
    args.add_argument("--api_key", type=str, default=api_key)
    args.add_argument("--model_name", type=str, default=model_name)
    args.add_argument("--base_url", type=str, default=base_url)

    args.add_argument("--text_path", type=str, default=text_path)
    args.add_argument("--image_path", type=str, default=image_path)
    args.add_argument("--output_dir", type=str, default=output_dir)
    args.add_argument("--result_filename", type=str, default=result_filename)
    args.add_argument("--rag_filename", type=str, default=rag_filename)
    args.add_argument("--process", type=str, default=process)
    args.add_argument("--rag_sample_nums", type=list, default=rag_sample_nums)
    args.add_argument("--searching_file_name", type=str, default=searching_file_name)
    args = args.parse_args()
    print(args)

    # Run for each rag_sample_num in the list
    for rag_sample_num in rag_sample_nums:
        print(f"\n{'=' * 100}")
        print(f"🚀 RUNNING WITH RAG_SAMPLE_NUM = {rag_sample_num}")
        print(f"{'=' * 100}")

        # Create a copy of args with current rag_sample_num
        current_args = argparse.Namespace(**vars(args))
        current_args.rag_sample_num = rag_sample_num

        run(current_args)

        print(f"\n✅ COMPLETED RAG_SAMPLE_NUM = {rag_sample_num}")

    print(f"\n{'=' * 100}")
    print("🎉 ALL RAG_SAMPLE_NUM VALUES COMPLETED!")
    print(f"Processed values: {rag_sample_nums}")
    print(f"{'=' * 100}")
