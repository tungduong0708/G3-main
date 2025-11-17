"""
Main script to run complete pipeline on im2gps3k dataset with Gemini 2.0 Flash
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

import pandas as pd
from dotenv import load_dotenv
from geopy.distance import geodesic

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from g3_batch_prediction import G3BatchPredictor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("im2gps3k_pipeline.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class Im2GPS3KPipeline:
    """Complete pipeline for im2gps3k dataset prediction"""

    def __init__(
        self,
        device: str = "cuda",
        model_name: str = "gemini-2.0-flash-exp",
        checkpoint_path: str = "checkpoints/g3.pth",
        index_path: str = "index/I_g3_im2gps3k.npy",
        database_csv_path: str = "checkpoints/im2gps3k_places365.csv",
        ground_truth_csv_path: str = "data/im2gps3k/im2gps3k_places365.csv",
        input_dir: str = "data/im2gps3k/test_images",
        output_dir: str = "results/im2gps3k",
    ):
        """
        Initialize pipeline

        Args:
            device: 'cuda' or 'cpu'
            model_name: Gemini model name
            checkpoint_path: Path to G3 model checkpoint
            index_path: Path to FAISS index
            database_csv_path: Path to database CSV
            ground_truth_csv_path: Path to ground truth CSV with actual LAT/LON
            input_dir: Directory containing test images
            output_dir: Directory to save results
        """
        self.device = device
        self.model_name = model_name
        self.input_dir = Path(input_dir)  # Original dataset directory (READ-ONLY)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Store configuration for parallel processing
        self.checkpoint_path = checkpoint_path
        self.index_path = index_path
        self.database_csv_path = database_csv_path

        # Load ground truth data
        self.ground_truth_df = pd.read_csv(ground_truth_csv_path)
        logger.info(f"Loaded ground truth data: {len(self.ground_truth_df)} images")

        # Create temporary processing directory (separate from dataset)
        self.temp_input_dir = self.output_dir / "temp_input"
        self.temp_input_dir.mkdir(parents=True, exist_ok=True)

        # Initialize predictor with TEMPORARY directory, NOT the dataset directory
        logger.info(f"Initializing G3BatchPredictor with {model_name}")
        logger.info(f"Dataset directory (READ-ONLY): {self.input_dir}")
        logger.info(f"Processing directory (temporary): {self.temp_input_dir}")
        self.predictor = G3BatchPredictor(
            device=device,
            input_dir=str(self.temp_input_dir),  # Use temporary directory
            prompt_dir=str(self.output_dir / "prompt_data"),
            cache_dir=str(self.output_dir / "cache"),
            index_path=index_path,
            database_csv_path=database_csv_path,
            checkpoint_path=checkpoint_path,
        )

        # Results storage
        self.results = []

        # Single CSV file for all results (without timestamp for consistency)
        self.combined_csv_path = self.output_dir / "predictions_all.csv"
        if self.combined_csv_path.exists():
            logger.info(f"Using existing combined CSV: {self.combined_csv_path}")
        else:
            logger.info(
                f"No existing combined CSV found. A new one will be created at: {self.combined_csv_path}"
            )

        # Load already processed images if CSV exists
        self.num_processed = 0
        self.processed_images = set()
        if self.combined_csv_path.exists():
            try:
                existing_df = pd.read_csv(self.combined_csv_path)
                self.num_processed = len(existing_df)
                self.processed_images = set(existing_df["IMG_ID"].tolist())
                logger.info(
                    f"Found existing results: {self.num_processed} images already processed"
                )
                logger.info(f"Will continue from image {self.num_processed + 1}")
            except Exception as e:
                logger.warning(f"Could not read existing CSV: {e}")
                self.num_processed = 0
                self.processed_images = set()

    async def predict_single_image(
        self,
        image_name: str,
        image_prediction: bool = True,
        text_prediction: bool = True,
    ):
        """
        Predict location for a single image

        Args:
            image_name: Name of the image file
            image_prediction: Use image-based prediction
            text_prediction: Use text-based prediction

        Returns:
            dict: Prediction results
        """
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Processing: {image_name}")
        logger.info(f"{'=' * 80}")

        try:
            # Run prediction
            result = await self.predictor.predict(
                model_name=self.model_name,
                image_prediction=image_prediction,
                text_prediction=text_prediction,
            )

            # Convert to dict and add metadata
            result_dict = result.model_dump()
            result_dict["image_id"] = image_name
            result_dict["timestamp"] = datetime.now().isoformat()
            result_dict["model_name"] = self.model_name
            result_dict["success"] = True

            logger.info(f"[OK] Success: {image_name}")
            logger.info(f"   Location: {result_dict['location']}")
            logger.info(
                f"   Coordinates: ({result_dict['latitude']}, {result_dict['longitude']})"
            )
            logger.info(f"   Evidence count: {len(result_dict['evidence'])}")

            return result_dict

        except Exception as e:
            logger.error(f"[FAIL] Failed: {image_name} - {e}")
            return {
                "image_id": image_name,
                "latitude": 0.0,
                "longitude": 0.0,
                "location": "",
                "evidence": [],
                "timestamp": datetime.now().isoformat(),
                "model_name": self.model_name,
                "success": False,
                "error": str(e),
            }

    async def run_batch_prediction(
        self,
        image_ids: list = None,
        max_images: int = None,
        start_index: int = None,
        end_index: int = None,
        image_prediction: bool = True,
        text_prediction: bool = True,
        max_parallel: int = 3,
    ):
        """
        Run prediction on multiple images - each image is processed independently

        Args:
            image_ids: List of specific image IDs to process
            max_images: Maximum number of images to process (from current position)
            start_index: Starting index (0-based) in the sorted image list
            end_index: Ending index (exclusive) in the sorted image list
            image_prediction: Use image-based prediction
            text_prediction: Use text-based prediction
            max_parallel: Maximum number of images to process in parallel (default: 3)

        Note:
            - If start_index/end_index are specified, they override num_processed and max_images
            - If only max_images is specified, it starts from the last processed position
        """
        logger.info("\n" + "=" * 80)
        logger.info("Starting Batch Prediction Pipeline")
        logger.info(f"Parallel processing enabled: {max_parallel} concurrent images")
        logger.info("=" * 80)

        # Get list of images to process from DATASET directory
        if image_ids:
            logger.info(f"Looking for specific images: {image_ids}")
            images_to_process = []
            for img_id in image_ids:
                img_path = self.input_dir / img_id
                if img_path.exists():
                    images_to_process.append(img_path)
                    logger.info(f"  Found: {img_id}")
                else:
                    logger.warning(f"  Not found: {img_id}")
        else:
            # Get all images and sort for consistent order
            all_images = sorted(list(self.input_dir.glob("*.jpg")))
            total_images = len(all_images)

            # Determine which images to process based on parameters
            if start_index is not None or end_index is not None:
                # Use explicit start/end indices
                start = start_index if start_index is not None else 0
                end = end_index if end_index is not None else total_images

                # Validate indices
                if start < 0 or start >= total_images:
                    logger.error(
                        f"Invalid start_index: {start} (total images: {total_images})"
                    )
                    return
                if end < 0 or end > total_images:
                    logger.error(
                        f"Invalid end_index: {end} (total images: {total_images})"
                    )
                    return
                if start >= end:
                    logger.error(
                        f"start_index ({start}) must be less than end_index ({end})"
                    )
                    return

                images_in_range = all_images[start:end]
                logger.info(
                    f"Selected range: index {start} to {end - 1} ({len(images_in_range)} images)"
                )

                # Filter out already processed images
                images_to_process = [
                    img
                    for img in images_in_range
                    if img.name not in self.processed_images
                ]

                skipped_count = len(images_in_range) - len(images_to_process)
                if skipped_count > 0:
                    logger.info(
                        f"Skipping {skipped_count} already processed images in this range"
                    )
                logger.info(
                    f"Will process {len(images_to_process)} new images from index {start} to {end - 1}"
                )
            else:
                # Use num_processed to continue from where we left off
                # Get all remaining images after the processed ones
                remaining_images = all_images[self.num_processed :]

                # Filter out any already processed images (in case CSV was modified)
                images_to_process = [
                    img
                    for img in remaining_images
                    if img.name not in self.processed_images
                ]

                # Limit to max_images if specified
                if max_images:
                    images_to_process = images_to_process[:max_images]

                logger.info(
                    f"Starting from image index {self.num_processed} (continuing from last run)"
                )
                logger.info(
                    f"Will process {len(images_to_process)} images in this batch"
                )

        if not images_to_process:
            logger.info("No images to process!")
            if start_index is not None or end_index is not None:
                logger.info(
                    f"Check your start_index ({start_index}) and end_index ({end_index})"
                )
            else:
                logger.info("All images have already been processed!")
                logger.info(f"Total processed: {self.num_processed} images")
            logger.info(f"Check results in: {self.combined_csv_path}")
            self.generate_final_summary()
            return

        # Process images in parallel batches
        import shutil

        async def process_image_task(img_path: Path, idx: int):
            """Process a single image as an independent task"""
            logger.info(f"\n{'=' * 80}")
            logger.info(f"[{idx}/{len(images_to_process)}] Processing: {img_path.name}")
            logger.info(f"{'=' * 80}")

            # Create unique temporary directories for this task
            task_temp_dir = self.output_dir / f"temp_{img_path.stem}"
            task_prompt_dir = self.output_dir / f"prompt_{img_path.stem}"

            try:
                # STEP 1: Clear and create temporary directories for this image
                if task_temp_dir.exists():
                    shutil.rmtree(task_temp_dir)
                task_temp_dir.mkdir(parents=True, exist_ok=True)

                if task_prompt_dir.exists():
                    shutil.rmtree(task_prompt_dir)
                task_prompt_dir.mkdir(parents=True, exist_ok=True)

                logger.info(f"Created isolated temp directories for {img_path.name}")

                # STEP 2: Copy ONLY this image to temp directory
                shutil.copy(img_path, task_temp_dir / img_path.name)
                logger.info(f"Copied {img_path.name} to processing directory")

                # STEP 3: Create a temporary predictor for this image
                temp_predictor = G3BatchPredictor(
                    device=self.device,
                    input_dir=str(task_temp_dir),
                    prompt_dir=str(task_prompt_dir),
                    cache_dir=str(self.output_dir / "cache"),
                    index_path=self.index_path,
                    database_csv_path=self.database_csv_path,
                    checkpoint_path=self.checkpoint_path,
                )

                # STEP 4: Run prediction for this single image
                result = await temp_predictor.predict(
                    model_name=self.model_name,
                    image_prediction=image_prediction,
                    text_prediction=text_prediction,
                )

                # Convert to dict and add metadata
                result_dict = result.model_dump()
                result_dict["image_id"] = img_path.name
                result_dict["timestamp"] = datetime.now().isoformat()
                result_dict["model_name"] = self.model_name
                result_dict["success"] = True

                logger.info(f"[OK] Success: {img_path.name}")
                logger.info(f"   Location: {result_dict['location']}")
                logger.info(
                    f"   Coordinates: ({result_dict['latitude']}, {result_dict['longitude']})"
                )
                logger.info(f"   Evidence count: {len(result_dict['evidence'])}")

                # STEP 5: Append result to combined CSV immediately
                self.append_result_to_csv(result_dict)
                logger.info(f"Appended result for {img_path.name} to CSV")

                return result_dict

            except Exception as e:
                logger.error(f"[FAIL] Failed: {img_path.name} - {e}")
                result_dict = {
                    "image_id": img_path.name,
                    "latitude": 0.0,
                    "longitude": 0.0,
                    "location": "",
                    "evidence": [],
                    "timestamp": datetime.now().isoformat(),
                    "model_name": self.model_name,
                    "success": False,
                    "error": str(e),
                }
                # Append error result to CSV
                self.append_result_to_csv(result_dict)
                return result_dict

            finally:
                # Clean up temporary directories after each image
                try:
                    if task_temp_dir.exists():
                        shutil.rmtree(task_temp_dir)
                        logger.info(f"Cleaned up temp directory: {task_temp_dir.name}")
                    if task_prompt_dir.exists():
                        shutil.rmtree(task_prompt_dir)
                        logger.info(
                            f"Cleaned up prompt directory: {task_prompt_dir.name}"
                        )
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup temp dirs: {cleanup_error}")

        # Process images in parallel batches using asyncio.Semaphore
        semaphore = asyncio.Semaphore(max_parallel)

        async def process_with_semaphore(img_path: Path, idx: int):
            async with semaphore:
                return await process_image_task(img_path, idx)

        # Create tasks for all images
        tasks = [
            process_with_semaphore(img_path, idx + 1)
            for idx, img_path in enumerate(images_to_process)
        ]

        # Run all tasks with progress bar
        results = []
        with tqdm(total=len(tasks), desc="Processing images", unit="image") as pbar:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                results.append(result)
                pbar.update(1)
                logger.info(
                    f"Progress: {len(results)}/{len(images_to_process)} completed"
                )

        logger.info("\n" + "=" * 80)
        logger.info("Batch Prediction Completed!")
        logger.info("=" * 80)

        # Generate final summary from the combined CSV file
        self.generate_final_summary()

    def append_result_to_csv(self, result: dict):
        """Append a single result to the combined CSV file"""
        # Get ground truth coordinates from dataset
        gt_row = self.ground_truth_df[
            self.ground_truth_df["IMG_ID"] == result["image_id"]
        ]

        if len(gt_row) > 0:
            lat_true = gt_row.iloc[0]["LAT"]
            lon_true = gt_row.iloc[0]["LON"]
        else:
            logger.warning(f"No ground truth found for {result['image_id']}")
            lat_true = 0.0
            lon_true = 0.0

        # Compute geodesic distance
        geodesic_dist = None
        if (
            result["success"]
            and result["latitude"] != 0.0
            and result["longitude"] != 0.0
        ):
            try:
                geodesic_dist = geodesic(
                    (lat_true, lon_true), (result["latitude"], result["longitude"])
                ).km
            except Exception:
                pass

        csv_row = {
            "IMG_ID": result["image_id"],
            "LAT": lat_true,
            "LON": lon_true,
            "LAT_pred": result["latitude"],
            "LON_pred": result["longitude"],
            "location": result["location"],
            "evidence_count": len(result["evidence"]),
            "success": result["success"],
            "model_name": result["model_name"],
            "timestamp": result["timestamp"],
            "geodesic": geodesic_dist,
        }

        # Create DataFrame for this row
        df_row = pd.DataFrame([csv_row])

        # Append to CSV (create with header if file doesn't exist)
        if not self.combined_csv_path.exists():
            df_row.to_csv(self.combined_csv_path, index=False, mode="w")
            logger.info(f"Created new CSV file: {self.combined_csv_path}")
        else:
            df_row.to_csv(self.combined_csv_path, index=False, mode="a", header=False)

    def generate_final_summary(self):
        """Generate final summary from the combined CSV file"""
        logger.info("\n" + "=" * 80)
        logger.info("Generating Final Summary from All Results")
        logger.info("=" * 80)

        # Read the combined CSV file
        if not self.combined_csv_path.exists():
            logger.warning("No combined CSV file found!")
            return

        combined_df = pd.read_csv(self.combined_csv_path)
        logger.info(f"Reading results from: {self.combined_csv_path}")

        # Generate statistics
        total = len(combined_df)
        successful = combined_df["success"].sum()
        failed = total - successful

        logger.info("\n" + "=" * 80)
        logger.info("Final Summary Statistics")
        logger.info("=" * 80)
        logger.info(f"Total images processed: {total}")
        logger.info(
            f"Successful predictions: {successful} ({successful / total * 100:.1f}%)"
        )
        logger.info(f"Failed predictions: {failed} ({failed / total * 100:.1f}%)")

        # Geodesic statistics for successful predictions
        valid_geodesic = combined_df[combined_df["geodesic"].notna()]

        if len(valid_geodesic) > 0:
            geodesic_distances = valid_geodesic["geodesic"].tolist()

            logger.info("\n" + "-" * 80)
            logger.info("Geodesic Distance Evaluation")
            logger.info("-" * 80)
            logger.info(f"Valid predictions with geodesic: {len(geodesic_distances)}")
            logger.info(
                f"Mean distance: {sum(geodesic_distances) / len(geodesic_distances):.2f} km"
            )
            logger.info(
                f"Median distance: {sorted(geodesic_distances)[len(geodesic_distances) // 2]:.2f} km"
            )

            # Distance thresholds
            thresholds = [1, 25, 200, 750, 2500]
            logger.info("\nAccuracy at distance thresholds:")
            for threshold in thresholds:
                count = sum(1 for d in geodesic_distances if d < threshold)
                accuracy = count / len(geodesic_distances) * 100
                logger.info(
                    f"  {threshold:>4d} km: {count:>3d}/{len(geodesic_distances)} ({accuracy:>5.1f}%)"
                )

            # Save summary
            summary = {
                "total_images": total,
                "successful": int(successful),
                "failed": int(failed),
                "success_rate": float(successful / total) if total > 0 else 0,
                "model_name": self.model_name,
                "timestamp": datetime.now().isoformat(),
                "geodesic_stats": {
                    "valid_predictions": len(geodesic_distances),
                    "mean_km": sum(geodesic_distances) / len(geodesic_distances),
                    "median_km": sorted(geodesic_distances)[
                        len(geodesic_distances) // 2
                    ],
                    "accuracy_1km": sum(1 for d in geodesic_distances if d < 1)
                    / len(geodesic_distances),
                    "accuracy_25km": sum(1 for d in geodesic_distances if d < 25)
                    / len(geodesic_distances),
                    "accuracy_200km": sum(1 for d in geodesic_distances if d < 200)
                    / len(geodesic_distances),
                    "accuracy_750km": sum(1 for d in geodesic_distances if d < 750)
                    / len(geodesic_distances),
                    "accuracy_2500km": sum(1 for d in geodesic_distances if d < 2500)
                    / len(geodesic_distances),
                },
            }
        else:
            summary = {
                "total_images": total,
                "successful": int(successful),
                "failed": int(failed),
                "success_rate": float(successful / total) if total > 0 else 0,
                "model_name": self.model_name,
                "timestamp": datetime.now().isoformat(),
            }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = self.output_dir / f"summary_final_{timestamp}.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"\nSaved final summary: {summary_path}")
        logger.info(f"All results saved in: {self.combined_csv_path}")

    def generate_summary(self):
        """Generate summary statistics with geodesic evaluation"""
        logger.info("\n" + "=" * 80)
        logger.info("Summary Statistics")
        logger.info("=" * 80)

        total = len(self.results)

        if total == 0:
            logger.warning("No images were processed!")
            return

        successful = sum(1 for r in self.results if r["success"])
        failed = total - successful

        logger.info(f"Total images processed: {total}")
        logger.info(
            f"Successful predictions: {successful} ({successful / total * 100:.1f}%)"
        )
        logger.info(f"Failed predictions: {failed} ({failed / total * 100:.1f}%)")

        geodesic_distances = []
        if successful > 0:
            avg_evidence = (
                sum(len(r["evidence"]) for r in self.results if r["success"])
                / successful
            )
            logger.info(f"Average evidence per prediction: {avg_evidence:.1f}")

            # Compute geodesic statistics
            geodesic_distances = []
            for result in self.results:
                if (
                    result["success"]
                    and result["latitude"] != 0.0
                    and result["longitude"] != 0.0
                ):
                    gt_row = self.ground_truth_df[
                        self.ground_truth_df["IMG_ID"] == result["image_id"]
                    ]
                    if len(gt_row) > 0:
                        lat_true = gt_row.iloc[0]["LAT"]
                        lon_true = gt_row.iloc[0]["LON"]
                        try:
                            dist = geodesic(
                                (lat_true, lon_true),
                                (result["latitude"], result["longitude"]),
                            ).km
                            geodesic_distances.append(dist)
                        except Exception:
                            pass

            if geodesic_distances:
                logger.info("\n" + "-" * 80)
                logger.info("Geodesic Distance Evaluation")
                logger.info("-" * 80)
                logger.info(
                    f"Valid predictions with geodesic: {len(geodesic_distances)}"
                )
                logger.info(
                    f"Mean distance: {sum(geodesic_distances) / len(geodesic_distances):.2f} km"
                )
                logger.info(
                    f"Median distance: {sorted(geodesic_distances)[len(geodesic_distances) // 2]:.2f} km"
                )

                # Distance thresholds (same as in IndexSearch.py)
                thresholds = [1, 25, 200, 750, 2500]
                logger.info("\nAccuracy at distance thresholds:")
                for threshold in thresholds:
                    count = sum(1 for d in geodesic_distances if d < threshold)
                    accuracy = count / len(geodesic_distances) * 100
                    logger.info(
                        f"  {threshold:>4d} km: {count:>3d}/{len(geodesic_distances)} ({accuracy:>5.1f}%)"
                    )

        # Save summary
        summary = {
            "total_images": total,
            "successful": successful,
            "failed": failed,
            "success_rate": successful / total if total > 0 else 0,
            "model_name": self.model_name,
            "timestamp": datetime.now().isoformat(),
        }

        if successful > 0 and geodesic_distances:
            summary["geodesic_stats"] = {
                "valid_predictions": len(geodesic_distances),
                "mean_km": sum(geodesic_distances) / len(geodesic_distances),
                "median_km": sorted(geodesic_distances)[len(geodesic_distances) // 2],
                "accuracy_1km": sum(1 for d in geodesic_distances if d < 1)
                / len(geodesic_distances),
                "accuracy_25km": sum(1 for d in geodesic_distances if d < 25)
                / len(geodesic_distances),
                "accuracy_200km": sum(1 for d in geodesic_distances if d < 200)
                / len(geodesic_distances),
                "accuracy_750km": sum(1 for d in geodesic_distances if d < 750)
                / len(geodesic_distances),
                "accuracy_2500km": sum(1 for d in geodesic_distances if d < 2500)
                / len(geodesic_distances),
            }

        summary_path = (
            self.output_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"\nSaved summary: {summary_path}")


async def main():
    """Main function"""
    # Load environment variables
    load_dotenv()

    # Check required API key
    if not os.getenv("GOOGLE_CLOUD_API_KEY"):
        logger.error("[ERROR] GOOGLE_CLOUD_API_KEY not found in environment!")
        logger.error("Please create a .env file with your API key")
        sys.exit(1)

    # Configuration
    config = {
        "device": "cpu",  # or 'cpu'
        "model_name": "gemini-2.0-flash",  # Đổi từ -exp sang bản chính thức
        "checkpoint_path": "checkpoints/g3.pth",
        "index_path": "index/I_g3_im2gps3k.npy",
        "database_csv_path": "checkpoints/im2gps3k_places365.csv",
        "ground_truth_csv_path": "data/im2gps3k/im2gps3k_places365.csv",
        "input_dir": "data/im2gps3k/images",  # Update this path
        "output_dir": "results/im2gps3k",
    }

    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")

    # Initialize pipeline
    pipeline = Im2GPS3KPipeline(**config)

    # Run prediction
    # Option 1: Test with specific images
    # await pipeline.run_batch_prediction(
    #     image_ids=['31700873_d7c4159106_22_25159586@N00.jpg',
    #                '32475180_a217d82b2e_21_14542551@N00.jpg', '33582773_435577ae9d_22_39303693@N00.jpg'
    #                ],
    #     image_prediction=True,
    #     text_prediction=True,
    #     max_parallel=3,  # Process 3 images in parallel
    # )

    # Option 2: Continue from last processed (auto-resume)
    # await pipeline.run_batch_prediction(
    #     max_images=1,  # Process next 500 images from where we left off
    #     image_prediction=True,
    #     text_prediction=True,
    #     max_parallel=6,  # Process 6 images in parallel
    # )

    # Option 3: Process specific range by index
    await pipeline.run_batch_prediction(
        start_index=0,  # Start from first image (0-based)
        end_index=500,  # Process up to (but not including) image 500
        image_prediction=True,
        text_prediction=True,
        max_parallel=6,  # Process 6 images in parallel
    )

    # Option 4: Process all remaining images
    # await pipeline.run_batch_prediction(
    #     image_prediction=True,
    #     text_prediction=True,
    #     max_parallel=5,  # Process 5 images in parallel
    # )

    logger.info("\n[SUCCESS] Pipeline completed successfully!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n[!] Pipeline interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n[ERROR] Pipeline failed: {e}", exc_info=True)
        sys.exit(1)
