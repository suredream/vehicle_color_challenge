# light_classifier.py
#
# This script implements a research-oriented pipeline for vehicle analysis using the MMRotate
# library. It is designed to first detect vehicles with oriented bounding boxes using a
# pre-trained model (like ReDet) and then extract internal deep features from the model's
# Region of Interest (RoI) head.
#
# The core idea is that these deep features are a richer, more robust representation of
# the vehicles than raw pixels, making them ideal for training lightweight downstream
# classifiers for tasks like color and type identification.
#
# NOTE: This script requires a full, correctly installed MMRotate environment and the
# corresponding model configuration and checkpoint files to run.

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import warnings
import os
import traceback

# --- Critical Fix: Import APIs and model builders from mmrotate ---
# This ensures that rotated detection models like ReDet are correctly registered.
try:
    from mmrotate.apis import init_detector, inference_detector
    from mmrotate.models.builder import build_detector
    print("✓ Successfully imported modules from mmrotate.")
except ImportError:
    print("❌ Error: Please ensure you have installed MMRotate correctly (`pip install mmrotate`).")
    # The script cannot proceed if mmrotate is not installed.
    exit()

from mmcv import Config
from mmcv.runner import load_checkpoint

warnings.filterwarnings('ignore')

class LightClassifier:
    def __init__(self, config_path, checkpoint_path, device='cuda:0', force_cpu_init=False):
        """
        Initializes a device-safe feature extractor for vehicles.

        Args:
            config_path (str): Path to the model config file (e.g., from MMRotate).
            checkpoint_path (str): Path to the model checkpoint file (.pth).
            device (str): The target device to run the model on (e.g., 'cuda:0' or 'cpu').
            force_cpu_init (bool): If True, the model is first initialized on the CPU and then
                                   moved to the GPU. This is a safer and more stable method
                                   that avoids potential device mismatch errors during initialization.
        """
        self.target_device = device
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path

        # Check for CUDA availability and fall back to CPU if necessary.
        if device.startswith('cuda') and not torch.cuda.is_available():
            print(f"⚠️  Warning: CUDA is not available. Falling back to device 'cpu'.")
            self.target_device = 'cpu'
            force_cpu_init = True

        print(f"Target device set to: {self.target_device}")

        # Clear GPU cache before loading the model to free up memory.
        if self.target_device.startswith('cuda'):
            torch.cuda.empty_cache()
            print("✓ Cleared GPU cache.")

        # Initialize the model using the selected strategy.
        if force_cpu_init or self.target_device == 'cpu':
            self.model = self._init_model_safe()
        else:
            try:
                print("Attempting direct initialization on GPU...")
                # Use MMRotate's high-level API for direct initialization.
                self.model = init_detector(config_path, checkpoint_path, device=self.target_device)
                print("✓ Direct GPU initialization successful.")
            except RuntimeError as e:
                # If direct GPU init fails, fall back to the safer CPU-first method.
                if "Expected all tensors to be on the same device" in str(e):
                    print("❌ Direct GPU initialization failed. Retrying with safe CPU-to-GPU migration...")
                    self.model = self._init_model_safe()
                else:
                    # Re-raise other unexpected runtime errors.
                    raise e

        self.roi_features = {}
        # NOTE: These class IDs are specific to the DOTA dataset. You must verify them against
        # the `class_names` list in your model's config file.
        # Example for DOTA: 'small-vehicle': 9, 'large-vehicle': 10
        self.vehicle_class_ids = [9, 10]
        self.class_names = {
            9: 'small-vehicle',
            10: 'large-vehicle'
        }

        print("✓ Model initialization complete.")
        self._print_model_device_info()

    def _init_model_safe(self):
        """
        Initializes the model safely on the CPU and then moves it to the target device.
        This is the most robust method for preventing device-related errors.
        """
        print("Using safe initialization strategy: CPU -> Target Device")

        # Step 1: Build the model architecture on the CPU.
        print("Step 1: Building model structure on CPU...")
        cfg = Config.fromfile(self.config_path)
        # Use the builder imported from mmrotate to construct the model.
        model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))

        # Step 2: Load the model weights (checkpoint) onto the CPU.
        print("Step 2: Loading checkpoint weights on CPU...")
        load_checkpoint(model, self.checkpoint_path, map_location='cpu')

        # Step 3: Set the model to evaluation mode.
        model.eval()

        # Step 4: If the target is a CUDA device, move the entire model to it.
        if self.target_device.startswith('cuda'):
            print(f"Step 3: Migrating model to {self.target_device}...")
            model.to(self.target_device)
            print("✓ Model successfully migrated to GPU.")
        else:
            print("✓ Model remains on CPU.")

        return model

    def _print_model_device_info(self):
        """
        A utility function to print the device placement of the model's parameters.
        Useful for debugging multi-GPU or CPU/GPU issues.
        """
        print("\n📋 Model Device Info:")
        try:
            devices = {p.device for p in self.model.parameters()}
            print(f"  Model parameters are on devices: {devices}")
            if len(devices) > 1:
                print("  ⚠️ Warning: Model parameters are spread across multiple devices!")
        except Exception as e:
            print(f"  Could not retrieve device info: {e}")


    def _safe_hook_fn(self, name):
        """
        Creates a device-safe forward hook function. A hook intercepts the output
        of a model layer during the forward pass without altering the model's code.
        """
        def hook(module, input_tensor, output_tensor):
            """
            The actual hook function. It detaches the output tensor from the computation
            graph and moves it to the CPU to prevent GPU memory leaks and allow for
            processing with libraries like NumPy.
            """
            try:
                # Handle both single tensor and tuple/list outputs.
                if isinstance(output_tensor, torch.Tensor):
                    self.roi_features[name] = output_tensor.detach().cpu()
                elif isinstance(output_tensor, (list, tuple)):
                    self.roi_features[name] = [o.detach().cpu() if isinstance(o, torch.Tensor) else o for o in output_tensor]

                # Optionally, capture the input to the module as well (useful for debugging).
                if len(input_tensor) > 0 and isinstance(input_tensor[0], torch.Tensor):
                    self.roi_features[f'{name}_input_rois'] = input_tensor[0].detach().cpu()

            except Exception as e:
                print(f"⚠️  Warning during execution of hook '{name}': {e}")

        return hook

    def extract_vehicle_features(self, image_path, confidence_thresh=0.3):
        """
        Extracts vehicle detections and their corresponding deep features from an image.
        This is the main orchestration method for the feature extraction process.
        """
        print(f"\n🔍 Analyzing image: {image_path}")
        print("-" * 50)

        self.roi_features.clear()
        hooks = []

        try:
            # Register a forward hook on the RoI extractor. The exact path to this module
            # is model-dependent. For ReDet, it's often in this location.
            # This is the critical step for capturing intermediate features.
            if hasattr(self.model, 'roi_head') and hasattr(self.model.roi_head, 'bbox_roi_extractor'):
                extractor = self.model.roi_head.bbox_roi_extractor
                hook = extractor.register_forward_hook(self._safe_hook_fn('roi_extractor'))
                hooks.append(hook)
                print(f"✓ Registered hook on '{extractor.__class__.__name__}'.")

            # Run inference using the MMRotate API.
            print("🔄 Performing inference...")
            result = inference_detector(self.model, image_path)
            print("✅ Inference complete.")

            # Post-process the raw detection results.
            vehicle_objects = self._process_detection_results(result, confidence_thresh)
            
            # Correlate the captured features from the hook with the detected objects.
            self._correlate_features_with_objects(vehicle_objects)

            return vehicle_objects, result

        except Exception as e:
            print(f"❌ Feature extraction failed: {e}")
            traceback.print_exc()
            return [], None

        finally:
            # CRITICAL: Always remove hooks after use to prevent memory leaks and side effects.
            for hook in hooks:
                hook.remove()
            print(f"🧹 Cleaned up {len(hooks)} hook(s).")

    def _process_detection_results(self, result, confidence_thresh):
        """

        Filters raw detection results from the model to get a clean list of vehicle objects.
        """
        vehicle_objects = []
        # The 'result' is a list where each index corresponds to a class ID.
        total_detections = sum(len(result[i]) for i in self.vehicle_class_ids if i < len(result))
        print(f"📊 Detection stats: Found {total_detections} raw vehicle candidates.")

        for class_id in self.vehicle_class_ids:
            if class_id < len(result):
                detections_for_class = result[class_id]
                for det in detections_for_class:
                    # OBB format: [x_center, y_center, width, height, angle_rad, score]
                    if len(det) >= 6 and det[5] > confidence_thresh:
                        vehicle_obj = {
                            'id': len(vehicle_objects),
                            'rbbox': det[:5],  # [xc, yc, w, h, angle]
                            'confidence': det[5],
                            'class_id': class_id,
                            'class_name': self.class_names.get(class_id, 'unknown')
                        }
                        vehicle_objects.append(vehicle_obj)
        
        print(f"🎯 Found {len(vehicle_objects)} vehicles above confidence threshold {confidence_thresh}.")
        return vehicle_objects

    def _correlate_features_with_objects(self, vehicle_objects):
        """
        Matches the extracted RoI features to their corresponding vehicle detections.
        """
        print(f"\n🔧 Correlating features:")
        roi_feature_key = 'roi_extractor'
        if roi_feature_key in self.roi_features:
            features = self.roi_features[roi_feature_key]
            if isinstance(features, torch.Tensor):
                print(f"  - Captured feature tensor with shape: {features.shape}")
                # Assign each feature vector to its corresponding vehicle object.
                for i, obj in enumerate(vehicle_objects):
                    if i < features.shape[0]:
                        # The feature for each RoI is typically [Channels, Height, Width]
                        obj[f'roi_feature'] = features[i]

    def _generate_semantic_vectors(self, vehicle_objects):
        """
        Converts the spatial RoI features (e.g., 256x7x7) into a single semantic
        feature vector (e.g., 256) using Adaptive Average Pooling. This creates a
        fixed-size "fingerprint" for each vehicle.
        """
        for obj in vehicle_objects:
            if 'roi_feature' in obj:
                roi_feat = obj['roi_feature']  # Shape: [C, H, W]
                # Add a batch dimension, apply pooling, and remove extra dimensions.
                semantic_vector = F.adaptive_avg_pool2d(roi_feat.unsqueeze(0), (1, 1)).squeeze()
                obj['semantic_vector'] = semantic_vector.numpy()
        print("✓ Generated semantic vectors for all detected vehicles.")

    def _analyze_color_with_rotated_mask(self, image, vehicle_objects):
        """
        A baseline color analysis method that uses the rotated bounding box to create
        a precise mask, reducing background noise in the color calculation.
        """
        for obj in vehicle_objects:
            # Convert the oriented bounding box to four corner points.
            # Note: angle from model is in radians, cv2.boxPoints expects degrees.
            rbbox = obj['rbbox']
            angle_deg = rbbox[4] * 180 / np.pi
            box_points = cv2.boxPoints(((rbbox[0], rbbox[1]), (rbbox[2], rbbox[3]), angle_deg)).astype(np.int0)

            # Create a mask for the vehicle polygon.
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [box_points], 255)
            
            # Calculate the mean color only within the masked region.
            # This is more accurate than averaging a simple rectangular crop.
            mean_bgr = cv2.mean(image, mask=mask)[:3]
            mean_bgr_uint = np.uint8([[mean_bgr]])
            mean_hsv = cv2.cvtColor(mean_bgr_uint, cv2.COLOR_BGR2HSV)[0][0]
            
            obj['color'] = self._classify_hsv_to_color_name(mean_hsv)

    def _classify_hsv_to_color_name(self, hsv):
        """A simple HSV-based color classifier. Prone to errors from shadow/light."""
        h, s, v = hsv
        if v < 60: return 'Black'
        if v > 210 and s < 40: return 'White'
        if s < 50: return 'Gray'
        if h < 10 or h > 165: return 'Red'
        if h < 25: return 'Orange'
        if h < 35: return 'Yellow'
        if h < 85: return 'Green'
        if h < 130: return 'Blue'
        if h < 150: return 'Purple' # A common color for cars
        return 'Brown' # Default for other hues like deep orange/reds

    def run_full_analysis(self, image_path, confidence_thresh=0.3):
        """
        Executes the entire analysis pipeline on a single image.
        """
        print("=" * 60)
        print("🚀 Starting Full Vehicle Analysis Pipeline")
        print("=" * 60)
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Error: Could not read image at {image_path}")
            return None

        # 1. Detect vehicles and extract their deep RoI features.
        vehicle_objects, _ = self.extract_vehicle_features(image_path, confidence_thresh)
        if not vehicle_objects:
            print("✅ Analysis complete: No vehicles detected.")
            return []

        # 2. Convert spatial features into semantic vectors.
        self._generate_semantic_vectors(vehicle_objects)

        # 3. Perform baseline color analysis using pixel data.
        self._analyze_color_with_rotated_mask(image, vehicle_objects)
        
        # 4. Compile and print the final report.
        print("\n📋 Final Analysis Report:")
        print("-" * 50)
        for i, obj in enumerate(vehicle_objects):
            print(f"🚗 Vehicle #{obj['id']}:")
            print(f"  - Type (from detector): {obj['class_name']}")
            print(f"  - Color (from pixels): {obj.get('color', 'N/A')}")
            print(f"  - Confidence: {obj['confidence']:.3f}")
            if 'semantic_vector' in obj:
                print(f"  - Semantic Vector Dim: {obj['semantic_vector'].shape}")
            print("-" * 20)
        
        return vehicle_objects


def main():
    """
    Main function to demonstrate the LightClassifier pipeline.
    """
    # --- USER CONFIGURATION ---
    # You must download the ReDet model for DOTA and its config file.
    # Model: https://download.openmmlab.com/mmrotate/v0.1.0/redet/redet_re50_fpn_1x_dota_ms_rr_le90/redet_re50_fpn_1x_dota_ms_rr_le90-fc9217b5.pth
    # Config: Find it in the MMRotate GitHub repo under configs/redet/
    config_file = './redet_re50_fpn_1x_dota_ms_rr_le90.py'
    checkpoint_file = './redet_re50_fpn_1x_dota_ms_rr_le90-fc9217b5.pth'
    
    # Use a demo image.
    # You may need to clone the mmrotate repo to get this image or use your own.
    image_file = './demo.jpg'

    # --- Pre-run Checks ---
    for f in [config_file, checkpoint_file, image_file]:
        if not os.path.exists(f):
            print(f"❌ Fatal Error: Required file not found at '{f}'.")
            print("Please download the necessary files and check the paths.")
            return

    try:
        # Initialize the classifier. Using force_cpu_init=True is recommended for stability.
        classifier = LightClassifier(
            config_file,
            checkpoint_file,
            device='cuda:0',
            force_cpu_init=True
        )
        # Run the full analysis.
        results = classifier.run_full_analysis(image_file, confidence_thresh=0.4)
        if results:
            print(f"\n✅ Pipeline finished successfully. Analyzed {len(results)} vehicles.")

    except Exception as e:
        print(f"\n❌ A critical error occurred during the main execution: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()