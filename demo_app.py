"""
Streamlit Web Demo for Enhanced Document Detection
Interactive interface for testing the detection model
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import requests
import json
import time
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import base64
import os
from typing import Dict, List, Optional, Tuple

# Configure Streamlit
st.set_page_config(
    page_title="Document Detection Demo",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .detection-box {
        border: 2px solid #4CAF50;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "demo_key")

# Initialize session state
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []
if 'current_results' not in st.session_state:
    st.session_state.current_results = None


class DocumentDetectionDemo:
    """Main demo application class"""
    
    def __init__(self):
        self.api_url = API_URL
        self.api_key = API_KEY
        self.class_colors = self._generate_class_colors()
    
    def _generate_class_colors(self) -> Dict[str, str]:
        """Generate distinct colors for each class"""
        classes = [
            'text', 'title', 'list', 'table', 'figure', 'caption',
            'header', 'footer', 'page_number', 'staff', 'measure',
            'note', 'clef', 'time_signature', 'lyrics', 'checkbox',
            'input_field', 'signature_field', 'dropdown', 'flowchart',
            'graph', 'equation', 'barcode', 'qr_code', 'logo', 'stamp'
        ]
        
        colors = px.colors.qualitative.Plotly + px.colors.qualitative.Dark24
        return {cls: colors[i % len(colors)] for i, cls in enumerate(classes)}
    
    def detect_document(self, image: np.ndarray, 
                       confidence_threshold: float = 0.3,
                       nms_threshold: float = 0.5,
                       use_ensemble: bool = True,
                       selected_classes: Optional[List[str]] = None) -> Optional[Dict]:
        """Send detection request to API"""
        try:
            # Convert image to bytes
            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            files = {'file': ('image.jpg', buffer.tobytes(), 'image/jpeg')}
            
            # Prepare parameters
            params = {
                'confidence_threshold': confidence_threshold,
                'nms_threshold': nms_threshold,
                'use_ensemble': use_ensemble,
                'return_visualization': False
            }
            
            if selected_classes:
                params['classes'] = ','.join(selected_classes)
            
            # Add authentication
            headers = {'Authorization': f'Bearer {self.api_key}'}
            
            # Send request
            response = requests.post(
                f"{self.api_url}/api/v1/detect",
                files=files,
                params=params,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            st.error("Request timed out. Please try again.")
            return None
        except Exception as e:
            st.error(f"Error: {str(e)}")
            return None
    
    def visualize_detections(self, image: np.ndarray, 
                           detections: List[Dict],
                           show_confidence: bool = True) -> np.ndarray:
        """Visualize detection results on image"""
        vis_image = image.copy()
        height, width = image.shape[:2]
        
        # Draw bounding boxes
        for det in detections:
            x1 = int(det['x1'])
            y1 = int(det['y1'])
            x2 = int(det['x2'])
            y2 = int(det['y2'])
            
            # Get color for class
            class_name = det['class_name']
            color = self.class_colors.get(class_name, '#FF0000')
            
            # Convert hex to RGB
            color_rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
            
            # Draw rectangle
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color_rgb, 2)
            
            # Prepare label
            label = class_name
            if show_confidence:
                label += f" {det['confidence']:.2f}"
            
            # Draw label background
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_y = y1 - 10 if y1 - 10 > 10 else y1 + label_size[1] + 10
            
            cv2.rectangle(vis_image,
                         (x1, label_y - label_size[1] - 5),
                         (x1 + label_size[0] + 5, label_y + 5),
                         color_rgb, -1)
            
            # Draw label text
            cv2.putText(vis_image, label,
                       (x1 + 2, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                       (255, 255, 255), 1)
        
        return vis_image
    
    def generate_statistics(self, detections: List[Dict]) -> Dict:
        """Generate statistics from detection results"""
        stats = {
            'total_detections': len(detections),
            'class_distribution': {},
            'confidence_stats': {
                'mean': 0,
                'min': 1,
                'max': 0,
                'std': 0
            },
            'area_stats': {
                'mean': 0,
                'min': float('inf'),
                'max': 0
            }
        }
        
        if not detections:
            return stats
        
        confidences = []
        areas = []
        
        for det in detections:
            # Class distribution
            class_name = det['class_name']
            stats['class_distribution'][class_name] = stats['class_distribution'].get(class_name, 0) + 1
            
            # Confidence
            confidences.append(det['confidence'])
            
            # Area
            area = det['area']
            areas.append(area)
        
        # Calculate statistics
        stats['confidence_stats'] = {
            'mean': np.mean(confidences),
            'min': np.min(confidences),
            'max': np.max(confidences),
            'std': np.std(confidences)
        }
        
        stats['area_stats'] = {
            'mean': np.mean(areas),
            'min': np.min(areas),
            'max': np.max(areas)
        }
        
        return stats


def main():
    # Initialize demo
    demo = DocumentDetectionDemo()
    
    # Header
    st.title("🔍 Enhanced Document Detection Demo")
    st.markdown("State-of-the-art document content detection with support for academic papers, music scores, forms, and more!")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Detection Settings")
        
        # Detection parameters
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
            help="Minimum confidence score for detections"
        )
        
        nms_threshold = st.slider(
            "NMS Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Non-maximum suppression threshold"
        )
        
        use_ensemble = st.checkbox(
            "Use Model Ensemble",
            value=True,
            help="Use multiple models for better accuracy"
        )
        
        # Class filter
        st.subheader("Class Filter")
        all_classes = list(demo.class_colors.keys())
        
        class_categories = {
            'Document': ['text', 'title', 'list', 'table', 'figure', 'caption', 
                        'header', 'footer', 'page_number'],
            'Music': ['staff', 'measure', 'note', 'clef', 'time_signature', 'lyrics'],
            'Form': ['checkbox', 'input_field', 'signature_field', 'dropdown'],
            'Diagram': ['flowchart', 'graph', 'equation'],
            'Special': ['barcode', 'qr_code', 'logo', 'stamp']
        }
        
        selected_categories = st.multiselect(
            "Select Categories",
            options=list(class_categories.keys()),
            default=list(class_categories.keys())
        )
        
        selected_classes = []
        for cat in selected_categories:
            selected_classes.extend(class_categories[cat])
        
        # Visualization options
        st.subheader("Visualization")
        show_confidence = st.checkbox("Show Confidence Scores", value=True)
        show_statistics = st.checkbox("Show Statistics", value=True)
        
        # API Status
        st.subheader("API Status")
        if st.button("Check API Health"):
            try:
                response = requests.get(f"{API_URL}/health", timeout=5)
                if response.status_code == 200:
                    health = response.json()
                    st.success(f"✅ API is {health['status']}")
                    st.json(health)
                else:
                    st.error("❌ API is not responding")
            except:
                st.error("❌ Cannot connect to API")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Document")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'tif', 'tiff', 'bmp'],
            help="Upload a document image for detection"
        )
        
        # Sample images
        st.subheader("Or try a sample image:")
        sample_images = {
            "Academic Paper": "samples/academic_paper.jpg",
            "Music Score": "samples/music_score.jpg",
            "Form Document": "samples/form.jpg",
            "Mixed Document": "samples/mixed.jpg"
        }
        
        sample_cols = st.columns(len(sample_images))
        for idx, (name, path) in enumerate(sample_images.items()):
            with sample_cols[idx]:
                if st.button(name, key=f"sample_{idx}"):
                    # Load sample image (placeholder)
                    st.info(f"Loading {name} sample...")
        
        if uploaded_file is not None:
            # Load and display image
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            
            st.image(image, caption="Uploaded Document", use_column_width=True)
            
            # Image info
            st.info(f"Image Size: {image_np.shape[1]} x {image_np.shape[0]} pixels")
            
            # Detect button
            if st.button("🚀 Run Detection", type="primary"):
                with st.spinner("Processing document..."):
                    start_time = time.time()
                    
                    # Run detection
                    results = demo.detect_document(
                        image_np,
                        confidence_threshold=confidence_threshold,
                        nms_threshold=nms_threshold,
                        use_ensemble=use_ensemble,
                        selected_classes=selected_classes if selected_classes else None
                    )
                    
                    processing_time = time.time() - start_time
                    
                    if results and results['success']:
                        st.session_state.current_results = results
                        st.session_state.detection_history.append({
                            'timestamp': datetime.now(),
                            'filename': uploaded_file.name,
                            'results': results,
                            'processing_time': processing_time
                        })
                        st.success(f"✅ Detection completed in {processing_time:.2f} seconds")
                    else:
                        st.error("❌ Detection failed")
    
    with col2:
        st.header("📊 Detection Results")
        
        if st.session_state.current_results:
            results = st.session_state.current_results
            detections = results['detections']
            
            # Summary metrics
            col1_metrics, col2_metrics, col3_metrics = st.columns(3)
            
            with col1_metrics:
                st.metric("Total Detections", results['num_detections'])
            
            with col2_metrics:
                st.metric("Processing Time", f"{results['processing_time']:.2f}s")
            
            with col3_metrics:
                st.metric("Unique Classes", len(set(d['class_name'] for d in detections)))
            
            # Visualized image
            if uploaded_file and detections:
                image = Image.open(uploaded_file)
                image_np = np.array(image)
                
                vis_image = demo.visualize_detections(
                    image_np,
                    detections,
                    show_confidence=show_confidence
                )
                
                st.image(vis_image, caption="Detection Results", use_column_width=True)
            
            # Detection list
            if detections:
                st.subheader("Detected Elements")
                
                # Create DataFrame
                df = pd.DataFrame(detections)
                df = df[['class_name', 'confidence', 'x1', 'y1', 'x2', 'y2', 'area']]
                df.columns = ['Class', 'Confidence', 'X1', 'Y1', 'X2', 'Y2', 'Area']
                
                # Display with formatting
                st.dataframe(
                    df.style.format({
                        'Confidence': '{:.3f}',
                        'X1': '{:.0f}',
                        'Y1': '{:.0f}',
                        'X2': '{:.0f}',
                        'Y2': '{:.0f}',
                        'Area': '{:.0f}'
                    }),
                    use_container_width=True
                )
                
                # Download results
                results_json = json.dumps(results, indent=2)
                st.download_button(
                    label="📥 Download Results (JSON)",
                    data=results_json,
                    file_name=f"detection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
    
    # Statistics section
    if show_statistics and st.session_state.current_results:
        st.header("📈 Detection Statistics")
        
        detections = st.session_state.current_results['detections']
        stats = demo.generate_statistics(detections)
        
        col1_stats, col2_stats = st.columns(2)
        
        with col1_stats:
            # Class distribution pie chart
            if stats['class_distribution']:
                fig_pie = go.Figure(data=[go.Pie(
                    labels=list(stats['class_distribution'].keys()),
                    values=list(stats['class_distribution'].values()),
                    hole=0.3
                )])
                fig_pie.update_layout(
                    title="Class Distribution",
                    height=400
                )
                st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2_stats:
            # Confidence distribution histogram
            if detections:
                confidences = [d['confidence'] for d in detections]
                fig_hist = go.Figure(data=[go.Histogram(
                    x=confidences,
                    nbinsx=20,
                    name='Confidence'
                )])
                fig_hist.update_layout(
                    title="Confidence Distribution",
                    xaxis_title="Confidence Score",
                    yaxis_title="Count",
                    height=400
                )
                st.plotly_chart(fig_hist, use_container_width=True)
        
        # Additional statistics
        with st.expander("Detailed Statistics"):
            col1_detail, col2_detail = st.columns(2)
            
            with col1_detail:
                st.subheader("Confidence Statistics")
                st.json(stats['confidence_stats'])
            
            with col2_detail:
                st.subheader("Area Statistics")
                st.json(stats['area_stats'])
    
    # History section
    if st.session_state.detection_history:
        st.header("📜 Detection History")
        
        history_df = pd.DataFrame([
            {
                'Timestamp': h['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                'Filename': h['filename'],
                'Detections': h['results']['num_detections'],
                'Processing Time': f"{h['processing_time']:.2f}s"
            }
            for h in st.session_state.detection_history[-10:]  # Last 10
        ])
        
        st.dataframe(history_df, use_container_width=True)
        
        if st.button("Clear History"):
            st.session_state.detection_history = []
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>Enhanced Document Detection System v1.0.0</p>
            <p>Built with ❤️ using PyTorch, Transformers, and Streamlit</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()