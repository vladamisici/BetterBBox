"""
Synthetic Data Generator for Document Detection
Generates realistic synthetic documents with accurate annotations
"""

import os
import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from faker import Faker
import imgaug.augmenters as iaa
import music21
from reportlab.lib.pagesizes import letter, A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import pandas as pd
from datetime import datetime
import xml.etree.ElementTree as ET


class SyntheticDocumentGenerator:
    """Base class for synthetic document generation"""
    
    def __init__(self, output_dir: str, num_samples: int = 1000):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_samples = num_samples
        self.faker = Faker()
        
        # Create subdirectories
        self.images_dir = self.output_dir / 'images'
        self.annotations_dir = self.output_dir / 'annotations'
        self.images_dir.mkdir(exist_ok=True)
        self.annotations_dir.mkdir(exist_ok=True)
        
        # Load fonts
        self.fonts = self._load_fonts()
        
        # Background textures
        self.backgrounds = self._load_backgrounds()
        
        # Augmentation pipeline
        self.augmenter = self._create_augmentation_pipeline()
    
    def _load_fonts(self) -> List[ImageFont.FreeTypeFont]:
        """Load various fonts for text rendering"""
        font_paths = [
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        ]
        
        fonts = []
        for size in [12, 14, 16, 18, 20, 24, 28, 32]:
            for font_path in font_paths:
                try:
                    font = ImageFont.truetype(font_path, size)
                    fonts.append(font)
                except:
                    # Use default font if specific font not found
                    font = ImageFont.load_default()
                    fonts.append(font)
        
        return fonts if fonts else [ImageFont.load_default()]
    
    def _load_backgrounds(self) -> List[np.ndarray]:
        """Generate or load background textures"""
        backgrounds = []
        
        # Generate paper textures
        for _ in range(10):
            # White paper with slight noise
            bg = np.ones((1200, 900, 3), dtype=np.uint8) * 255
            noise = np.random.normal(0, 5, bg.shape).astype(np.uint8)
            bg = cv2.add(bg, noise)
            backgrounds.append(bg)
            
            # Aged paper
            aged = np.ones((1200, 900, 3), dtype=np.uint8) * 245
            aged[:, :, 0] = np.clip(aged[:, :, 0] + np.random.normal(0, 10, aged[:, :, 0].shape), 200, 255)
            aged[:, :, 1] = np.clip(aged[:, :, 1] + np.random.normal(0, 8, aged[:, :, 1].shape), 200, 255)
            aged[:, :, 2] = np.clip(aged[:, :, 2] + np.random.normal(0, 5, aged[:, :, 2].shape), 180, 245)
            backgrounds.append(aged)
        
        return backgrounds
    
    def _create_augmentation_pipeline(self):
        """Create augmentation pipeline for synthetic documents"""
        return iaa.Sequential([
            iaa.Sometimes(0.3, iaa.Affine(
                scale=(0.95, 1.05),
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                rotate=(-2, 2),
                shear=(-2, 2)
            )),
            iaa.Sometimes(0.2, iaa.PerspectiveTransform(scale=(0.01, 0.03))),
            iaa.Sometimes(0.3, iaa.AdditiveGaussianNoise(scale=(0, 10))),
            iaa.Sometimes(0.2, iaa.GaussianBlur(sigma=(0, 1.0))),
            iaa.Sometimes(0.3, iaa.LinearContrast((0.8, 1.2))),
            iaa.Sometimes(0.2, iaa.Multiply((0.9, 1.1))),
            iaa.Sometimes(0.1, iaa.JpegCompression(compression=(70, 95)))
        ])
    
    def generate_dataset(self):
        """Generate complete synthetic dataset"""
        raise NotImplementedError("Subclasses must implement generate_dataset")
    
    def save_annotations(self, annotations: List[Dict], split: str = 'train'):
        """Save annotations in unified format"""
        output_file = self.annotations_dir / f'{split}_annotations.json'
        with open(output_file, 'w') as f:
            json.dump(annotations, f, indent=2)
        print(f"Saved {len(annotations)} annotations to {output_file}")


class AcademicDocumentGenerator(SyntheticDocumentGenerator):
    """Generate synthetic academic documents"""
    
    def __init__(self, output_dir: str, num_samples: int = 1000):
        super().__init__(output_dir, num_samples)
        
        # Academic-specific elements
        self.title_templates = [
            "A Study on {} in the Context of {}",
            "Investigating the Effects of {} on {}",
            "{}: A Comprehensive Analysis of {}",
            "Novel Approaches to {} Using {}",
            "Empirical Evaluation of {} for {}"
        ]
        
        self.section_headers = [
            "Abstract", "Introduction", "Related Work", "Methodology",
            "Experiments", "Results", "Discussion", "Conclusion",
            "References", "Appendix"
        ]
    
    def generate_dataset(self):
        """Generate synthetic academic documents"""
        annotations = []
        
        for i in tqdm(range(self.num_samples), desc="Generating academic documents"):
            doc_image, doc_annotations = self._generate_document(i)
            
            # Save image
            img_path = self.images_dir / f'academic_{i:05d}.jpg'
            cv2.imwrite(str(img_path), doc_image)
            
            # Create annotation entry
            annotations.append({
                'image_id': f'academic_{i:05d}',
                'image_path': str(img_path),
                'width': doc_image.shape[1],
                'height': doc_image.shape[0],
                'annotations': doc_annotations
            })
        
        # Save annotations
        self.save_annotations(annotations, 'train')
    
    def _generate_document(self, doc_id: int) -> Tuple[np.ndarray, List[Dict]]:
        """Generate a single academic document"""
        # Select background
        bg = random.choice(self.backgrounds).copy()
        height, width = bg.shape[:2]
        
        # Create PIL image for drawing
        img = Image.fromarray(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        
        annotations = []
        y_offset = 50
        
        # Generate title
        title_text = random.choice(self.title_templates).format(
            self.faker.catch_phrase(),
            self.faker.catch_phrase()
        )
        title_font = random.choice([f for f in self.fonts if f.size >= 20])
        
        # Calculate title position
        title_bbox = draw.textbbox((0, 0), title_text, font=title_font)
        title_width = title_bbox[2] - title_bbox[0]
        title_height = title_bbox[3] - title_bbox[1]
        
        title_x = (width - title_width) // 2
        draw.text((title_x, y_offset), title_text, fill=(0, 0, 0), font=title_font)
        
        annotations.append({
            'bbox': [title_x, y_offset, title_x + title_width, y_offset + title_height],
            'category': 'title',
            'confidence': 1.0
        })
        
        y_offset += title_height + 30
        
        # Generate authors
        num_authors = random.randint(1, 4)
        authors = [self.faker.name() for _ in range(num_authors)]
        author_text = ", ".join(authors)
        author_font = random.choice([f for f in self.fonts if f.size <= 16])
        
        author_bbox = draw.textbbox((0, 0), author_text, font=author_font)
        author_width = author_bbox[2] - author_bbox[0]
        author_height = author_bbox[3] - author_bbox[1]
        
        author_x = (width - author_width) // 2
        draw.text((author_x, y_offset), author_text, fill=(0, 0, 0), font=author_font)
        
        annotations.append({
            'bbox': [author_x, y_offset, author_x + author_width, y_offset + author_height],
            'category': 'text',
            'confidence': 1.0
        })
        
        y_offset += author_height + 40
        
        # Generate abstract
        if random.random() < 0.8:
            abstract_title = "Abstract"
            abstract_font = random.choice([f for f in self.fonts if f.size >= 16])
            draw.text((50, y_offset), abstract_title, fill=(0, 0, 0), font=abstract_font)
            
            abstract_bbox = draw.textbbox((0, 0), abstract_title, font=abstract_font)
            annotations.append({
                'bbox': [50, y_offset, 50 + abstract_bbox[2], y_offset + abstract_bbox[3]],
                'category': 'title',
                'confidence': 1.0
            })
            
            y_offset += abstract_bbox[3] + 10
            
            # Abstract text
            abstract_text = self.faker.paragraph(nb_sentences=random.randint(4, 8))
            text_font = random.choice([f for f in self.fonts if f.size <= 14])
            
            # Wrap text
            lines = self._wrap_text(abstract_text, text_font, width - 100)
            for line in lines:
                draw.text((50, y_offset), line, fill=(0, 0, 0), font=text_font)
                line_bbox = draw.textbbox((0, 0), line, font=text_font)
                line_height = line_bbox[3] - line_bbox[1]
                
                annotations.append({
                    'bbox': [50, y_offset, width - 50, y_offset + line_height],
                    'category': 'text',
                    'confidence': 1.0
                })
                
                y_offset += line_height + 5
            
            y_offset += 20
        
        # Generate sections
        num_sections = random.randint(3, 6)
        selected_sections = random.sample(self.section_headers[1:], num_sections)
        
        for section in selected_sections:
            if y_offset > height - 200:
                break
            
            # Section header
            section_font = random.choice([f for f in self.fonts if f.size >= 16])
            draw.text((50, y_offset), section, fill=(0, 0, 0), font=section_font)
            
            section_bbox = draw.textbbox((0, 0), section, font=section_font)
            annotations.append({
                'bbox': [50, y_offset, 50 + section_bbox[2], y_offset + section_bbox[3]],
                'category': 'title',
                'confidence': 1.0
            })
            
            y_offset += section_bbox[3] + 10
            
            # Section content
            num_paragraphs = random.randint(1, 3)
            for _ in range(num_paragraphs):
                if y_offset > height - 150:
                    break
                
                paragraph = self.faker.paragraph(nb_sentences=random.randint(3, 6))
                text_font = random.choice([f for f in self.fonts if f.size <= 14])
                
                lines = self._wrap_text(paragraph, text_font, width - 100)
                for line in lines:
                    if y_offset > height - 100:
                        break
                    
                    draw.text((50, y_offset), line, fill=(0, 0, 0), font=text_font)
                    line_bbox = draw.textbbox((0, 0), line, font=text_font)
                    line_height = line_bbox[3] - line_bbox[1]
                    
                    annotations.append({
                        'bbox': [50, y_offset, width - 50, y_offset + line_height],
                        'category': 'text',
                        'confidence': 1.0
                    })
                    
                    y_offset += line_height + 5
                
                y_offset += 10
            
            # Occasionally add a figure or table
            if random.random() < 0.3 and y_offset < height - 250:
                if random.random() < 0.5:
                    # Add figure
                    fig_width = random.randint(200, 400)
                    fig_height = random.randint(150, 300)
                    fig_x = (width - fig_width) // 2
                    
                    draw.rectangle(
                        [fig_x, y_offset, fig_x + fig_width, y_offset + fig_height],
                        outline=(0, 0, 0),
                        width=2
                    )
                    
                    # Simple plot inside
                    self._draw_simple_plot(draw, fig_x, y_offset, fig_width, fig_height)
                    
                    annotations.append({
                        'bbox': [fig_x, y_offset, fig_x + fig_width, y_offset + fig_height],
                        'category': 'figure',
                        'confidence': 1.0
                    })
                    
                    y_offset += fig_height + 10
                    
                    # Figure caption
                    caption = f"Figure {random.randint(1, 10)}: {self.faker.sentence()}"
                    caption_font = random.choice([f for f in self.fonts if f.size <= 12])
                    
                    caption_lines = self._wrap_text(caption, caption_font, fig_width)
                    for line in caption_lines:
                        draw.text((fig_x, y_offset), line, fill=(0, 0, 0), font=caption_font)
                        line_bbox = draw.textbbox((0, 0), line, font=caption_font)
                        line_height = line_bbox[3] - line_bbox[1]
                        
                        annotations.append({
                            'bbox': [fig_x, y_offset, fig_x + len(line) * 8, y_offset + line_height],
                            'category': 'caption',
                            'confidence': 1.0
                        })
                        
                        y_offset += line_height + 3
                    
                    y_offset += 20
                
                else:
                    # Add table
                    table_width = random.randint(300, 500)
                    num_rows = random.randint(3, 8)
                    num_cols = random.randint(2, 5)
                    cell_height = 30
                    table_height = num_rows * cell_height
                    table_x = (width - table_width) // 2
                    
                    # Draw table
                    for i in range(num_rows + 1):
                        y = y_offset + i * cell_height
                        draw.line([(table_x, y), (table_x + table_width, y)], 
                                fill=(0, 0, 0), width=1)
                    
                    for i in range(num_cols + 1):
                        x = table_x + i * (table_width // num_cols)
                        draw.line([(x, y_offset), (x, y_offset + table_height)], 
                                fill=(0, 0, 0), width=1)
                    
                    annotations.append({
                        'bbox': [table_x, y_offset, table_x + table_width, y_offset + table_height],
                        'category': 'table',
                        'confidence': 1.0
                    })
                    
                    y_offset += table_height + 20
        
        # Convert back to numpy array
        img_array = np.array(img)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Apply augmentations
        if random.random() < 0.7:
            img_array = self.augmenter(image=img_array)
        
        return img_array, annotations
    
    def _wrap_text(self, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
        """Wrap text to fit within max_width"""
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            bbox = font.getbbox(test_line)
            if bbox[2] - bbox[0] <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    lines.append(word)
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines
    
    def _draw_simple_plot(self, draw: ImageDraw.Draw, x: int, y: int, 
                         width: int, height: int):
        """Draw a simple plot inside a figure"""
        # Draw axes
        margin = 20
        draw.line([(x + margin, y + height - margin), 
                  (x + width - margin, y + height - margin)], 
                 fill=(0, 0, 0), width=1)
        draw.line([(x + margin, y + margin), 
                  (x + margin, y + height - margin)], 
                 fill=(0, 0, 0), width=1)
        
        # Draw some data points
        num_points = random.randint(5, 15)
        points = []
        for i in range(num_points):
            px = x + margin + (width - 2 * margin) * i / (num_points - 1)
            py = y + margin + random.randint(0, height - 2 * margin)
            points.append((px, py))
        
        # Connect points
        for i in range(len(points) - 1):
            draw.line([points[i], points[i + 1]], fill=(0, 0, 255), width=2)


class MusicScoreGenerator(SyntheticDocumentGenerator):
    """Generate synthetic music scores"""
    
    def __init__(self, output_dir: str, num_samples: int = 1000):
        super().__init__(output_dir, num_samples)
        
        # Music-specific settings
        self.time_signatures = ['4/4', '3/4', '6/8', '2/4']
        self.keys = ['C', 'G', 'D', 'A', 'F', 'Bb', 'Eb']
        self.clefs = ['treble', 'bass']
    
    def generate_dataset(self):
        """Generate synthetic music scores"""
        annotations = []
        
        for i in tqdm(range(self.num_samples), desc="Generating music scores"):
            score_image, score_annotations = self._generate_score(i)
            
            # Save image
            img_path = self.images_dir / f'score_{i:05d}.jpg'
            cv2.imwrite(str(img_path), score_image)
            
            # Create annotation entry
            annotations.append({
                'image_id': f'score_{i:05d}',
                'image_path': str(img_path),
                'width': score_image.shape[1],
                'height': score_image.shape[0],
                'annotations': score_annotations
            })
        
        # Save annotations
        self.save_annotations(annotations, 'train')
    
    def _generate_score(self, score_id: int) -> Tuple[np.ndarray, List[Dict]]:
        """Generate a single music score"""
        # Create a simple score using music21
        score = music21.stream.Score()
        part = music21.stream.Part()
        
        # Add metadata
        score.metadata = music21.metadata.Metadata()
        score.metadata.title = self.faker.catch_phrase()
        score.metadata.composer = self.faker.name()
        
        # Set time signature and key
        ts = music21.meter.TimeSignature(random.choice(self.time_signatures))
        key = music21.key.Key(random.choice(self.keys))
        part.append(ts)
        part.append(key)
        
        # Generate measures
        num_measures = random.randint(8, 16)
        for _ in range(num_measures):
            measure = music21.stream.Measure()
            
            # Fill measure with random notes
            duration_left = ts.numerator
            while duration_left > 0:
                # Random note duration
                possible_durations = [0.25, 0.5, 1.0, 2.0]
                duration = random.choice([d for d in possible_durations if d <= duration_left])
                
                # Random pitch or rest
                if random.random() < 0.8:
                    pitch = random.randint(60, 84)  # C4 to C6
                    note = music21.note.Note(pitch, quarterLength=duration)
                else:
                    note = music21.note.Rest(quarterLength=duration)
                
                measure.append(note)
                duration_left -= duration
            
            part.append(measure)
        
        score.append(part)
        
        # Convert to image
        # Note: This requires MuseScore or Lilypond to be installed
        try:
            # Save as MusicXML temporarily
            temp_path = f'/tmp/temp_score_{score_id}.xml'
            score.write('musicxml', fp=temp_path)
            
            # Convert to image using external tool
            # This is a placeholder - actual implementation would use
            # MuseScore or similar to render the score
            img = self._render_score_placeholder(score)
            
            # Generate annotations
            annotations = self._generate_score_annotations(img, score)
            
            return img, annotations
            
        except Exception as e:
            print(f"Error generating score: {e}")
            # Return placeholder image
            return self._create_placeholder_score()
    
    def _render_score_placeholder(self, score) -> np.ndarray:
        """Create a placeholder score image"""
        # This is a simplified version - real implementation would render actual notation
        width, height = 800, 600
        img = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Create PIL image for drawing
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        
        # Draw staves
        staff_height = 60
        staff_spacing = 100
        num_staves = 4
        
        for i in range(num_staves):
            y_base = 100 + i * staff_spacing
            
            # Draw 5 staff lines
            for j in range(5):
                y = y_base + j * (staff_height // 4)
                draw.line([(50, y), (width - 50, y)], fill=(0, 0, 0), width=1)
            
            # Draw clef (simplified)
            clef_x = 60
            clef_y = y_base + staff_height // 2
            draw.ellipse([clef_x - 10, clef_y - 20, clef_x + 10, clef_y + 20], 
                        outline=(0, 0, 0), width=2)
            
            # Draw time signature
            ts_x = 120
            font = random.choice(self.fonts)
            draw.text((ts_x, y_base), "4", fill=(0, 0, 0), font=font)
            draw.text((ts_x, y_base + staff_height // 2), "4", fill=(0, 0, 0), font=font)
            
            # Draw some notes (simplified)
            note_spacing = 50
            num_notes = random.randint(8, 16)
            
            for j in range(num_notes):
                note_x = 180 + j * note_spacing
                if note_x > width - 100:
                    break
                
                # Random note position on staff
                note_y = y_base + random.randint(0, staff_height)
                
                # Draw note head
                draw.ellipse([note_x - 5, note_y - 3, note_x + 5, note_y + 3], 
                           fill=(0, 0, 0))
                
                # Draw stem
                if random.random() < 0.7:
                    stem_dir = -1 if note_y > y_base + staff_height // 2 else 1
                    draw.line([(note_x + 5 * stem_dir, note_y), 
                             (note_x + 5 * stem_dir, note_y + 30 * stem_dir)], 
                            fill=(0, 0, 0), width=1)
        
        # Convert back to numpy
        img_array = np.array(pil_img)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        return img_array
    
    def _generate_score_annotations(self, img: np.ndarray, score) -> List[Dict]:
        """Generate annotations for music score elements"""
        annotations = []
        height, width = img.shape[:2]
        
        # Staff annotations
        staff_height = 60
        staff_spacing = 100
        num_staves = 4
        
        for i in range(num_staves):
            y_base = 100 + i * staff_spacing
            
            # Staff
            annotations.append({
                'bbox': [50, y_base, width - 50, y_base + staff_height],
                'category': 'staff',
                'confidence': 1.0
            })
            
            # Clef
            annotations.append({
                'bbox': [50, y_base, 80, y_base + staff_height],
                'category': 'clef',
                'confidence': 1.0
            })
            
            # Time signature
            annotations.append({
                'bbox': [110, y_base, 140, y_base + staff_height],
                'category': 'time_signature',
                'confidence': 1.0
            })
            
            # Measures (simplified)
            measure_width = (width - 200) // 4
            for j in range(4):
                measure_x = 180 + j * measure_width
                annotations.append({
                    'bbox': [measure_x, y_base, measure_x + measure_width, y_base + staff_height],
                    'category': 'measure',
                    'confidence': 1.0
                })
        
        return annotations
    
    def _create_placeholder_score(self) -> Tuple[np.ndarray, List[Dict]]:
        """Create a simple placeholder score when music21 rendering fails"""
        return self._render_score_placeholder(None), []


class FormGenerator(SyntheticDocumentGenerator):
    """Generate synthetic forms and structured documents"""
    
    def __init__(self, output_dir: str, num_samples: int = 1000):
        super().__init__(output_dir, num_samples)
        
        # Form field types
        self.field_types = ['text', 'checkbox', 'radio', 'dropdown', 'signature']
        
        # Common form fields
        self.common_fields = [
            "Full Name", "Email Address", "Phone Number", "Date of Birth",
            "Address", "City", "State", "ZIP Code", "Country",
            "Company Name", "Job Title", "Department", "Employee ID",
            "Emergency Contact", "Relationship", "Medical Conditions",
            "Signature", "Date"
        ]
    
    def generate_dataset(self):
        """Generate synthetic forms"""
        annotations = []
        
        for i in tqdm(range(self.num_samples), desc="Generating forms"):
            form_image, form_annotations = self._generate_form(i)
            
            # Save image
            img_path = self.images_dir / f'form_{i:05d}.jpg'
            cv2.imwrite(str(img_path), form_image)
            
            # Create annotation entry
            annotations.append({
                'image_id': f'form_{i:05d}',
                'image_path': str(img_path),
                'width': form_image.shape[1],
                'height': form_image.shape[0],
                'annotations': form_annotations
            })
        
        # Save annotations
        self.save_annotations(annotations, 'train')
    
    def _generate_form(self, form_id: int) -> Tuple[np.ndarray, List[Dict]]:
        """Generate a single form"""
        # Select background
        bg = random.choice(self.backgrounds).copy()
        height, width = bg.shape[:2]
        
        # Create PIL image
        img = Image.fromarray(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        
        annotations = []
        
        # Form title
        title = f"{self.faker.company()} {random.choice(['Application Form', 'Registration Form', 'Information Form', 'Request Form'])}"
        title_font = random.choice([f for f in self.fonts if f.size >= 24])
        
        title_bbox = draw.textbbox((0, 0), title, font=title_font)
        title_width = title_bbox[2] - title_bbox[0]
        title_height = title_bbox[3] - title_bbox[1]
        title_x = (width - title_width) // 2
        
        draw.text((title_x, 30), title, fill=(0, 0, 0), font=title_font)
        
        annotations.append({
            'bbox': [title_x, 30, title_x + title_width, 30 + title_height],
            'category': 'title',
            'confidence': 1.0
        })
        
        # Form fields
        y_offset = 100
        num_fields = random.randint(8, 15)
        selected_fields = random.sample(self.common_fields, min(num_fields, len(self.common_fields)))
        
        label_font = random.choice([f for f in self.fonts if f.size <= 14])
        
        for field_name in selected_fields:
            if y_offset > height - 100:
                break
            
            # Field label
            draw.text((50, y_offset), field_name + ":", fill=(0, 0, 0), font=label_font)
            
            label_bbox = draw.textbbox((0, 0), field_name + ":", font=label_font)
            label_width = label_bbox[2] - label_bbox[0]
            label_height = label_bbox[3] - label_bbox[1]
            
            annotations.append({
                'bbox': [50, y_offset, 50 + label_width, y_offset + label_height],
                'category': 'text',
                'confidence': 1.0
            })
            
            # Field input
            field_x = 250
            field_width = 300
            field_height = 30
            
            field_type = random.choice(['text', 'checkbox', 'dropdown'])
            
            if field_type == 'text':
                # Text input field
                draw.rectangle(
                    [field_x, y_offset, field_x + field_width, y_offset + field_height],
                    outline=(0, 0, 0),
                    width=1
                )
                
                # Sometimes add pre-filled text
                if random.random() < 0.3:
                    sample_text = self._generate_sample_text(field_name)
                    text_font = random.choice([f for f in self.fonts if f.size <= 12])
                    draw.text((field_x + 5, y_offset + 5), sample_text, 
                            fill=(0, 0, 0), font=text_font)
                
                annotations.append({
                    'bbox': [field_x, y_offset, field_x + field_width, y_offset + field_height],
                    'category': 'input_field',
                    'confidence': 1.0
                })
                
            elif field_type == 'checkbox':
                # Checkbox
                checkbox_size = 20
                draw.rectangle(
                    [field_x, y_offset + 5, field_x + checkbox_size, y_offset + 5 + checkbox_size],
                    outline=(0, 0, 0),
                    width=1
                )
                
                # Sometimes check it
                if random.random() < 0.3:
                    draw.line([(field_x + 3, y_offset + 8), 
                             (field_x + checkbox_size - 3, y_offset + checkbox_size + 2)],
                            fill=(0, 0, 0), width=2)
                    draw.line([(field_x + checkbox_size - 3, y_offset + 8), 
                             (field_x + 3, y_offset + checkbox_size + 2)],
                            fill=(0, 0, 0), width=2)
                
                annotations.append({
                    'bbox': [field_x, y_offset + 5, field_x + checkbox_size, y_offset + 5 + checkbox_size],
                    'category': 'checkbox',
                    'confidence': 1.0
                })
                
            elif field_type == 'dropdown':
                # Dropdown
                draw.rectangle(
                    [field_x, y_offset, field_x + field_width, y_offset + field_height],
                    outline=(0, 0, 0),
                    width=1
                )
                
                # Dropdown arrow
                arrow_x = field_x + field_width - 20
                arrow_y = y_offset + field_height // 2
                draw.polygon([(arrow_x, arrow_y - 5), 
                            (arrow_x + 10, arrow_y - 5), 
                            (arrow_x + 5, arrow_y + 5)],
                           fill=(0, 0, 0))
                
                annotations.append({
                    'bbox': [field_x, y_offset, field_x + field_width, y_offset + field_height],
                    'category': 'dropdown',
                    'confidence': 1.0
                })
            
            y_offset += field_height + 20
        
        # Signature field
        if y_offset < height - 150:
            y_offset += 20
            
            # Signature label
            draw.text((50, y_offset), "Signature:", fill=(0, 0, 0), font=label_font)
            
            sig_label_bbox = draw.textbbox((0, 0), "Signature:", font=label_font)
            annotations.append({
                'bbox': [50, y_offset, 50 + sig_label_bbox[2], y_offset + sig_label_bbox[3]],
                'category': 'text',
                'confidence': 1.0
            })
            
            # Signature field
            sig_y = y_offset + 30
            sig_width = 300
            sig_height = 60
            
            draw.rectangle(
                [50, sig_y, 50 + sig_width, sig_y + sig_height],
                outline=(0, 0, 0),
                width=1
            )
            
            # Draw line for signature
            draw.line([(50, sig_y + sig_height - 10), 
                     (50 + sig_width, sig_y + sig_height - 10)],
                    fill=(0, 0, 0), width=1)
            
            # Sometimes add a signature
            if random.random() < 0.3:
                self._draw_signature(draw, 60, sig_y + 10, sig_width - 20, sig_height - 30)
            
            annotations.append({
                'bbox': [50, sig_y, 50 + sig_width, sig_y + sig_height],
                'category': 'signature_field',
                'confidence': 1.0
            })
            
            # Date field next to signature
            date_x = 400
            draw.text((date_x, y_offset), "Date:", fill=(0, 0, 0), font=label_font)
            
            date_field_x = date_x + 50
            date_field_width = 150
            
            draw.rectangle(
                [date_field_x, sig_y + sig_height - 30, 
                 date_field_x + date_field_width, sig_y + sig_height - 5],
                outline=(0, 0, 0),
                width=1
            )
            
            # Sometimes add date
            if random.random() < 0.3:
                date_text = self.faker.date()
                date_font = random.choice([f for f in self.fonts if f.size <= 12])
                draw.text((date_field_x + 5, sig_y + sig_height - 25), 
                        date_text, fill=(0, 0, 0), font=date_font)
            
            annotations.append({
                'bbox': [date_field_x, sig_y + sig_height - 30, 
                        date_field_x + date_field_width, sig_y + sig_height - 5],
                'category': 'input_field',
                'confidence': 1.0
            })
        
        # Convert back to numpy
        img_array = np.array(img)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Apply augmentations
        if random.random() < 0.7:
            img_array = self.augmenter(image=img_array)
        
        return img_array, annotations
    
    def _generate_sample_text(self, field_name: str) -> str:
        """Generate appropriate sample text for a field"""
        field_lower = field_name.lower()
        
        if "name" in field_lower:
            return self.faker.name()
        elif "email" in field_lower:
            return self.faker.email()
        elif "phone" in field_lower:
            return self.faker.phone_number()
        elif "address" in field_lower:
            return self.faker.street_address()
        elif "city" in field_lower:
            return self.faker.city()
        elif "state" in field_lower:
            return self.faker.state()
        elif "zip" in field_lower or "postal" in field_lower:
            return self.faker.zipcode()
        elif "country" in field_lower:
            return self.faker.country()
        elif "company" in field_lower:
            return self.faker.company()
        elif "job" in field_lower or "title" in field_lower:
            return self.faker.job()
        elif "date" in field_lower:
            return self.faker.date()
        else:
            return self.faker.word()
    
    def _draw_signature(self, draw: ImageDraw.Draw, x: int, y: int, 
                       width: int, height: int):
        """Draw a simulated signature"""
        # Generate random curved lines to simulate signature
        num_strokes = random.randint(2, 4)
        
        for _ in range(num_strokes):
            points = []
            num_points = random.randint(5, 10)
            
            for i in range(num_points):
                px = x + random.randint(0, width)
                py = y + random.randint(0, height)
                points.append((px, py))
            
            # Sort points by x coordinate for smoother lines
            points.sort(key=lambda p: p[0])
            
            # Draw curved line through points
            for i in range(len(points) - 1):
                draw.line([points[i], points[i + 1]], fill=(0, 0, 150), width=2)


class SyntheticDatasetGenerator:
    """Main class to orchestrate synthetic data generation"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_all_datasets(self, samples_per_type: int = 1000):
        """Generate all types of synthetic documents"""
        
        # Generate academic documents
        print("\nGenerating Academic Documents...")
        academic_gen = AcademicDocumentGenerator(
            self.output_dir / 'academic',
            samples_per_type
        )
        academic_gen.generate_dataset()
        
        # Generate music scores
        print("\nGenerating Music Scores...")
        music_gen = MusicScoreGenerator(
            self.output_dir / 'music',
            samples_per_type // 2  # Fewer music samples as they're more specialized
        )
        music_gen.generate_dataset()
        
        # Generate forms
        print("\nGenerating Forms...")
        form_gen = FormGenerator(
            self.output_dir / 'forms',
            samples_per_type
        )
        form_gen.generate_dataset()
        
        # Merge all annotations
        self._merge_annotations()
        
        print("\nSynthetic dataset generation complete!")
    
    def _merge_annotations(self):
        """Merge annotations from all generators"""
        all_annotations = []
        
        for subdir in ['academic', 'music', 'forms']:
            ann_file = self.output_dir / subdir / 'annotations' / 'train_annotations.json'
            if ann_file.exists():
                with open(ann_file, 'r') as f:
                    annotations = json.load(f)
                    all_annotations.extend(annotations)
        
        # Save merged annotations
        merged_file = self.output_dir / 'merged_synthetic_annotations.json'
        with open(merged_file, 'w') as f:
            json.dump(all_annotations, f, indent=2)
        
        print(f"Merged {len(all_annotations)} synthetic annotations")
        
        # Generate statistics
        self._generate_statistics(all_annotations)
    
    def _generate_statistics(self, annotations: List[Dict]):
        """Generate statistics about the synthetic dataset"""
        stats = {
            'total_images': len(annotations),
            'total_annotations': 0,
            'class_distribution': {},
            'avg_annotations_per_image': 0
        }
        
        for ann in annotations:
            num_boxes = len(ann['annotations'])
            stats['total_annotations'] += num_boxes
            
            for box in ann['annotations']:
                category = box['category']
                stats['class_distribution'][category] = stats['class_distribution'].get(category, 0) + 1
        
        stats['avg_annotations_per_image'] = stats['total_annotations'] / stats['total_images']
        
        # Save statistics
        stats_file = self.output_dir / 'synthetic_dataset_stats.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Print summary
        print("\nSynthetic Dataset Statistics:")
        print(f"Total images: {stats['total_images']}")
        print(f"Total annotations: {stats['total_annotations']}")
        print(f"Average annotations per image: {stats['avg_annotations_per_image']:.2f}")
        print("\nClass distribution:")
        for class_name, count in sorted(stats['class_distribution'].items(), 
                                      key=lambda x: x[1], reverse=True):
            print(f"  {class_name}: {count}")


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic document dataset')
    parser.add_argument('--output', type=str, default='data/synthetic',
                       help='Output directory for synthetic data')
    parser.add_argument('--samples', type=int, default=1000,
                       help='Number of samples per document type')
    parser.add_argument('--types', nargs='+', 
                       default=['academic', 'music', 'forms'],
                       choices=['academic', 'music', 'forms'],
                       help='Types of documents to generate')
    
    args = parser.parse_args()
    
    if 'all' in args.types:
        # Generate all types
        generator = SyntheticDatasetGenerator(args.output)
        generator.generate_all_datasets(args.samples)
    else:
        # Generate specific types
        for doc_type in args.types:
            if doc_type == 'academic':
                gen = AcademicDocumentGenerator(
                    Path(args.output) / 'academic',
                    args.samples
                )
            elif doc_type == 'music':
                gen = MusicScoreGenerator(
                    Path(args.output) / 'music',
                    args.samples
                )
            elif doc_type == 'forms':
                gen = FormGenerator(
                    Path(args.output) / 'forms',
                    args.samples
                )
            
            gen.generate_dataset()


if __name__ == '__main__':
    main()