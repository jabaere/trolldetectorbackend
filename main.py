from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict
import cv2
import numpy as np
from deepface import DeepFace
import re
from transformers import pipeline
import requests
import os
import json
from io import BytesIO
import logging
from pydantic import BaseModel
from bs4 import BeautifulSoup
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Troll Detector API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables
PERSPECTIVE_API_KEY = os.getenv("PERSPECTIVE_API_KEY")
LIBRE_TRANSLATE_API_URL = "https://libretranslate.de/translate"  # Free translation API

# Initialize NLP models
try:
    name_classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
    logger.info("NLP model loaded successfully")
except Exception as e:
    logger.error(f"Error loading NLP model: {e}")
    name_classifier = None

class AnalysisResult(BaseModel):
    overall_score: float
    image_analysis: dict
    name_analysis: dict
    comment_analysis: Optional[dict] = None
    summary: str

class ProfileData(BaseModel):
    name: Optional[str] = None
    image_url: Optional[str] = None

@app.get("/")
async def root():
    return {"message": "Troll Detector API is running"}

@app.post("/v1/analyze", response_model=AnalysisResult)
async def analyze_profile(
    profile_image: Optional[UploadFile] = File(None),
    profile_name: str = Form(""),
    comments: str = Form(""),
    country: str = Form("ge")  # Default to Georgia
):
    try:
        # Process image if provided
        image_analysis = {}
        if profile_image:
            image_analysis = await analyze_image(profile_image)
        else:
            image_analysis = {"fake_probability": 0.0, "attributes": []}
        
        # Analyze name with country context
        name_analysis = analyze_name(profile_name, country)
        
        # Analyze comments if provided
        comment_analysis = None
        if comments.strip():
            # Translate comments from Georgian to English if needed
            translated_comments = await translate_text(comments, "ka", "en")
            comment_analysis = analyze_comments(translated_comments)
            logger.info(f"translated Comment: {translated_comments}")
            # print(translated_comments)
        # Calculate overall score (weighted average)
        image_weight = 0.6
        name_weight = 0.4
        
        if comment_analysis:
            # Adjust weights if comment analysis is available
            image_weight = 0.4
            name_weight = 0.3
            comment_weight = 0.3
            
            overall_score = (
                image_analysis["fake_probability"] * image_weight +
                name_analysis["fake_probability"] * name_weight +
                comment_analysis["toxicity_score"] * comment_weight
            )
        else:
            overall_score = (
                image_analysis["fake_probability"] * image_weight +
                name_analysis["fake_probability"] * name_weight
            )
        
        # Generate summary
        summary = generate_summary(overall_score, image_analysis, name_analysis, comment_analysis)
        
        return {
            "overall_score": overall_score,
            "image_analysis": image_analysis,
            "name_analysis": name_analysis,
            "comment_analysis": comment_analysis,
            "summary": summary
        }
    
    except Exception as e:
        logger.error(f"Error in profile analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# async def translate_text(text: str, source_lang: str, target_lang: str) -> str:
#     """Translate text from source language to target language using LibreTranslate API"""
#     try:
#         # First try LibreTranslate
#         payload = {
#             "q": text,
#             "source": source_lang,
#             "target": target_lang,
#             "format": "text"
#         }
        
#         headers = {"Content-Type": "application/json"}
        
#         response = requests.post(LIBRE_TRANSLATE_API_URL, 
#                                 json=payload, 
#                                 headers=headers)
        
#         if response.status_code == 200:
#             result = response.json()
#             if "translatedText" in result:
#                 return result["translatedText"]
        
#         # Fallback to Google Translate API (no API key required for this endpoint, but has limitations)
#         # Note: This is not an official API and may stop working
#         url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl={source_lang}&tl={target_lang}&dt=t&q={text}"
#         response = requests.get(url)
        
#         if response.status_code == 200:
#             result = response.json()
#             if result and isinstance(result, list) and len(result) > 0:
#                 translated_text = ""
#                 for sentence in result[0]:
#                     if sentence and isinstance(sentence, list) and len(sentence) > 0:
#                         translated_text += sentence[0]
#                 return translated_text
        
#         # If both methods fail, return original text
#         logger.warning("Translation failed, returning original text")
#         return text
        
#     except Exception as e:
#         logger.error(f"Translation error: {e}")
#         # Return original text if translation fails
#         return text

import requests
import logging

logger = logging.getLogger("main")

async def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    """Translate text using the unofficial Google Translate endpoint"""
    try:
        url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl={source_lang}&tl={target_lang}&dt=t&q={text}"
        response = requests.get(url)

        if response.status_code == 200:
            result = response.json()
            translated_text = "".join([s[0] for s in result[0] if s[0]])
            return translated_text

        logger.warning(f"Google Translate failed with status {response.status_code}")
        return text

    except Exception as e:
        logger.error(f"Translation error: {e}")
        return text



async def extract_data_from_profile_url(url: str) -> Dict[str, str]:
    """Extract profile name and image URL from a Facebook profile URL"""
    try:
        # Add user agent to avoid being blocked
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            logger.warning(f"Failed to fetch profile page: {response.status_code}")
            return {}
        
        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract data
        extracted_data = {}
        
        # Try to extract name (this is approximate and may need adjustment)
        # Facebook dynamically loads content, so this is a best effort approach
        title = soup.find('title')
        if title and title.text:
            # Facebook titles are often in format "Name - Facebook"
            name_match = re.match(r'(.+?)(?:\s+[-|]\s+Facebook)?$', title.text)
            if name_match:
                extracted_data["name"] = name_match.group(1).strip()
        
        # Try to extract profile image
        # This is challenging due to Facebook's dynamic loading and structure
        # Look for og:image meta tag which often contains profile picture
        og_image = soup.find('meta', property='og:image')
        if og_image and og_image.get('content'):
            extracted_data["image_url"] = og_image.get('content')
        
        logger.info(f"Extracted data from profile URL: {extracted_data}")
        return extracted_data
    
    except Exception as e:
        logger.error(f"Error extracting data from profile URL: {e}")
        return {}

async def analyze_image(image: UploadFile):
    try:
        # Read image
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Could not decode image")
        
        # Analyze with DeepFace
        analysis = DeepFace.analyze(img, actions=['age', 'gender', 'race', 'emotion'], enforce_detection=False)
        
        # Check if it's a single result or list
        if isinstance(analysis, list):
            analysis = analysis[0]
        
        # Extract attributes
        attributes = [
            {"name": "Age", "value": analysis["age"]},
            {"name": "Gender", "value": analysis["dominant_gender"]},
            {"name": "Emotion", "value": analysis["dominant_emotion"]},
            {"name": "Race", "value": analysis["dominant_race"]}
        ]
        
        # Detect potential fake indicators
        fake_probability = 0.0
        
        # Check for perfect symmetry (potential AI generated)
        h, w = img.shape[:2]
        left_half = img[:, :w//2]
        right_half = cv2.flip(img[:, w//2:], 1)
        
        if left_half.shape == right_half.shape:
            symmetry_score = cv2.matchTemplate(left_half, right_half, cv2.TM_CCOEFF_NORMED)[0][0]
            # Extremely high symmetry is suspicious
            if symmetry_score > 0.9:
                fake_probability += 0.3
        
        # Check for unusual smoothness (potential filter/generated)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Very low variance indicates unusual smoothness
        if laplacian_var < 100:
            fake_probability += 0.3
        
        # Check for unnatural eye/face proportions
        try:
            faces = DeepFace.extract_faces(img, enforce_detection=False)
            if faces:
                face = faces[0]
                if "facial_area" in face:
                    facial_area = face["facial_area"]
                    face_width = facial_area["w"]
                    face_height = facial_area["h"]
                    
                    # Check for unusual aspect ratio
                    if face_width / face_height > 1.5 or face_height / face_width > 1.5:
                        fake_probability += 0.2
        except Exception as e:
            logger.warning(f"Face extraction error: {e}")
        
        # Cap at 1.0
        fake_probability = min(fake_probability, 1.0)
        
        return {
            "fake_probability": fake_probability,
            "attributes": attributes
        }
    
    except Exception as e:
        logger.error(f"Image analysis error: {e}")
        return {
            "fake_probability": 0.5,  # Default to medium risk if analysis fails
            "attributes": [{"name": "Error", "value": str(e)}]
        }

async def analyze_image_from_url(image_url: str):
    try:
        # Download image from URL
        response = requests.get(image_url)
        if response.status_code != 200:
            raise ValueError(f"Failed to download image from URL: {response.status_code}")
        
        # Convert to OpenCV format
        nparr = np.frombuffer(response.content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Could not decode image from URL")
        
        # The rest of the analysis is the same as in analyze_image function
        # Analyze with DeepFace
        analysis = DeepFace.analyze(img, actions=['age', 'gender', 'race', 'emotion'], enforce_detection=False)
        
        # Check if it's a single result or list
        if isinstance(analysis, list):
            analysis = analysis[0]
        
        # Extract attributes
        attributes = [
            {"name": "Age", "value": analysis["age"]},
            {"name": "Gender", "value": analysis["dominant_gender"]},
            {"name": "Emotion", "value": analysis["dominant_emotion"]},
            {"name": "Race", "value": analysis["dominant_race"]}
        ]
        
        # Detect potential fake indicators
        fake_probability = 0.0
        
        # Check for perfect symmetry (potential AI generated)
        h, w = img.shape[:2]
        left_half = img[:, :w//2]
        right_half = cv2.flip(img[:, w//2:], 1)
        
        if left_half.shape == right_half.shape:
            symmetry_score = cv2.matchTemplate(left_half, right_half, cv2.TM_CCOEFF_NORMED)[0][0]
            # Extremely high symmetry is suspicious
            if symmetry_score > 0.9:
                fake_probability += 0.3
        
        # Check for unusual smoothness (potential filter/generated)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Very low variance indicates unusual smoothness
        if laplacian_var < 100:
            fake_probability += 0.3
        
        # Check for unnatural eye/face proportions
        try:
            faces = DeepFace.extract_faces(img, enforce_detection=False)
            if faces:
                face = faces[0]
                if "facial_area" in face:
                    facial_area = face["facial_area"]
                    face_width = facial_area["w"]
                    face_height = facial_area["h"]
                    
                    # Check for unusual aspect ratio
                    if face_width / face_height > 1.5 or face_height / face_width > 1.5:
                        fake_probability += 0.2
        except Exception as e:
            logger.warning(f"Face extraction error: {e}")
        
        # Cap at 1.0
        fake_probability = min(fake_probability, 1.0)
        
        return {
            "fake_probability": fake_probability,
            "attributes": attributes
        }
    
    except Exception as e:
        logger.error(f"Image URL analysis error: {e}")
        return {
            "fake_probability": 0.5,  # Default to medium risk if analysis fails
            "attributes": [{"name": "Error", "value": str(e)}]
        }

def analyze_name(name: str, country: str = "ge"):
    """
    Analyze a name for suspicious patterns, taking into account country-specific naming conventions.
    
    Args:
        name: The name to analyze
        country: Country code (default: "ge" for Georgia)
        
    Returns:
        Dictionary with fake_probability and patterns
    """
    try:
        fake_probability = 0.0
        
        if not name:
            return {"fake_probability": 0.5, "patterns": [], "country_specific": []}
        
        patterns = []
        country_specific = []
        
        # Split into first name and last name (if possible)
        name_parts = name.strip().split()
        
        # Georgian-specific name analysis
        if country.lower() == "ge":
            # Check for common Georgian surname endings
            georgian_surname_patterns = [
                (r'შვილი$', "Georgian surname ending with -shvili"),
                (r'ძე$', "Georgian surname ending with -dze"),
                (r'ია$', "Georgian surname ending with -ia"),
                (r'ანი$', "Georgian surname ending with -ani"),
                (r'ური$', "Georgian surname ending with -uri"),
                (r'ავა$', "Georgian surname ending with -ava"),
                (r'იძე$', "Georgian surname ending with -idze"),
                (r'ელი$', "Georgian surname ending with -eli")
            ]
            
            # Transliterated versions for Latin script
            georgian_surname_latin = [
                (r'shvili$', "Georgian surname ending with -shvili"),
                (r'dze$', "Georgian surname ending with -dze"),
                (r'ia$', "Georgian surname ending with -ia"),
                (r'ani$', "Georgian surname ending with -ani"),
                (r'uri$', "Georgian surname ending with -uri"),
                (r'ava$', "Georgian surname ending with -ava"),
                (r'idze$', "Georgian surname ending with -idze"),
                (r'eli$', "Georgian surname ending with -eli")
            ]
            
            # Check if the name has Georgian characters
            has_georgian_chars = bool(re.search(r'[\u10A0-\u10FF]', name))
            
            # Apply appropriate patterns based on script
            surname_patterns = georgian_surname_patterns if has_georgian_chars else georgian_surname_latin
            
            # Check if last name follows Georgian patterns
            if len(name_parts) > 1:
                last_name = name_parts[-1]
                
                has_valid_surname = False
                for pattern, description in surname_patterns:
                    if re.search(pattern, last_name, re.IGNORECASE):
                        country_specific.append(f"Valid {description}")
                        has_valid_surname = True
                        # Reduce fake probability for valid Georgian surnames
                        fake_probability -= 0.2
                        break
                
                if not has_valid_surname:
                    country_specific.append("Surname doesn't match Georgian patterns")
                    fake_probability += 0.2
            
            # Check for Georgian first name patterns
            if len(name_parts) > 0:
                first_name = name_parts[0]
                
                # Common Georgian first name endings
                georgian_first_name_endings = ['a', 'i', 'o'] if not has_georgian_chars else ['ა', 'ი', 'ო']
                
                if any(first_name.lower().endswith(ending) for ending in georgian_first_name_endings):
                    country_specific.append("First name has typical Georgian ending")
                    # Reduce fake probability for valid Georgian first names
                    fake_probability -= 0.1
        
        # General suspicious patterns (apply to any country)
        if re.search(r'\d', name):
            patterns.append("Contains numbers")
            fake_probability += 0.2
        
        if re.search(r'[^\w\s\u10A0-\u10FF]', name):  # Include Georgian Unicode range
            patterns.append("Contains special characters")
            fake_probability += 0.2
        
        # Check for random capitalization (for Latin script)
        if not re.search(r'[\u10A0-\u10FF]', name) and re.search(r'[A-Z][a-z]*[A-Z]', name):
            patterns.append("Unusual capitalization")
            fake_probability += 0.15
        
        # Check for excessive repeating characters
        if re.search(r'(.)\1{2,}', name):
            patterns.append("Repeating characters")
            fake_probability += 0.15
        
        # Use NLP model if available (for non-Georgian names or transliterated Georgian names)
        if name_classifier and not re.search(r'[\u10A0-\u10FF]', name):
            result = name_classifier(name)
            # Assuming negative sentiment might indicate fake name
            if result[0]['label'] == 'NEGATIVE' and result[0]['score'] > 0.7:
                patterns.append("Unusual name pattern detected by NLP")
                fake_probability += 0.3
        
        # Ensure fake_probability is between 0 and 1
        fake_probability = max(0.0, min(fake_probability, 1.0))
        
        return {
            "fake_probability": fake_probability,
            "patterns": patterns,
            "country_specific": country_specific
        }
    
    except Exception as e:
        logger.error(f"Name analysis error: {e}")
        return {"fake_probability": 0.0, "patterns": [], "country_specific": []}

def analyze_comments(comments_text: str):
    try:
        if not comments_text or not PERSPECTIVE_API_KEY:
            return {"toxicity_score": 0.0, "toxic_comments": []}
        
        comments_list = [c.strip() for c in comments_text.split('\n') if c.strip()]
        toxic_comments = []
        total_toxicity = 0.0
        
        for comment in comments_list:
            if not comment:
                continue
                
            # Call Perspective API
            analyze_request = {
                'comment': {'text': comment},
                'requestedAttributes': {'TOXICITY': {}}
            }
            
            response = requests.post(
                f'https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze?key={PERSPECTIVE_API_KEY}',
                json=analyze_request
            )
            
            if response.status_code == 200:
                data = response.json()
                toxicity = data['attributeScores']['TOXICITY']['summaryScore']['value']
                total_toxicity += toxicity
                
                if toxicity > 0.7:  # High toxicity threshold
                    toxic_comments.append({
                        "text": comment,
                        "score": toxicity
                    })
        
        # Calculate average toxicity
        avg_toxicity = total_toxicity / len(comments_list) if comments_list else 0.0
        logger.info(f"Comment: {comment}")
        logger.info(f"Status Code: {response.status_code}")
        logger.info(f"Response JSON: {response.json()}")

        return {
            "toxicity_score": avg_toxicity,
            "toxic_comments": toxic_comments
        }
    
    except Exception as e:
        logger.error(f"Comment analysis error: {e}")
        return {"toxicity_score": 0.0, "toxic_comments": []}

def generate_summary(overall_score: float, image_analysis: dict, name_analysis: dict, comment_analysis: Optional[dict] = None) -> str:
    if overall_score < 0.3:
        risk_level = "low"
    elif overall_score < 0.7:
        risk_level = "moderate"
    else:
        risk_level = "high"
    
    summary = f"This profile presents a {risk_level} risk of being a troll account. "
    
    # Add image analysis
    if image_analysis["fake_probability"] > 0.7:
        summary += "The profile image shows strong indicators of being AI-generated or heavily manipulated. "
    elif image_analysis["fake_probability"] > 0.3:
        summary += "The profile image has some unusual characteristics that may indicate manipulation. "
    else:
        summary += "The profile image appears to be authentic. "
    
    # Add name analysis with country-specific context
    if "country_specific" in name_analysis and name_analysis["country_specific"]:
        if name_analysis["fake_probability"] > 0.7:
            summary += "The name contains multiple suspicious patterns and doesn't follow typical Georgian naming conventions. "
        elif name_analysis["fake_probability"] > 0.3:
            summary += "The name contains some unusual patterns that may indicate a fake account, though some Georgian naming elements are present. "
        else:
            summary += "The name follows typical Georgian naming conventions. "
    else:
        if name_analysis["fake_probability"] > 0.7:
            summary += "The name contains multiple suspicious patterns typical of fake accounts. "
        elif name_analysis["fake_probability"] > 0.3:
            summary += "The name contains some unusual patterns that may indicate a fake account. "
    
    # Add comment analysis if available
    if comment_analysis and comment_analysis["toxicity_score"] > 0:
        if comment_analysis["toxicity_score"] > 0.7:
            summary += f"The comments show a high level of toxicity with {len(comment_analysis['toxic_comments'])} flagged comments. "
        elif comment_analysis["toxicity_score"] > 0.3:
            summary += "The comments show a moderate level of toxicity. "
        else:
            summary += "The comments appear to be non-toxic. "
    
    return summary

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
