"""
Vision Agent Module
Handles LLM calls (GPT-4o or Gemini) with strict schema validation using Pydantic.
"""

import json
import base64
from typing import List, Dict, Optional
from pydantic import BaseModel, Field, validator
from pathlib import Path
import os


# ============================================================================
# PYDANTIC SCHEMAS - STRICT VALIDATION
# ============================================================================

class VisionAgentResponse(BaseModel):
    """Strict schema for Vision LLM response"""
    
    issue_type: str = Field(..., description="Type of damage/issue detected")
    object_part: str = Field(..., description="Specific part of object affected")
    claim_status: str = Field(..., description="Overall claim evaluation status")
    claim_status_justification: str = Field(..., description="Reasoning for status decision")
    supporting_image_ids: str = Field(..., description="Image filenames without extension, semicolon-separated")
    valid_image: List[bool] = Field(..., description="Validity boolean for each image")
    severity: str = Field(..., description="Severity level of damage")
    risk_flags: List[str] = Field(default_factory=list, description="List of detected risk flags")
    evidence_standard_met: bool = Field(..., description="Whether evidence standard is met")
    evidence_standard_met_reason: str = Field(..., description="Reasoning for evidence standard decision")
    model_call_cost: float = Field(0.0, description="Estimated API cost for this call")
    
    @validator('issue_type')
    def validate_issue_type(cls, v):
        allowed = {
            'dent', 'scratch', 'crack', 'glass_shatter', 'broken_part', 
            'missing_part', 'torn_packaging', 'crushed_packaging', 
            'water_damage', 'stain', 'none', 'unknown'
        }
        return v.lower() if v.lower() in allowed else 'unknown'
    
    @validator('object_part')
    def validate_object_part(cls, v):
        allowed = {
            # Car parts
            'front_bumper', 'rear_bumper', 'door', 'hood', 'windshield', 
            'side_mirror', 'headlight', 'taillight', 'fender', 'quarter_panel', 'body',
            # Laptop parts
            'screen', 'keyboard', 'trackpad', 'hinge', 'lid', 'corner', 'port', 'base',
            # Package parts
            'box', 'package_corner', 'package_side', 'seal', 'label', 'contents', 'item',
            # Defaults
            'unknown'
        }
        return v.lower() if v.lower() in allowed else 'unknown'
    
    @validator('claim_status')
    def validate_claim_status(cls, v):
        allowed = {'supported', 'contradicted', 'not_enough_information'}
        return v.lower() if v.lower() in allowed else 'not_enough_information'
    
    @validator('severity')
    def validate_severity(cls, v):
        allowed = {'none', 'low', 'medium', 'high', 'unknown'}
        return v.lower() if v.lower() in allowed else 'unknown'
    
    @validator('risk_flags')
    def validate_risk_flags(cls, v):
        allowed = {
            'none', 'blurry_image', 'cropped_or_obstructed', 'low_light_or_glare',
            'wrong_angle', 'wrong_object', 'wrong_object_part', 'damage_not_visible',
            'claim_mismatch', 'possible_manipulation', 'non_original_image',
            'text_instruction_present', 'user_history_risk', 'manual_review_required'
        }
        return [flag.lower() for flag in v if flag.lower() in allowed]


class VisionAgentRequest(BaseModel):
    """Request schema for Vision Agent"""
    
    user_id: str
    claim_object: str
    user_claim: str
    image_paths: List[str]
    user_history: Optional[Dict] = None
    evidence_requirements: Optional[Dict] = None


# ============================================================================
# VISION AGENT CLASS
# ============================================================================

class VisionAgent:
    """
    Orchestrates LLM calls with strict schema validation.
    Supports both OpenAI (GPT-4o) and Google (Gemini) models.
    """
    
    def __init__(self, model_provider: str = "mock", api_key: str = None):
        """
        Initialize Vision Agent
        
        Args:
            model_provider: "openai", "google", or "mock" (for testing)
            api_key: API key for the provider
        """
        self.model_provider = model_provider
        self.api_key = api_key or os.getenv("VISION_API_KEY")
        
        self.model_calls = 0
        self.images_processed = 0
        self.estimated_input_tokens = 0
        self.estimated_output_tokens = 0
        self.total_cost = 0.0
        
        print(f"✓ Vision Agent initialized with provider: {model_provider}")
    
    def encode_image_to_base64(self, image_path: str) -> Optional[str]:
        """Convert image file to base64 string"""
        try:
            path = Path(image_path)
            if not path.exists():
                print(f"⚠ Image not found: {image_path}")
                return None
            
            with open(path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"✗ Error encoding image {image_path}: {e}")
            return None
    
    def estimate_tokens(self, text: str, num_images: int) -> tuple:
        """Estimate tokens for API call"""
        # Rough estimates: ~1.3 tokens per character, ~200 tokens per image
        text_tokens = int(len(text) / 4) + 100  # +100 base
        image_tokens = num_images * 200
        
        input_tokens = text_tokens + image_tokens
        output_tokens = 500  # Estimated output
        
        return input_tokens, output_tokens
    
    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate estimated cost"""
        # GPT-4o pricing (per 1M tokens)
        input_rate = 0.003 / 1000  # $0.003 per 1K tokens
        output_rate = 0.006 / 1000  # $0.006 per 1K tokens
        
        cost = (input_tokens * input_rate) + (output_tokens * output_rate)
        return round(cost, 6)
    
    def extract_image_filename(self, path: str) -> str:
        """Extract filename without extension"""
        return Path(path).stem
    
    def create_vision_prompt(self, request: VisionAgentRequest) -> str:
        """Create detailed prompt for vision LLM"""
        
        prompt = f"""
You are an expert evidence reviewer for insurance claims. Analyze the provided images and evaluate the claim.

CLAIM DETAILS:
- User ID: {request.user_id}
- Object Type: {request.claim_object}
- User Claim: {request.user_claim}
- Number of Images: {len(request.image_paths)}

EVALUATION RULES:
1. Examine each image for damage evidence
2. Determine if the damage matches the user's claim
3. Assess image quality (blurry, obstructed, low light, etc.)
4. Check for manipulation or inconsistencies
5. Rate severity of damage
6. Make a final claim status decision

ALLOWED VALUES:
- issue_type: dent, scratch, crack, glass_shatter, broken_part, missing_part, torn_packaging, crushed_packaging, water_damage, stain, none, unknown
- object_part: For {request.claim_object.lower()}, use appropriate part names
- claim_status: supported, contradicted, not_enough_information
- severity: low, medium, high, none, unknown
- risk_flags: blurry_image, cropped_or_obstructed, low_light_or_glare, wrong_angle, wrong_object, wrong_object_part, damage_not_visible, claim_mismatch, possible_manipulation, non_original_image, text_instruction_present, user_history_risk, manual_review_required

USER HISTORY:
{json.dumps(request.user_history) if request.user_history else 'No history'}

Provide your response as valid JSON matching this exact structure:
{{
    "issue_type": "string",
    "object_part": "string",
    "claim_status": "supported|contradicted|not_enough_information",
    "claim_status_justification": "string",
    "supporting_image_ids": "comma_separated_image_names",
    "valid_image": [true/false for each image],
    "severity": "low|medium|high|none|unknown",
    "risk_flags": ["flag1", "flag2"],
    "evidence_standard_met": true/false,
    "evidence_standard_met_reason": "string"
}}
"""
        return prompt
    
    def parse_llm_response(self, response_text: str) -> Dict:
        """
        Parse and validate LLM response
        Extracts JSON and validates against schema
        """
        try:
            # Try to find JSON in response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")
            
            json_str = response_text[json_start:json_end]
            response_data = json.loads(json_str)
            
            # Validate with Pydantic
            validated = VisionAgentResponse(**response_data)
            return validated.dict()
        
        except Exception as e:
            print(f"⚠ Error parsing LLM response: {e}")
            # Return safe defaults
            return {
                'issue_type': 'unknown',
                'object_part': 'unknown',
                'claim_status': 'not_enough_information',
                'claim_status_justification': 'Failed to parse LLM response',
                'supporting_image_ids': '',
                'valid_image': [False],
                'severity': 'unknown',
                'risk_flags': ['manual_review_required'],
                'evidence_standard_met': False,
                'evidence_standard_met_reason': 'Unable to parse response',
                'model_call_cost': 0.0
            }
    
    def evaluate_claim(self, request: VisionAgentRequest) -> Dict:
        """
        Main method to evaluate a claim using vision LLM
        Returns structured response with strict validation
        """
        
        # Estimate tokens and cost
        prompt = self.create_vision_prompt(request)
        input_tokens, output_tokens = self.estimate_tokens(
            prompt + request.user_claim,
            len(request.image_paths)
        )
        cost = self.calculate_cost(input_tokens, output_tokens)
        
        # Update metrics
        self.model_calls += 1
        self.images_processed += len(request.image_paths)
        self.estimated_input_tokens += input_tokens
        self.estimated_output_tokens += output_tokens
        self.total_cost += cost
        
        # Mock response for testing (replace with actual API calls)
        if self.model_provider == "mock":
            mock_response = self.create_mock_response(request, cost)
            return mock_response
        
        # TODO: Implement actual API calls for OpenAI and Google
        print("⚠ Real LLM provider not implemented. Using mock response.")
        return self.create_mock_response(request, cost)
    
    def create_mock_response(self, request: VisionAgentRequest, cost: float) -> Dict:
        """Create mock response for testing"""
        
        image_filenames = [self.extract_image_filename(path) for path in request.image_paths]
        
        # Deterministic responses based on object type
        if 'crack' in request.user_claim.lower() or 'windshield' in request.user_claim.lower():
            response = {
                'issue_type': 'crack',
                'object_part': 'windshield' if 'car' in request.claim_object.lower() else 'screen',
                'claim_status': 'supported',
                'claim_status_justification': 'Clear evidence of crack visible in images',
                'supporting_image_ids': ';'.join(image_filenames),
                'valid_image': [True] * len(request.image_paths),
                'severity': 'high',
                'risk_flags': [],
                'evidence_standard_met': True,
                'evidence_standard_met_reason': f'Valid images: {len(request.image_paths)}/{len(request.image_paths)}',
                'model_call_cost': cost
            }
        elif 'scratch' in request.user_claim.lower() or 'screen' in request.user_claim.lower():
            response = {
                'issue_type': 'scratch',
                'object_part': 'screen',
                'claim_status': 'contradicted',
                'claim_status_justification': 'Images show minor wear, inconsistent with claim severity',
                'supporting_image_ids': image_filenames[0] if image_filenames else '',
                'valid_image': [False if i > 0 else True for i in range(len(request.image_paths))],
                'severity': 'low',
                'risk_flags': ['blurry_image'],
                'evidence_standard_met': False,
                'evidence_standard_met_reason': 'Insufficient clear evidence',
                'model_call_cost': cost
            }
        else:
            response = {
                'issue_type': 'unknown',
                'object_part': 'unknown',
                'claim_status': 'not_enough_information',
                'claim_status_justification': 'Unable to determine claim validity from images',
                'supporting_image_ids': '',
                'valid_image': [False] * len(request.image_paths),
                'severity': 'unknown',
                'risk_flags': ['damage_not_visible'],
                'evidence_standard_met': False,
                'evidence_standard_met_reason': 'Damage not clearly visible in provided images',
                'model_call_cost': cost
            }
        
        # Add user history risk if applicable
        if request.user_history and request.user_history.get('risk_level') == 'high':
            if 'user_history_risk' not in response['risk_flags']:
                response['risk_flags'].append('user_history_risk')
        
        return response
    
    def get_metrics(self) -> Dict:
        """Return current metrics"""
        return {
            'model_calls': self.model_calls,
            'images_processed': self.images_processed,
            'estimated_input_tokens': self.estimated_input_tokens,
            'estimated_output_tokens': self.estimated_output_tokens,
            'total_cost': round(self.total_cost, 6)
        }
