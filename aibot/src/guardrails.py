"""
Guardrails Module
Post-processing validation and error handling for LLM responses.
Ensures graceful degradation with safe defaults for invalid values.
"""

from typing import Dict, List, Optional
import re


class GuardrailValidator:
    """
    Validates and sanitizes LLM outputs
    Handles invalid/hallucinated values with graceful defaults
    """
    
    # Allowed values mapping
    ALLOWED_VALUES = {
        'issue_type': {
            'dent', 'scratch', 'crack', 'glass_shatter', 'broken_part',
            'missing_part', 'torn_packaging', 'crushed_packaging',
            'water_damage', 'stain', 'none', 'unknown'
        },
        'claim_status': {
            'supported', 'contradicted', 'not_enough_information'
        },
        'severity': {
            'none', 'low', 'medium', 'high', 'unknown'
        },
        'risk_flags': {
            'none', 'blurry_image', 'cropped_or_obstructed', 'low_light_or_glare',
            'wrong_angle', 'wrong_object', 'wrong_object_part', 'damage_not_visible',
            'claim_mismatch', 'possible_manipulation', 'non_original_image',
            'text_instruction_present', 'user_history_risk', 'manual_review_required'
        },
        'object_part_car': {
            'front_bumper', 'rear_bumper', 'door', 'hood', 'windshield',
            'side_mirror', 'headlight', 'taillight', 'fender', 'quarter_panel', 'body', 'unknown'
        },
        'object_part_laptop': {
            'screen', 'keyboard', 'trackpad', 'hinge', 'lid', 'corner', 'port', 'base', 'body', 'unknown'
        },
        'object_part_package': {
            'box', 'package_corner', 'package_side', 'seal', 'label', 'contents', 'item', 'unknown'
        }
    }
    
    def __init__(self):
        self.validation_log = []
    
    def log_validation(self, field: str, original: str, corrected: str, reason: str):
        """Log validation changes for audit trail"""
        self.validation_log.append({
            'field': field,
            'original': original,
            'corrected': corrected,
            'reason': reason
        })
    
    def validate_issue_type(self, value: str) -> str:
        """Validate issue_type field"""
        if not value or not isinstance(value, str):
            self.log_validation('issue_type', str(value), 'unknown', 'Invalid type')
            return 'unknown'
        
        normalized = value.lower().strip()
        
        if normalized in self.ALLOWED_VALUES['issue_type']:
            return normalized
        
        # Try partial match
        for allowed in self.ALLOWED_VALUES['issue_type']:
            if allowed in normalized or normalized in allowed:
                self.log_validation('issue_type', value, allowed, 'Partial match found')
                return allowed
        
        self.log_validation('issue_type', value, 'unknown', 'Not in allowed values')
        return 'unknown'
    
    def validate_object_part(self, value: str, claim_object: str) -> str:
        """Validate object_part based on claim_object type"""
        if not value or not isinstance(value, str):
            self.log_validation('object_part', str(value), 'unknown', 'Invalid type')
            return 'unknown'
        
        normalized = value.lower().strip()
        claim_object = claim_object.lower().strip()
        
        # Select allowed parts based on claim object
        if claim_object == 'car':
            allowed_parts = self.ALLOWED_VALUES['object_part_car']
        elif claim_object == 'laptop':
            allowed_parts = self.ALLOWED_VALUES['object_part_laptop']
        elif claim_object == 'package':
            allowed_parts = self.ALLOWED_VALUES['object_part_package']
        else:
            allowed_parts = self.ALLOWED_VALUES['object_part_car'] | \
                           self.ALLOWED_VALUES['object_part_laptop'] | \
                           self.ALLOWED_VALUES['object_part_package']
        
        if normalized in allowed_parts:
            return normalized
        
        # Try partial match
        for allowed in allowed_parts:
            if allowed in normalized or normalized in allowed:
                self.log_validation('object_part', value, allowed, 'Partial match found')
                return allowed
        
        self.log_validation('object_part', value, 'unknown', 'Not in allowed values')
        return 'unknown'
    
    def validate_claim_status(self, value: str) -> str:
        """Validate claim_status field"""
        if not value or not isinstance(value, str):
            self.log_validation('claim_status', str(value), 'not_enough_information', 'Invalid type')
            return 'not_enough_information'
        
        normalized = value.lower().strip()
        
        if normalized in self.ALLOWED_VALUES['claim_status']:
            return normalized
        
        # Try to map similar values
        if 'support' in normalized or 'valid' in normalized or 'true' in normalized:
            self.log_validation('claim_status', value, 'supported', 'Semantic mapping')
            return 'supported'
        elif 'contradict' in normalized or 'invalid' in normalized or 'false' in normalized:
            self.log_validation('claim_status', value, 'contradicted', 'Semantic mapping')
            return 'contradicted'
        
        self.log_validation('claim_status', value, 'not_enough_information', 'Not in allowed values')
        return 'not_enough_information'
    
    def validate_severity(self, value: str) -> str:
        """Validate severity field"""
        if not value or not isinstance(value, str):
            self.log_validation('severity', str(value), 'unknown', 'Invalid type')
            return 'unknown'
        
        normalized = value.lower().strip()
        
        if normalized in self.ALLOWED_VALUES['severity']:
            return normalized
        
        # Try semantic mapping
        if any(word in normalized for word in ['high', 'severe', 'major', 'critical']):
            self.log_validation('severity', value, 'high', 'Semantic mapping')
            return 'high'
        elif any(word in normalized for word in ['medium', 'moderate', 'mid']):
            self.log_validation('severity', value, 'medium', 'Semantic mapping')
            return 'medium'
        elif any(word in normalized for word in ['low', 'minor', 'light']):
            self.log_validation('severity', value, 'low', 'Semantic mapping')
            return 'low'
        
        self.log_validation('severity', value, 'unknown', 'Not in allowed values')
        return 'unknown'
    
    def validate_risk_flags(self, values: List[str]) -> List[str]:
        """Validate risk_flags list"""
        if not isinstance(values, list):
            self.log_validation('risk_flags', str(values), '[]', 'Not a list')
            return []
        
        validated_flags = []
        
        for flag in values:
            if not isinstance(flag, str):
                continue
            
            normalized = flag.lower().strip()
            
            if normalized in self.ALLOWED_VALUES['risk_flags']:
                validated_flags.append(normalized)
            else:
                # Try to find close match
                found = False
                for allowed in self.ALLOWED_VALUES['risk_flags']:
                    if allowed in normalized or normalized in allowed:
                        validated_flags.append(allowed)
                        found = True
                        break
                
                if not found:
                    self.log_validation('risk_flags', flag, 'skipped', 'Not in allowed values')
        
        # Remove duplicates and 'none' if other flags exist
        validated_flags = list(set(validated_flags))
        if len(validated_flags) > 1 and 'none' in validated_flags:
            validated_flags.remove('none')
        
        return validated_flags if validated_flags else ['none']
    
    def validate_supporting_image_ids(self, value: str, image_paths: List[str]) -> str:
        """
        Validate supporting_image_ids
        Extract filenames without extensions from image_paths
        """
        if not isinstance(value, str):
            value = str(value) if value else ''
        
        # Extract filenames from provided image paths
        import os
        expected_ids = [os.path.splitext(os.path.basename(path))[0] for path in image_paths]
        
        if not value or value.strip() == '':
            # If empty, use first valid image
            return expected_ids[0] if expected_ids else ''
        
        # Normalize provided IDs
        provided_ids = [id.strip() for id in value.split(';') if id.strip()]
        
        # Validate that provided IDs match expected image filenames
        validated_ids = [id for id in provided_ids if id in expected_ids]
        
        if not validated_ids:
            # If no valid IDs provided, use expected ones
            validated_ids = expected_ids
            self.log_validation('supporting_image_ids', value, ';'.join(validated_ids), 'Reset to expected IDs')
        
        return ';'.join(validated_ids)
    
    def validate_valid_image(self, values: List[bool], num_images: int) -> List[bool]:
        """Validate valid_image boolean list"""
        if not isinstance(values, list):
            self.log_validation('valid_image', str(values), f'[False]*{num_images}', 'Not a list')
            return [False] * num_images
        
        # Ensure length matches number of images
        if len(values) != num_images:
            self.log_validation('valid_image', str(values), f'Resized to {num_images}', 'Length mismatch')
            # Pad or truncate
            return (values + [False] * num_images)[:num_images]
        
        # Convert all values to boolean
        return [bool(v) for v in values]
    
    def validate_evidence_standard_met(self, value: bool, claim_status: str) -> bool:
        """Validate evidence_standard_met boolean"""
        if not isinstance(value, bool):
            self.log_validation('evidence_standard_met', str(value), 'False', 'Not a boolean')
            return False
        
        # Consistency check: if status is 'supported', evidence should be met
        if claim_status == 'supported' and not value:
            self.log_validation('evidence_standard_met', str(value), 'True', 'Inconsistent with status=supported')
            return True
        
        return value
    
    def validate_model_call_cost(self, value) -> float:
        """Validate model_call_cost is a valid float"""
        try:
            cost = float(value) if value else 0.0
            return max(0.0, round(cost, 6))  # Ensure non-negative
        except (ValueError, TypeError):
            self.log_validation('model_call_cost', str(value), '0.0', 'Invalid float')
            return 0.0
    
    def validate_response(self, response: Dict, claim_object: str, image_paths: List[str]) -> Dict:
        """
        Validate entire response and apply guardrails
        Returns corrected response with all invalid values replaced by safe defaults
        """
        self.validation_log.clear()
        
        validated = {
            'issue_type': self.validate_issue_type(response.get('issue_type', 'unknown')),
            'object_part': self.validate_object_part(response.get('object_part', 'unknown'), claim_object),
            'claim_status': self.validate_claim_status(response.get('claim_status', 'not_enough_information')),
            'claim_status_justification': str(response.get('claim_status_justification', 'No justification provided')),
            'supporting_image_ids': self.validate_supporting_image_ids(
                response.get('supporting_image_ids', ''),
                image_paths
            ),
            'valid_image': self.validate_valid_image(
                response.get('valid_image', []),
                len(image_paths)
            ),
            'severity': self.validate_severity(response.get('severity', 'unknown')),
            'risk_flags': self.validate_risk_flags(response.get('risk_flags', [])),
            'evidence_standard_met': self.validate_evidence_standard_met(
                response.get('evidence_standard_met', False),
                response.get('claim_status', 'not_enough_information')
            ),
            'evidence_standard_met_reason': str(response.get('evidence_standard_met_reason', 'No reason provided')),
            'model_call_cost': self.validate_model_call_cost(response.get('model_call_cost', 0.0))
        }
        
        return validated
    
    def get_validation_report(self) -> Dict:
        """Return validation log and statistics"""
        return {
            'total_validations': len(self.validation_log),
            'corrections_made': len([v for v in self.validation_log if v['original'] != v['corrected']]),
            'log': self.validation_log
        }


class ResponseSanitizer:
    """Sanitizes text responses to prevent injection attacks"""
    
    @staticmethod
    def sanitize_text(text: str, max_length: int = 1000) -> str:
        """Remove potentially harmful content from text"""
        if not isinstance(text, str):
            return ''
        
        # Remove special characters except common punctuation
        text = re.sub(r'[<>"{};]', '', text)
        
        # Limit length
        text = text[:max_length]
        
        return text.strip()
    
    @staticmethod
    def sanitize_response(response: Dict) -> Dict:
        """Sanitize text fields in response"""
        sanitized = response.copy()
        
        if 'claim_status_justification' in sanitized:
            sanitized['claim_status_justification'] = ResponseSanitizer.sanitize_text(
                sanitized['claim_status_justification']
            )
        
        if 'evidence_standard_met_reason' in sanitized:
            sanitized['evidence_standard_met_reason'] = ResponseSanitizer.sanitize_text(
                sanitized['evidence_standard_met_reason']
            )
        
        return sanitized
