"""
Data Loader Module
Loads and manages CSV datasets for claims processing, user history, and evidence requirements.
"""

import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path


class DataLoader:
    """Load and manage data from CSV files"""
    
    def __init__(self, dataset_dir: str = "dataset"):
        self.dataset_dir = Path(dataset_dir)
        self.dataset_dir.mkdir(exist_ok=True)
        
        self.claims_df = None
        self.user_history_df = None
        self.evidence_requirements_df = None
        
        self._create_sample_datasets()
        self.load_all_data()
    
    def _create_sample_datasets(self):
        """Create sample CSV files if they don't exist"""
        
        # Create claims.csv
        claims_path = self.dataset_dir / "claims.csv"
        if not claims_path.exists():
            claims_data = {
                'claim_id': ['claim_001', 'claim_002', 'claim_003', 'claim_004', 'claim_005'],
                'user_id': ['user_001', 'user_002', 'user_003', 'user_001', 'user_004'],
                'claim_object': ['car', 'laptop', 'package', 'car', 'laptop'],
                'user_claim': [
                    'Windshield has a large crack from rock impact',
                    'Screen has multiple scratches and wear marks',
                    'Box is crushed on multiple sides',
                    'Front bumper has a dent from minor collision',
                    'Keyboard keys are broken and unresponsive'
                ],
                'image_paths': [
                    'images/car_windshield_1.jpg;images/car_windshield_2.jpg',
                    'images/laptop_screen_1.jpg;images/laptop_screen_2.jpg',
                    'images/package_1.jpg;images/package_2.jpg',
                    'images/car_bumper_1.jpg',
                    'images/laptop_keyboard_1.jpg;images/laptop_keyboard_2.jpg'
                ],
                'ground_truth_status': ['supported', 'contradicted', 'not_enough_information', 'supported', 'supported']
            }
            pd.DataFrame(claims_data).to_csv(claims_path, index=False)
        
        # Create user_history.csv
        user_history_path = self.dataset_dir / "user_history.csv"
        if not user_history_path.exists():
            user_history_data = {
                'user_id': ['user_001', 'user_002', 'user_003', 'user_004'],
                'total_claims': [5, 3, 1, 8],
                'denied_claims': [1, 2, 0, 2],
                'fraud_score': [0.25, 0.65, 0.0, 0.3],
                'risk_level': ['low', 'high', 'none', 'low']
            }
            pd.DataFrame(user_history_data).to_csv(user_history_path, index=False)
        
        # Create evidence_requirements.csv
        evidence_req_path = self.dataset_dir / "evidence_requirements.csv"
        if not evidence_req_path.exists():
            evidence_data = {
                'claim_object': ['car', 'laptop', 'package'],
                'min_images': [1, 1, 2],
                'allowed_issue_types': [
                    'dent|crack|glass_shatter|broken_part',
                    'crack|broken_part|water_damage|stain',
                    'crushed_packaging|torn_packaging|water_damage'
                ],
                'allowed_object_parts': [
                    'front_bumper|rear_bumper|door|hood|windshield|side_mirror|headlight|taillight|fender|quarter_panel|body',
                    'screen|keyboard|trackpad|hinge|lid|corner|port|base|body',
                    'box|package_corner|package_side|seal|label|contents|item'
                ]
            }
            pd.DataFrame(evidence_data).to_csv(evidence_req_path, index=False)
        
        # Create sample_claims.csv for evaluation
        sample_claims_path = self.dataset_dir / "sample_claims.csv"
        if not sample_claims_path.exists():
            sample_data = {
                'claim_id': ['eval_001', 'eval_002', 'eval_003'],
                'user_id': ['user_001', 'user_002', 'user_003'],
                'claim_object': ['car', 'laptop', 'package'],
                'user_claim': [
                    'Windshield crack',
                    'Screen damage',
                    'Crushed packaging'
                ],
                'image_paths': [
                    'images/eval_car_1.jpg',
                    'images/eval_laptop_1.jpg',
                    'images/eval_package_1.jpg;images/eval_package_2.jpg'
                ],
                'expected_status': ['supported', 'contradicted', 'not_enough_information']
            }
            pd.DataFrame(sample_data).to_csv(sample_claims_path, index=False)
    
    def load_all_data(self):
        """Load all CSV files"""
        try:
            self.claims_df = pd.read_csv(self.dataset_dir / "claims.csv")
            self.user_history_df = pd.read_csv(self.dataset_dir / "user_history.csv")
            self.evidence_requirements_df = pd.read_csv(self.dataset_dir / "evidence_requirements.csv")
            print(f"✓ Loaded {len(self.claims_df)} claims")
            print(f"✓ Loaded {len(self.user_history_df)} user profiles")
            print(f"✓ Loaded {len(self.evidence_requirements_df)} evidence requirements")
        except Exception as e:
            print(f"✗ Error loading datasets: {e}")
    
    def get_user_history(self, user_id: str) -> Optional[Dict]:
        """Get user's historical risk profile"""
        if self.user_history_df is None:
            return None
        
        user_data = self.user_history_df[self.user_history_df['user_id'] == user_id]
        if user_data.empty:
            return None
        
        return user_data.iloc[0].to_dict()
    
    def get_evidence_requirement(self, claim_object: str) -> Optional[Dict]:
        """Get evidence requirements for a claim object type"""
        if self.evidence_requirements_df is None:
            return None
        
        req_data = self.evidence_requirements_df[self.evidence_requirements_df['claim_object'] == claim_object.lower()]
        if req_data.empty:
            return None
        
        return req_data.iloc[0].to_dict()
    
    def get_all_claims(self) -> pd.DataFrame:
        """Get all claims from dataset"""
        return self.claims_df
    
    def get_claim_by_id(self, claim_id: str) -> Optional[Dict]:
        """Get specific claim by ID"""
        if self.claims_df is None:
            return None
        
        claim_data = self.claims_df[self.claims_df['claim_id'] == claim_id]
        if claim_data.empty:
            return None
        
        return claim_data.iloc[0].to_dict()
    
    def get_sample_claims_for_evaluation(self) -> pd.DataFrame:
        """Get sample claims for evaluation"""
        try:
            return pd.read_csv(self.dataset_dir / "sample_claims.csv")
        except FileNotFoundError:
            return None
