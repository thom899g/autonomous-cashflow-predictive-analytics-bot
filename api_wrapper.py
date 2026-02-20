from typing import Dict, List, Optional
import logging
import requests
from .models.transaction import Transaction

class APIWrapper:
    def __init__(self):
        self.base_url: str