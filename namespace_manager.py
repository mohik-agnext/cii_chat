"""
Namespace Manager

This module provides intelligent namespace filtering to reduce vector search time
by only searching in relevant namespaces based on query content.
"""
import re
from collections import defaultdict

class NamespaceManager:
    def __init__(self, namespaces):
        """
        Initialize the namespace manager.
        
        Args:
            namespaces: List of available namespaces
        """
        self.namespaces = namespaces
        self.namespace_keywords = self._initialize_namespace_keywords()
        print(f"Initialized namespace manager with {len(namespaces)} namespaces")
    
    def _initialize_namespace_keywords(self):
        """Initialize keyword mappings for each namespace"""
        # Map keywords to namespaces
        keyword_map = {
            # EV related namespaces
            "ev3": ["electric", "vehicle", "ev", "charging", "battery", "motor"],
            "electric vehicles6": ["electric", "vehicle", "ev", "charging", "battery", "motor", "transport"],
            
            # Industry related namespaces
            "industry 2015_8": ["industry", "industrial", "manufacturing", "msme", "factory", "production", "business"],
            "policy_industrial_policy_12": ["industry", "industrial", "policy", "msme", "manufacturing", "business"],
            
            # Waste management related namespaces
            "disposal": ["waste", "disposal", "garbage", "trash", "recycle", "pollution"],
            "c&d waste 4": ["waste", "construction", "demolition", "debris", "disposal"],
            
            # Land and property related namespaces
            "policy_parking_policy_11": ["parking", "vehicle", "car", "transport", "space", "urban"],
            "excise2": ["excise", "tax", "duty", "revenue", "alcohol", "license"],
            
            # IT and data related namespaces
            "policy_it_10": ["it", "information", "technology", "software", "digital", "computer"],
            "policy_ites_9": ["it", "ites", "software", "service", "technology", "export"],
            "data sharing 7": ["data", "sharing", "information", "digital", "privacy", "access"],
            
            # Special economic zone
            "sez5": ["sez", "zone", "economic", "special", "export", "business", "tax"]
        }
        
        # Invert the mapping for faster lookup
        namespace_keywords = defaultdict(list)
        for namespace, keywords in keyword_map.items():
            for keyword in keywords:
                namespace_keywords[keyword].append(namespace)
        
        return namespace_keywords
    
    def get_relevant_namespaces(self, query, min_namespaces=3):
        """
        Get relevant namespaces for a query.
        
        Args:
            query: The search query
            min_namespaces: Minimum number of namespaces to return
            
        Returns:
            List of relevant namespaces
        """
        query = query.lower()
        
        # Extract query keywords (words with 3+ characters)
        query_keywords = [word for word in re.findall(r'\b\w{3,}\b', query)]
        
        # Count namespace matches
        namespace_scores = defaultdict(int)
        for keyword in query_keywords:
            for namespace in self.namespaces:
                # Direct namespace match
                if keyword in namespace.lower():
                    namespace_scores[namespace] += 3  # Higher weight for direct namespace match
                
                # Keyword match
                if keyword in self.namespace_keywords:
                    for matched_namespace in self.namespace_keywords[keyword]:
                        if matched_namespace in self.namespaces:
                            namespace_scores[matched_namespace] += 1
        
        # Sort namespaces by score
        sorted_namespaces = sorted(
            [(namespace, score) for namespace, score in namespace_scores.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Get top namespaces
        relevant_namespaces = [ns for ns, score in sorted_namespaces if score > 0]
        
        # If we don't have enough relevant namespaces, add more until we reach min_namespaces
        if len(relevant_namespaces) < min_namespaces:
            # Add remaining namespaces that weren't matched
            remaining = [ns for ns in self.namespaces if ns not in relevant_namespaces]
            relevant_namespaces.extend(remaining[:min_namespaces - len(relevant_namespaces)])
        
        print(f"Query '{query}' matched {len(relevant_namespaces)} namespaces")
        return relevant_namespaces
    
    def classify_query(self, query):
        """
        Classify the query type based on content.
        
        Args:
            query: The search query
            
        Returns:
            Query classification (string)
        """
        query = query.lower()
        
        # Check for question type
        if re.search(r'\bhow\b.*\bapply\b', query) or re.search(r'\bprocess\b', query) or re.search(r'\bsteps\b', query):
            return "process"
        elif re.search(r'\beligib', query) or re.search(r'\bqualif', query) or re.search(r'\bwho can\b', query):
            return "eligibility"
        elif re.search(r'\bfee\b', query) or re.search(r'\bcost\b', query) or re.search(r'\bprice\b', query) or re.search(r'\brate\b', query):
            return "fee"
        elif re.search(r'\bdocument', query) or re.search(r'\brequire', query):
            return "requirements"
        elif re.search(r'\bpolicy\b', query) or re.search(r'\bregulation\b', query) or re.search(r'\blaw\b', query):
            return "policy"
        elif re.search(r'\bcompare\b', query) or re.search(r'\bdifference\b', query) or re.search(r'\bversus\b', query) or re.search(r'\bvs\b', query):
            return "comparison"
        elif len(query.split()) > 10 or re.search(r'\band\b.*\bhow\b', query):
            return "complex"
        else:
            return "general"
