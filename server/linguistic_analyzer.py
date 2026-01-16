"""Linguistic analysis for text completeness."""
import re
from typing import Dict
from dataclasses import dataclass

from .config import config


@dataclass
class LinguisticAnalysis:
    """Result of linguistic analysis."""
    completeness_score: int  # 0-100
    is_question: bool
    is_complete: bool
    word_count: int
    sentence_count: int
    ends_with_continuation: bool
    ends_with_punctuation: bool


class LinguisticAnalyzer:
    """
    Analyzes text for completeness and structure.
    
    Features:
    - Completeness scoring (0-100)
    - Question detection
    - Continuation word detection
    - Sentence counting
    - Punctuation analysis
    """
    
    def __init__(self):
        """Initialize linguistic analyzer."""
        self.continuation_words = set(word.lower() for word in config.continuation_words)
        
        # Question words
        self.question_words = {
            "what", "when", "where", "who", "whom", "whose", "why", "which", "how",
            "is", "are", "was", "were", "do", "does", "did", "can", "could",
            "will", "would", "should", "shall", "may", "might", "must"
        }
    
    def analyze_completeness(self, text: str) -> LinguisticAnalysis:
        """
        Analyze text completeness.
        
        Args:
            text: Text to analyze
        
        Returns:
            LinguisticAnalysis with scores and flags
        """
        if not text or not text.strip():
            return LinguisticAnalysis(
                completeness_score=0,
                is_question=False,
                is_complete=False,
                word_count=0,
                sentence_count=0,
                ends_with_continuation=False,
                ends_with_punctuation=False
            )
        
        text = text.strip()
        
        # Basic metrics
        words = text.split()
        word_count = len(words)
        
        # Check for very short utterances
        if word_count < 3:
            return LinguisticAnalysis(
                completeness_score=20,
                is_question=self.is_question(text),
                is_complete=False,
                word_count=word_count,
                sentence_count=0,
                ends_with_continuation=False,
                ends_with_punctuation=self.ends_with_punctuation(text)
            )
        
        # Analyze features
        ends_with_punct = self.ends_with_punctuation(text)
        ends_with_cont = self.last_word_is_continuation(text)
        is_quest = self.is_question(text)
        sent_count = self.count_sentences(text)
        has_subj_verb = self.has_subject_and_verb(text)
        
        # Calculate completeness score
        score = self._calculate_score(
            word_count=word_count,
            ends_with_punctuation=ends_with_punct,
            ends_with_continuation=ends_with_cont,
            is_question=is_quest,
            sentence_count=sent_count,
            has_subject_verb=has_subj_verb
        )
        
        return LinguisticAnalysis(
            completeness_score=score,
            is_question=is_quest,
            is_complete=score >= 70,
            word_count=word_count,
            sentence_count=sent_count,
            ends_with_continuation=ends_with_cont,
            ends_with_punctuation=ends_with_punct
        )
    
    def _calculate_score(
        self,
        word_count: int,
        ends_with_punctuation: bool,
        ends_with_continuation: bool,
        is_question: bool,
        sentence_count: int,
        has_subject_verb: bool
    ) -> int:
        """
        Calculate completeness score.
        
        Returns:
            Score from 0-100
        """
        score = 0
        
        # Penalty for ending with continuation word
        if ends_with_continuation:
            return 30
        
        # Bonus for ending with punctuation
        if ends_with_punctuation:
            score += 40
        
        # Bonus for having subject and verb
        if has_subject_verb:
            score += 20
        
        # Bonus for complete sentence structure
        if sentence_count >= 1 and ends_with_punctuation:
            score += 30
        
        # Bonus for questions (usually complete)
        if is_question and ends_with_punctuation:
            score += 10
        
        return min(score, 100)
    
    def ends_with_punctuation(self, text: str) -> bool:
        """Check if text ends with sentence-ending punctuation."""
        text = text.strip()
        return len(text) > 0 and text[-1] in '.?!'
    
    def last_word_is_continuation(self, text: str) -> bool:
        """Check if last word is a continuation word."""
        words = text.strip().lower().split()
        if not words:
            return False
        
        last_word = words[-1].rstrip('.,!?;:')
        return last_word in self.continuation_words
    
    def is_question(self, text: str) -> bool:
        """Detect if text is a question."""
        text = text.strip()
        
        # Ends with question mark
        if text.endswith('?'):
            return True
        
        # Starts with question word
        words = text.lower().split()
        if words and words[0] in self.question_words:
            return True
        
        return False
    
    def count_sentences(self, text: str) -> int:
        """Count sentences in text."""
        # Simple sentence counting by punctuation
        count = text.count('.') + text.count('?') + text.count('!')
        
        # If no punctuation but has words, count as 1 sentence
        if count == 0 and text.strip():
            count = 1
        
        return count
    
    def has_subject_and_verb(self, text: str) -> bool:
        """
        Simple heuristic for subject-verb detection.
        
        This is a basic implementation. A more sophisticated version
        would use NLP libraries like spaCy.
        """
        words = text.lower().split()
        
        # Very basic: check if we have at least 2 words and some common verbs
        if len(words) < 2:
            return False
        
        # Common verb forms (very simplified)
        common_verbs = {
            'is', 'are', 'was', 'were', 'am', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did',
            'can', 'could', 'will', 'would', 'should', 'shall',
            'may', 'might', 'must',
            'go', 'goes', 'went', 'going',
            'get', 'gets', 'got', 'getting',
            'make', 'makes', 'made', 'making',
            'know', 'knows', 'knew', 'knowing',
            'think', 'thinks', 'thought', 'thinking',
            'see', 'sees', 'saw', 'seeing',
            'want', 'wants', 'wanted', 'wanting',
            'need', 'needs', 'needed', 'needing'
        }
        
        # Check if any word is a verb
        for word in words:
            clean_word = word.rstrip('.,!?;:')
            if clean_word in common_verbs:
                return True
        
        # If we have 3+ words, assume it has subject-verb
        if len(words) >= 3:
            return True
        
        return False
