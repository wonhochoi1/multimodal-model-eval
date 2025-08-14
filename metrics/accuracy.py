from typing import Dict, List, Any
import difflib

class QualityMetrics:
    """Compute quality metrics for different multimodal tasks"""
    
    @staticmethod
    def compute_wer(trials: List[Dict]) -> Dict[str, float]:
        """Word Error Rate for ASR tasks"""
        total_errors = 0
        total_words = 0
        
        for trial in trials:
            if "output" in trial and "pred" in trial["output"]:
                pred_text = trial["output"]["pred"].get("text", "")
                ref_text = trial.get("refs", {}).get("text", "")
                
                if pred_text and ref_text:
                    pred_words = pred_text.lower().split()
                    ref_words = ref_text.lower().split()
                    
                    # Simple edit distance
                    matcher = difflib.SequenceMatcher(None, ref_words, pred_words)
                    errors = len(ref_words) + len(pred_words) - 2 * matcher.matching_blocks[0].size
                    
                    total_errors += errors
                    total_words += len(ref_words)
        
        wer = total_errors / total_words if total_words > 0 else 1.0
        return {"wer": wer}
    
    @staticmethod
    def compute_topk_accuracy(trials: List[Dict], k: int = 1) -> Dict[str, float]:
        """Top-k accuracy for classification tasks"""
        correct = 0
        total = 0
        
        for trial in trials:
            if "output" in trial and "pred" in trial["output"]:
                pred = trial["output"]["pred"]
                ref_label = trial.get("refs", {}).get("label", "")
                
                if ref_label and "topk" in pred:
                    topk_labels = [item[0] for item in pred["topk"][:k]]
                    if ref_label in topk_labels:
                        correct += 1
                    total += 1
        
        accuracy = correct / total if total > 0 else 0.0
        return {f"top{k}_accuracy": accuracy}
    
    @staticmethod
    def compute_exact_match(trials: List[Dict]) -> Dict[str, float]:
        """Exact match for text generation tasks"""
        correct = 0
        total = 0
        
        for trial in trials:
            if "output" in trial and "pred" in trial["output"]:
                pred_text = trial["output"]["pred"].get("text", "")
                ref_text = trial.get("refs", {}).get("text", "")
                
                if pred_text and ref_text:
                    if pred_text.strip().lower() == ref_text.strip().lower():
                        correct += 1
                    total += 1
        
        exact_match = correct / total if total > 0 else 0.0
        return {"exact_match": exact_match}
    
    # @staticmethod
    # def compute_bleu_score(trials: List[Dict]) -> Dict[str, float]:
    #     """BLEU score for text generation (simplified)"""
    #     # This would use a proper BLEU implementation
    #     # For now, return a placeholder
    #     import re
    #     from collections import Counter
        
    #     def ngram_precision(pred_tokens, ref_tokens, n):
    #         """Calculate n-gram precision"""
    #         if len(pred_tokens) < n:
    #             return 0.0
            
    #         pred_ngrams = Counter()
    #         ref_ngrams = Counter()
            
    #         # Generate n-grams for prediction
    #         for i in range(len(pred_tokens) - n + 1):
    #             ngram = tuple(pred_tokens[i:i+n])
    #             pred_ngrams[ngram] += 1
            
    #         # Generate n-grams for reference
    #         for i in range(len(ref_tokens) - n + 1):
    #             ngram = tuple(ref_tokens[i:i+n])
    #             ref_ngrams[ngram] += 1
            
    #         # Calculate precision
    #         matches = 0
    #         total = sum(pred_ngrams.values())
            
    #         for ngram, count in pred_ngrams.items():
    #             matches += min(count, ref_ngrams.get(ngram, 0))
            
    #         return matches / total if total > 0 else 0.0
        
    #     def brevity_penalty(pred_len, ref_len):
    #         """Calculate brevity penalty"""
    #         if pred_len > ref_len:
    #             return 1.0
    #         elif pred_len == 0:
    #             return 0.0
    #         else:
    #             import math
    #             return math.exp(1 - ref_len / pred_len)
        
    #     total_score = 0.0
    #     count = 0
        
    #     for trial in trials:
    #         if "output" in trial and "pred" in trial["output"]:
    #             pred_text = trial["output"]["pred"].get("text", "")
    #             ref_text = trial.get("refs", {}).get("text", "")
                
    #             if pred_text and ref_text:
    #                 # Tokenize (simple whitespace tokenization)
    #                 pred_tokens = re.findall(r'\w+', pred_text.lower())
    #                 ref_tokens = re.findall(r'\w+', ref_text.lower())
                    
    #                 if pred_tokens and ref_tokens:
    #                     # Calculate n-gram precisions (1-gram to 4-gram)
    #                     precisions = []
    #                     for n in range(1, 5):
    #                         precision = ngram_precision(pred_tokens, ref_tokens, n)
    #                         precisions.append(precision)
                        
    #                     # Calculate geometric mean of precisions
    #                     if all(p > 0 for p in precisions):
    #                         import math
    #                         geo_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
    #                     else:
    #                         geo_mean = 0.0
                        
    #                     # Apply brevity penalty
    #                     bp = brevity_penalty(len(pred_tokens), len(ref_tokens))
    #                     bleu = bp * geo_mean
                        
    #                     total_score += bleu
    #                     count += 1
        
    #     bleu_score = total_score / count if count > 0 else 0.0
    #     return {"bleu_score": bleu_score}